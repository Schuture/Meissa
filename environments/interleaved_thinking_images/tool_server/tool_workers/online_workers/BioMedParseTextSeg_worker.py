import os
import sys

# Must set matplotlib backend before importing pyplot
os.environ["MPLBACKEND"] = "Agg"

# If you really want offline, keep these; but note: BioMedParse may still need HF assets
os.environ["HF_DATASETS_OFFLINE"] = os.environ.get("HF_DATASETS_OFFLINE", "1")
os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# OpenThinkIMG root + tool_server
_TOOL_WORKERS_DIR = os.path.dirname(_THIS_DIR)
_TOOL_SERVER_DIR = os.path.dirname(_TOOL_WORKERS_DIR)
_OPENTHINKIMG_ROOT = os.path.dirname(_TOOL_SERVER_DIR)
for p in [_OPENTHINKIMG_ROOT, _TOOL_SERVER_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# BiomedParse root
_BIOMEDPARSE_ROOT = os.environ.get("BIOMEDPARSE_ROOT", "${PROJECT_ROOT}/BiomedParse")
if os.path.isdir(_BIOMEDPARSE_ROOT) and _BIOMEDPARSE_ROOT not in sys.path:
    sys.path.insert(0, _BIOMEDPARSE_ROOT)

import uuid
import re
import argparse
import torch
import numpy as np
from PIL import Image

# Flush prints
import functools
print = functools.partial(print, flush=True)

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- BioMedParse imports ----
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image

GB = 1 << 30
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
np.random.seed(3)


def _parse_prompts(param: str):
    """
    Accept formats:
      - "a; b; c"
      - "a, b"
      - ["a","b"]
      - newline separated
    Return list[str]
    """
    if param is None:
        return []
    s = str(param).strip()
    if not s:
        return []

    # JSON-ish list: ["a","b"]
    if s.startswith("[") and s.endswith("]"):
        items = re.findall(r"['\"]([^'\"]+)['\"]", s)
        if items:
            return [x.strip() for x in items if x.strip()]

    parts = re.split(r"[\n;,\uff0c]+", s)
    prompts = [p.strip() for p in parts if p.strip()]
    return prompts


def _to_numpy_mask(m):
    """
    Convert possible mask types into numpy array.
    Accept:
      - numpy array
      - torch tensor
      - PIL Image
      - list
    Return numpy array (no resize yet).
    """
    if m is None:
        return None

    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().float().numpy()

    if isinstance(m, Image.Image):
        m = np.array(m)

    if isinstance(m, (list, tuple)):
        m = np.array(m)

    if not isinstance(m, np.ndarray):
        # fallback
        try:
            m = np.array(m)
        except Exception:
            return None

    return m


def _normalize_mask_to_hw(m, H, W):
    """
    Convert to float32 (H, W) in [0,1] (best effort).
    Handles shapes like:
      (H,W), (1,H,W), (H,W,1), (C,H,W), (H,W,C)
    """
    arr = _to_numpy_mask(m)
    if arr is None:
        return None

    # squeeze trivial dims
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    # If still 3D, try to reduce to 2D
    if arr.ndim == 3:
        # common cases: (H,W,C) or (C,H,W)
        if arr.shape[0] in (1, 3) and arr.shape[1] == H and arr.shape[2] == W:
            # (C,H,W) -> take first channel
            arr = arr[0]
        elif arr.shape[-1] in (1, 3) and arr.shape[0] == H and arr.shape[1] == W:
            # (H,W,C) -> take first channel
            arr = arr[..., 0]
        else:
            # unknown 3D, just take first slice
            arr = arr[..., 0]

    if arr.ndim != 2:
        # last resort
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            return None

    # Convert type to float
    if arr.dtype == np.bool_:
        arr = arr.astype(np.float32)
    elif np.issubdtype(arr.dtype, np.integer):
        # Often 0/255 or 0/1; map to [0,1]
        maxv = float(arr.max()) if arr.size > 0 else 1.0
        if maxv > 1.0:
            arr = arr.astype(np.float32) / maxv
        else:
            arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32)

        # sometimes logits or arbitrary float, clamp a bit
        # if values seem like 0..255, normalize
        maxv = float(np.nanmax(arr)) if arr.size > 0 else 1.0
        if maxv > 1.5:
            arr = arr / maxv
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)

    # Resize to (H,W) if mismatch
    if arr.shape != (H, W):
        # use PIL nearest for hard mask-ish, but we keep float
        pil = Image.fromarray((arr * 255.0).astype(np.uint8))
        pil = pil.resize((W, H), resample=Image.NEAREST)
        arr = (np.array(pil).astype(np.float32) / 255.0)

    return arr


def _hex_color(rgb01):
    r, g, b = rgb01
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _get_color_list(n):
    """
    Deterministic distinct colors using matplotlib tab10/tab20.
    Return list of (r,g,b) in 0..1
    """
    if n <= 10:
        cmap = matplotlib.cm.get_cmap("tab10", 10)
    else:
        cmap = matplotlib.cm.get_cmap("tab20", 20)
    colors = []
    for i in range(n):
        c = cmap(i % cmap.N)
        colors.append((float(c[0]), float(c[1]), float(c[2])))
    return colors


def _overlay_masks_on_image(pil_img: Image.Image, masks, prompts, thr=0.5, alpha=0.45):
    """
    Multi-class overlay with per-prompt color + legend.
    Returns: (edited_img_pil, colors_hex_list)
    """
    img = pil_img.convert("RGB")
    W, H = img.size

    # normalize masks to (H,W) float
    norm_masks = []
    for m in masks:
        nm = _normalize_mask_to_hw(m, H, W)
        norm_masks.append(nm)

    n = len(prompts)
    colors = _get_color_list(max(n, 1))
    colors_hex = [_hex_color(colors[i]) for i in range(n)]

    fig, ax = plt.subplots(figsize=(W / 200.0, H / 200.0), dpi=200)
    ax.imshow(img)

    # Overlay each mask with its color
    for i in range(n):
        nm = norm_masks[i] if i < len(norm_masks) else None
        if nm is None:
            continue
        binm = (nm > thr).astype(np.float32)
        if binm.sum() <= 0:
            continue

        r, g, b = colors[i]
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[..., 0] = r
        overlay[..., 1] = g
        overlay[..., 2] = b
        overlay[..., 3] = binm * alpha
        ax.imshow(overlay)

    # Legend: color block + label
    handles = []
    for i in range(n):
        label = prompts[i]
        handles.append(Patch(facecolor=colors[i], edgecolor="black", label=f"{i+1}: {label}"))

    # Put legend inside top-left with semi-transparent frame
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        framealpha=0.7,
        fontsize=8,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4,
    )
    for text in leg.get_texts():
        text.set_color("black")

    ax.axis("off")

    import io
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    return out, colors_hex, norm_masks


class BioMedParseToolWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="BioMedParseTextSeg",
        model_path="",
        model_base="",
        load_8bit=False,
        load_4bit=False,
        device="",
        limit_model_concurrency=1,
        host="0.0.0.0",
        port=None,
        model_semaphore=None,
        biomedparse_ckpt="${DATA_ROOT}/biomedparse_weights/biomedparse_v1.pt",
        biomedparse_cfg="configs/biomedparse_inference.yaml",
        biomedparse_repo_root="${PROJECT_ROOT}/BiomedParse",
    ):
        self.biomedparse_ckpt = biomedparse_ckpt
        self.biomedparse_cfg = biomedparse_cfg
        self.biomedparse_repo_root = biomedparse_repo_root
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
        )

    def init_model(self):
        print("DEBUG: Entering init_model...")

        if self.biomedparse_repo_root and os.path.isdir(self.biomedparse_repo_root):
            os.chdir(self.biomedparse_repo_root)
            if self.biomedparse_repo_root not in sys.path:
                sys.path.insert(0, self.biomedparse_repo_root)

        logger.info(f"Initializing model {self.model_name}...")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")
        print(f"DEBUG: Using device: {self.device}")

        print(f"DEBUG: Loading config from {self.biomedparse_cfg} ...")
        opt = load_opt_from_config_files([self.biomedparse_cfg])

        # Ensure opt has device & disable distributed
        if isinstance(opt, dict):
            opt["device"] = self.device
            opt["distributed"] = False
            opt["rank"] = 0
            opt["local_rank"] = 0
            opt["world_size"] = 1
        else:
            for k, v in [("device", self.device), ("distributed", False), ("rank", 0), ("local_rank", 0), ("world_size", 1)]:
                try:
                    setattr(opt, k, v)
                except Exception:
                    pass

        print("DEBUG: Building model structure...")
        model_struct = build_model(opt)

        print("DEBUG: Initializing BaseModel wrapper...")
        base_model = BaseModel(opt, model_struct)

        print(f"DEBUG: Loading weights from {self.biomedparse_ckpt} ...")
        self.model = base_model.from_pretrained(self.biomedparse_ckpt).eval().to(self.device)

        print("DEBUG: Pre-computing text embeddings...")
        with torch.no_grad():
            self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                BIOMED_CLASSES + ["background"],
                is_eval=True,
            )

        logger.info("BioMedParse model initialized.")
        print("DEBUG: BioMedParse model initialization COMPLETE.")

    def generate(self, params):
        generate_param = params.get("param", None)
        image_b64 = params.get("image", None)

        if generate_param is None or image_b64 is None:
            return {"text": "Missing inputs: require both 'image' and 'param'.", "edited_image": None, "error_code": 1}

        ret = {"text": "", "error_code": 0}

        try:
            prompts = _parse_prompts(generate_param)
            if len(prompts) == 0:
                ret["text"] = "Empty prompts after parsing 'param'. Please pass semicolon-separated short noun phrases."
                ret["edited_image"] = None
                ret["error_code"] = 1
                return ret

            image = base64_to_pil(image_b64).convert("RGB")
            W, H = image.size

            with torch.no_grad():
                pred_masks = interactive_infer_image(self.model, image, prompts)

            # Ensure list-like
            if pred_masks is None:
                raise RuntimeError("interactive_infer_image returned None")
            if not isinstance(pred_masks, (list, tuple)):
                # some implementations may return a single mask for single prompt
                pred_masks = [pred_masks]

            # If length mismatch, we still handle safely
            edited_img, colors_hex, norm_masks = _overlay_masks_on_image(
                image, pred_masks, prompts, thr=0.5, alpha=0.45
            )

            stats = []
            color_stats = []
            for i, p in enumerate(prompts):
                nm = norm_masks[i] if i < len(norm_masks) else None
                if nm is None:
                    area = 0
                else:
                    area = int((nm > 0.5).sum())
                stats.append(f"{i+1}:{p} area={area}")
                color_stats.append(f"{i+1}={colors_hex[i]}")

            ret["text"] = "BioMedParse segmentation completed. " + " | ".join(stats) + " | colors: " + ", ".join(color_stats)
            ret["edited_image"] = pil_to_base64(edited_img)

        except Exception as e:
            logger.exception(f"BioMedParse generate error: {e}")
            ret["text"] = f"Error: {e}"
            ret["edited_image"] = None
            ret["error_code"] = 1

        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20037)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")

    parser.add_argument("--biomedparse_ckpt", type=str, default="${DATA_ROOT}/biomedparse_weights/biomedparse_v1.pt")
    parser.add_argument("--biomedparse_cfg", type=str, default="configs/biomedparse_inference.yaml")
    parser.add_argument("--biomedparse_repo_root", type=str, default="${PROJECT_ROOT}/BiomedParse")

    args = parser.parse_args()
    print(f"DEBUG: Worker script started. Connecting to {args.controller_address}")

    worker = BioMedParseToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        no_register=args.no_register,
        biomedparse_ckpt=args.biomedparse_ckpt,
        biomedparse_cfg=args.biomedparse_cfg,
        biomedparse_repo_root=args.biomedparse_repo_root,
    )
    worker.run()
