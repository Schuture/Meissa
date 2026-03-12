from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger

import os
import json
import re
import random
import string
from datasets import Dataset

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

import math
from PIL import Image


def _safe_open_rgb(p: str):
    im = Image.open(p)
    return im.convert("RGB")

def _make_composite(image_paths, out_path, max_side=1400, pad=10):
    """
    Make a grid composite from multiple images.
    - keep aspect ratios
    - resize each tile so the longest side <= max_side
    - grid size = ceil(sqrt(n)) x ceil(n/cols)
    """
    ims = []
    for p in image_paths:
        if p is None or (not os.path.exists(p)):
            continue
        try:
            im = _safe_open_rgb(p)
            # downscale per-tile
            w, h = im.size
            scale = min(1.0, float(max_side) / float(max(w, h)))
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)))
            ims.append(im)
        except Exception:
            continue

    if len(ims) == 0:
        return False

    if len(ims) == 1:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ims[0].save(out_path)
        return True

    n = len(ims)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    # make all tiles same size (use max tile w/h)
    max_w = max(im.size[0] for im in ims)
    max_h = max(im.size[1] for im in ims)

    canvas_w = cols * max_w + (cols + 1) * pad
    canvas_h = rows * max_h + (rows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    for k, im in enumerate(ims):
        r = k // cols
        c = k % cols
        x = pad + c * (max_w + pad)
        y = pad + r * (max_h + pad)

        # center the tile inside its cell
        iw, ih = im.size
        ox = x + (max_w - iw) // 2
        oy = y + (max_h - ih) // 2
        canvas.paste(im, (ox, oy))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    canvas.save(tmp, format="PNG")
    os.replace(tmp, out_path)

    return True


def _basic_norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("&", "and")
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _read_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_dataset(file_path, num_samples=None, seed=42):
    data = _read_jsonl(file_path)

    # fixed shuffle (optional, but keep deterministic if num_sample used)
    rng = random.Random(int(seed))
    rng.shuffle(data)

    if num_samples is None:
        return Dataset.from_dict({"data": data})
    n = min(int(num_samples), len(data))
    return Dataset.from_dict({"data": data[:n]})


def _format_mc_question(question_text: str, options: dict):
    """
    MedXpertQA's question field already contains answer choices formatted,
    but we still add a hard instruction to answer with a letter only.
    """
    q = str(question_text).strip()

    # also add an explicit options block (robust if question formatting varies)
    lines = [q, ""]
    # for k in ["A", "B", "C", "D", "E"]:
    #     if isinstance(options, dict) and k in options:
    #         lines.append(f"{k}) {str(options[k]).strip()}")
    # lines.append("")
    lines.append("""You may use tools (ZoomInSubfigure / SegmentRegionAroundPoint / BioMedParseTextSeg) if they help.
                In the end, your final answer in Terminate(ans=...) must be a single letter: A, B, C, D, or E.""")
    lines.append("The image may contain multiple panels. Use all panels to answer.")

    return "\n".join(lines)


def load_data_function():
    dataset_path = task_config["dataset_path"]
    image_root = task_config["image_root"]
    num_samples = task_config.get("num_sample", None)
    seed = task_config.get("seed", 42)

    ds = load_dataset(dataset_path, num_samples=num_samples, seed=seed)

    meta_data = []
    for i, row in enumerate(ds):
        item = row["data"]

        qid = str(item.get("id", f"MM_{i}")).strip()
        item_id = f"medxpertqa_mm_{qid}"

        question = item.get("question", "")
        options = item.get("options", {})
        label = str(item.get("label", "")).strip().upper()

        # build text prompt
        text = _format_mc_question(question, options)

        # image(s)
        imgs = item.get("images", [])
        if imgs is None:
            imgs = []
        if isinstance(imgs, str):
            imgs = [imgs]

        # For OpenThinkIMG's BaseEvalDataset/getitem_fn, meta_data should contain one image_path.
        # If multiple images exist, use the first one for now.
        imgs = item.get("images", [])
        if imgs is None:
            imgs = []
        if isinstance(imgs, str):
            imgs = [imgs]

        abs_imgs = [os.path.join(image_root, x) for x in imgs]

        # build a composite image if multiple images exist
        cache_dir = os.path.join(os.path.dirname(dataset_path), "cache_images_mm")
        os.makedirs(cache_dir, exist_ok=True)

        composite_path = os.path.join(cache_dir, f"{item_id}.png")

        # if cache exists but corrupted, delete it
        if os.path.exists(composite_path):
            try:
                t = Image.open(composite_path)
                t.verify()
            except Exception:
                try:
                    os.remove(composite_path)
                except Exception:
                    pass

        if not os.path.exists(composite_path):
            ok = _make_composite(abs_imgs, composite_path)
            if not ok:
                logger.warning(f"Failed to build composite for {item_id}, images={imgs}")
                continue

        image_path = composite_path

        meta = {
            "idx": item_id,
            "text": text,
            "label": label,
            "image_path": image_path,

            # extra fields for analysis
            "id": qid,
            "images": imgs,
            "medical_task": item.get("medical_task", None),
            "body_system": item.get("body_system", None),
            "question_type": item.get("question_type", None),
            "options": options,
        }
        meta_data.append(meta)

    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def evaluate_function(results, meta_data):
    """
    Multi-choice accuracy (A-E).
    Extract predicted letter if possible; otherwise try mapping option text.
    """
    def _extract_letter(s: str):
        if s is None:
            return ""
        t = str(s).strip().upper()

        # common patterns: "A", "(A)", "Answer: A", "Option A", "A."
        m = re.search(r"\b([A-E])\b", t)
        if m:
            return m.group(1)

        m = re.search(r"\bOPTION\s*([A-E])\b", t)
        if m:
            return m.group(1)

        m = re.search(r"\bANSWER\s*[:\-]?\s*([A-E])\b", t)
        if m:
            return m.group(1)

        return ""

    # reverse map: normalized option text -> letter
    def _build_opt_rev(options: dict):
        rev = {}
        if not isinstance(options, dict):
            return rev
        for k, v in options.items():
            kk = str(k).strip().upper()
            if kk not in ["A", "B", "C", "D", "E"]:
                continue
            rev[_basic_norm(v)] = kk
        return rev

    results_dict = {res["idx"]: res for res in results}
    meta_dict = {m["idx"]: m for m in meta_data}

    scores = []
    compare_logs = []

    for idx, meta in meta_dict.items():
        pred_raw = results_dict.get(idx, {}).get("results", {}).get("final_answer", "")
        gold = str(meta.get("label", "")).strip().upper()

        pred_letter = _extract_letter(pred_raw)

        reason = "letter"
        if pred_letter == "":
            # try map from option text
            rev = _build_opt_rev(meta.get("options", {}))
            pred_norm = _basic_norm(pred_raw)
            pred_letter = rev.get(pred_norm, "")
            reason = "option_text" if pred_letter else "unparsed"

        score = 1.0 if pred_letter == gold and pred_letter != "" else 0.0
        scores.append(score)

        compare_logs.append({
            "idx": idx,
            "gold": gold,
            "pred": pred_letter if pred_letter != "" else _basic_norm(pred_raw),
            "pred_raw": str(pred_raw),
            "score": float(score),
            "reason": reason,
        })

        meta["prediction"] = pred_raw

    acc = sum(scores) / len(scores) if len(scores) > 0 else 0.0
    return {
        "Acc": acc,
        "compare_logs": compare_logs,
        "results": results,
        "meta_data": meta_data,
    }
