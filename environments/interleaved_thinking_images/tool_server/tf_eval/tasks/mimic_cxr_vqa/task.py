from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger

import os
import re
import json
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image 
from datasets import Dataset

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

# ==========================================
# Configuration for Resizing
# ==========================================
TARGET_SIZE = (1024, 1024)
CACHE_SUFFIX = "_1024.jpg" 

# ----------------------------
# Helpers: load json list
# ----------------------------
def _load_json_list(file_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {file_path}, got {type(data)}")

    random.seed(0)
    random.shuffle(data)

    if num_samples is None:
        return data
    return data[:num_samples]


def load_dataset(file_path: str, num_samples: Optional[int] = None) -> Dataset:
    data = _load_json_list(file_path, num_samples)
    return Dataset.from_dict({"data": data})


# ----------------------------
# Evaluation Logic Classes
# ----------------------------
class TextUtils:
    _PUNCT_RE = re.compile(r"[^a-z0-9\s]")

    @staticmethod
    def safe_to_text(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, (list, dict)):
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                pass
        return str(x)

    @classmethod
    def normalize_text(cls, s: str, remove_fluff: bool = False) -> str:
        s = (s or "").lower().strip()
        s = s.replace("_", " ").replace("-", " ").replace("/", " ")

        replacements = {
            "picc line": "picc",
            "peripherally inserted central catheter": "picc",
            "endotracheal tube": "ett",
            "enteric tube": "ng tube",
            "nasogastric tube": "ng tube",
            "orogastric tube": "og tube",
            "cardiac pacemaker": "pacemaker",
            "cardiac pacer": "pacemaker",
            "pacer": "pacemaker",
            "wires": "wire",
            "leads": "wire",
            "elevation": "elevated",
            "opacities": "opacity",
            "effusions": "effusion",
            "abnormalities": "abnormality",
            "structures": "structure",
        }

        for k, v in replacements.items():
            if k in s:
                s = s.replace(k, v)

        s = cls._PUNCT_RE.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()

        if remove_fluff:
            fluff = [
                "present", "seen", "visible", "identified", "noted", "evidence of",
                "demonstrated", "observed", "cannot determine", "suspicious for",
                "consistent with", "suggestive of", "findings of"
            ]
            for w in fluff:
                s = s.replace(w, " ")
            s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def normalize_yesno(text: str) -> Optional[str]:
        s = TextUtils.normalize_text(text, remove_fluff=True)
        if s in ["yes", "no"]:
            return s
        if re.search(r"\byes\b", s):
            return "yes"
        if re.search(r"\bno\b", s):
            return "no"
        return None

    @staticmethod
    def parse_terms(s: str) -> Set[str]:
        s = TextUtils.normalize_text(s, remove_fluff=True)
        if not s:
            return set()
        parts = re.split(r"\s*(?:,|;|\band\b|\bor\b|\.|\n)\s*", s)
        out: Set[str] = set()
        for p in parts:
            p = p.strip()
            if p:
                out.add(p)
        return out

    @staticmethod
    def is_negative_finding(text: str) -> bool:
        text = (text or "").lower()
        if text.strip() in ["no", "none", "normal", "negative"]:
            return True
        clean_text = re.sub(r"[^a-z]", "", text)
        if clean_text in ["none", "non", "nothing", "normal"]:
            return True
        strong_negatives = ["no acute", "no active", "no significant", "lungs are clear", "heart is normal"]
        for sn in strong_negatives:
            if sn in text:
                return True
        return False


class Evaluator:
    @staticmethod
    def extract_final_answer(text: str) -> str:
        if not text:
            return ""
        t = str(text).strip()
        if "Final Answer:" in t:
            try:
                t = t.split("Final Answer:", 1)[1].strip()
            except:
                pass
        
        patterns = [
            r"\[FINAL\]\s*(.+?)\s*$",
            r"\*\*Final Answer:?\*\*\s*(.+?)\s*$",
            r"Final Answer:\s*(.+?)\s*$",
            r"\[DECISION\]\s*(.+?)\s*$",
        ]
        for pat in patterns:
            m = re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if m:
                return m.group(1).strip()
        
        if t.endswith("."):
            t = t[:-1]
        return t

    @staticmethod
    def get_gt_list(sample: Dict[str, Any]) -> Tuple[List[str], bool]:
        raw = None
        for k in ["gt_answer_norm", "answer", "answers", "gt_answer", "label"]:
            if k in sample:
                raw = sample[k]
                break

        is_empty_list = False
        if isinstance(raw, list) and len(raw) == 0:
            is_empty_list = True
        if raw is None:
            is_empty_list = True

        norm_list: List[str] = []
        if isinstance(raw, list):
            for x in raw:
                s = TextUtils.normalize_text(TextUtils.safe_to_text(x))
                if s:
                    norm_list.append(s)
        else:
            s = TextUtils.normalize_text(TextUtils.safe_to_text(raw))
            if s:
                norm_list.append(s)

        if is_empty_list and not norm_list:
            return ["none"], True

        return norm_list, False

    @staticmethod
    def _is_soft_match(gt_term: str, pred_term: str) -> bool:
        gt_term = (gt_term or "").strip()
        pred_term = (pred_term or "").strip()
        if not gt_term or not pred_term:
            return False

        gt_words = set(gt_term.split())
        pred_words = set(pred_term.split())

        if gt_term in pred_term:
            return True
        if gt_words.issubset(pred_words):
            return True

        intersection = gt_words.intersection(pred_words)
        valid_overlap = [w for w in intersection if len(w) > 2 and w not in ["and", "with", "the"]]

        if len(valid_overlap) >= 2:
            return True
        if len(gt_words) > 0 and len(valid_overlap) / len(gt_words) >= 0.5:
            return True
        if len(gt_words) == 1 and list(gt_words)[0] in pred_words:
            return True
        return False

    @staticmethod
    def check_correctness(pred_text: str, gt_list: List[str]) -> bool:
        if not gt_list:
            return False

        pred_final = Evaluator.extract_final_answer(pred_text)
        pred_norm_str = TextUtils.normalize_text(pred_final, remove_fluff=True)
        pred_terms = TextUtils.parse_terms(pred_final)
        gt_set = set(gt_list)

        if gt_set == {"none"}:
            if TextUtils.is_negative_finding(pred_norm_str):
                return True
            if TextUtils.is_negative_finding(pred_final):
                return True
            if not pred_norm_str:
                return True
            return False

        if gt_set.issubset({"yes", "no"}):
            p_yn = TextUtils.normalize_yesno(pred_final)
            if p_yn:
                return p_yn in gt_set
            return False

        if TextUtils.is_negative_finding(pred_norm_str) and gt_set != {"none"}:
            return False

        match_count = 0
        for g in gt_set:
            if g in pred_norm_str:
                match_count += 1
                continue

            found_in_terms = False
            for p_term in pred_terms:
                if Evaluator._is_soft_match(g, p_term):
                    found_in_terms = True
                    break
            if found_in_terms:
                match_count += 1
                continue

        return match_count > 0


# ----------------------------
# tf_eval interface
# ----------------------------
def load_data_function():
    dataset_path = task_config["dataset_path"]
    num_samples = task_config.get("num_sample", None)
    
    image_root = task_config.get(
        "image_root",
        "${DATA_ROOT}/mimic-cxr-jpg/2.1.0/files",
    ).rstrip("/")

    parent_dir = os.path.dirname(image_root) # .../mimic-cxr-jpg/2.1.0
    cache_root = os.path.join(parent_dir, "files_resized_1024")
    
    if not os.path.exists(cache_root):
        try:
            os.makedirs(cache_root, exist_ok=True)
            logger.info(f"Created cache directory at {cache_root}")
        except Exception as e:
            logger.warning(f"Failed to create cache dir {cache_root}, will try to use original images. Error: {e}")

    ds = load_dataset(dataset_path, num_samples)

    meta_data: List[Dict[str, Any]] = []
    skipped = 0
    resized_count = 0

    for i, row in enumerate(ds):
        item = row["data"]

        orig_idx = item.get("idx", i)
        split = str(item.get("split", "")).strip() or "unknown"
        image_id = str(item.get("image_id", "")).strip() or "noimageid"
        item_id = f"mimic_cxr_vqa_{split}_{orig_idx}_{image_id}"

        q = str(item.get("question", "")).strip()
        if not q:
            skipped += 1
            continue

        rel_img = item.get("image_path", None)
        if rel_img is None or (isinstance(rel_img, str) and len(rel_img.strip()) == 0):
            skipped += 1
            continue

        rel_img = str(rel_img).lstrip("/")
        
        if os.path.isabs(item.get("image_path")):
             abs_img_orig = item.get("image_path")
        else:
             abs_img_orig = os.path.join(image_root, rel_img)

        abs_img_target = os.path.join(cache_root, rel_img)

        final_img_path = abs_img_orig

        if os.path.exists(abs_img_orig):
            if os.path.exists(abs_img_target):
                final_img_path = abs_img_target
            else:
                try:
                    target_dir = os.path.dirname(abs_img_target)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    with Image.open(abs_img_orig) as img:
                        img = img.convert('RGB')
                        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                        img.save(abs_img_target, quality=95)
                    
                    final_img_path = abs_img_target
                    resized_count += 1
                    if resized_count % 100 == 0:
                        logger.info(f"Resized {resized_count} images so far...")
                        
                except Exception as e:
                    logger.warning(f"Failed to resize {abs_img_orig}: {e}. Using original.")
                    final_img_path = abs_img_orig
        else:
            skipped += 1
            continue

        gold_raw = item.get("answer", [])
        gt_list, _ = Evaluator.get_gt_list({"answer": gold_raw})

        meta = {
            "idx": item_id,
            "text": q,
            "label": gold_raw,
            "gt_answer_norm": gt_list,
            "image_path": final_img_path,
            "split": split,
            "orig_idx": orig_idx,
            "subject_id": item.get("subject_id", None),
            "study_id": item.get("study_id", None),
            "image_id": item.get("image_id", None),
        }
        meta_data.append(meta)

    logger.info(f"Total data number: {len(meta_data)} (skipped={skipped}). Newly resized: {resized_count}.")
    return meta_data


def evaluate_function(results, meta_data):
    """
    Standard evaluate function.
    """
    results_dict = {res["idx"]: res for res in results}
    meta_dict = {m["idx"]: m for m in meta_data}

    scores: List[float] = []
    compare_logs: List[Dict[str, Any]] = []

    for idx, meta in meta_dict.items():
        pred_raw = results_dict.get(idx, {}).get("results", {}).get("final_answer", "")
        pred_final = Evaluator.extract_final_answer(pred_raw)

        gt_norm = meta.get("gt_answer_norm", None)
        if gt_norm is None:
            gt_norm, _ = Evaluator.get_gt_list({"answer": meta.get("label", [])})

        ok = Evaluator.check_correctness(pred_final, gt_norm)
        score = 1.0 if ok else 0.0

        scores.append(score)
        compare_logs.append(
            {
                "idx": idx,
                "orig_idx": meta.get("orig_idx", None),
                "question": meta.get("text", ""),
                "gold": str(gt_norm),
                "pred": pred_final,
                "gold_raw": meta.get("label", None),
                "pred_raw": str(pred_raw),
                "score": score,
            }
        )

        meta["prediction"] = pred_raw

    acc = sum(scores) / len(scores) if len(scores) > 0 else 0.0
    return {
        "Acc": acc,
        "compare_logs": compare_logs,
        "results": results,
        "meta_data": meta_data,
    }