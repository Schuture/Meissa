from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger

import os
import json
import re
import random
import string
import glob
from datasets import Dataset

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)


def _basic_norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("&", "and")
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_all_openaccess(dataset_root: str, qa_dir_rel: str):
    qa_dir = os.path.join(dataset_root, qa_dir_rel)
    if not os.path.isdir(qa_dir):
        raise ValueError(f"qa_dir not found: {qa_dir}")

    json_paths = sorted(glob.glob(os.path.join(qa_dir, "*.json")))
    if len(json_paths) == 0:
        raise ValueError(f"No json files found under: {qa_dir}")

    all_items = []
    for p in json_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning(f"Skip non-list json: {p}")
            continue
        all_items.extend(data)

    return all_items, json_paths


def load_data_function():
    dataset_root = task_config["dataset_root"]
    qa_dir_rel = task_config.get("qa_dir", "QA_information/Open-access")
    seed = int(task_config.get("seed", 42))
    num_samples = task_config.get("num_sample", None)
    if num_samples is not None:
        num_samples = int(num_samples)

    all_items, json_paths = _load_all_openaccess(dataset_root, qa_dir_rel)
    logger.info(f"Loaded {len(all_items)} QA pairs from {len(json_paths)} open-access json files.")

    # fixed shuffle
    rng = random.Random(seed)
    rng.shuffle(all_items)

    if num_samples is not None:
        all_items = all_items[:num_samples]

    meta_data = []
    image_root = dataset_root  # because json image_path is like "Images/ACRIMA/xxx.png"

    for i, item in enumerate(all_items):
        item_id = f"omnimedvqa_{i}"

        rel_img = str(item.get("image_path", "")).strip()
        if len(rel_img) == 0:
            continue

        image_path = os.path.join(image_root, rel_img)
        if not os.path.exists(image_path):
            logger.warning(f"Missing image: {image_path}")
            continue

        question = str(item.get("question", "")).strip()
        gt = str(item.get("gt_answer", "")).strip()

        # options (most OmniMedVQA entries are multi-choice with A-D)
        optA = item.get("option_A", None)
        optB = item.get("option_B", None)
        optC = item.get("option_C", None)
        optD = item.get("option_D", None)

        # If options exist, format as multiple-choice and force a concise answer.
        if all(v is not None and str(v).strip() != "" for v in [optA, optB, optC, optD]):
            text = (
                f"{question}\n"
                f"A) {str(optA).strip()}\n"
                f"B) {str(optB).strip()}\n"
                f"C) {str(optC).strip()}\n"
                f"D) {str(optD).strip()}\n"
                f"Answer with the option letter (A, B, C, or D) only."
            )
        else:
            # fallback for rare non-mc entries
            text = question

        meta = {
            "idx": item_id,
            "text": text,
            "label": gt,
            "image_path": image_path,

            # keep rich fields for debugging/analysis
            "question_id": item.get("question_id", None),
            "dataset": item.get("dataset", None),
            "question_type": item.get("question_type", None),
            "modality_type": item.get("modality_type", None),
            "rel_image_path": rel_img,

            "option_A": optA,
            "option_B": optB,
            "option_C": optC,
            "option_D": optD,
        }
        meta_data.append(meta)

    logger.info(f"Total data number (after filtering missing images): {len(meta_data)}")
    return meta_data


def evaluate_function(results, meta_data):
    """
    OmniMedVQA open-access evaluation:
      - Parse model output to an option (A/B/C/D) if possible
      - Otherwise treat it as free-form and match against gt_answer and options
      - normalized exact match + substring + fuzzy (optional)
    """
    try:
        from thefuzz import fuzz
        _HAS_FUZZ = True
    except Exception:
        _HAS_FUZZ = False

    FUZZY_TH = 80

    def _norm_yesno(s: str):
        if s is None:
            return None
        t = _basic_norm(s)

        # strong signals
        if re.search(r"\byes\b", t):
            return "yes"
        if re.search(r"\bno\b", t):
            return "no"

        # common variants
        if re.search(r"\btrue\b", t):
            return "yes"
        if re.search(r"\bfalse\b", t):
            return "no"

        # "normal/abnormal" style answers (often correspond to no/yes anomaly)
        # If question is phrased as "Is there any anomaly/abnormality?"
        # "normal" ~= "no anomaly"; "abnormal" ~= "yes"
        if re.search(r"\b(normal|no anomaly|no abnormal)\b", t):
            return "no"
        if re.search(r"\b(abnormal|anomaly|abnormality present|lesion|fracture)\b", t):
            return "yes"

        return None

    def _extract_choice_letter(s: str):
        if s is None:
            return None
        t = str(s).strip().lower()

        # common formats: "A", "(A)", "Answer: A", "Option A", "A."
        m = re.search(r"\b([abcd])\b", t)
        if m:
            return m.group(1).upper()

        m = re.search(r"\boption\s*([abcd])\b", t)
        if m:
            return m.group(1).upper()

        m = re.search(r"\banswer\s*[:\-]?\s*([abcd])\b", t)
        if m:
            return m.group(1).upper()

        return None

    def _option_text(meta, letter: str):
        if letter == "A":
            return meta.get("option_A", None)
        if letter == "B":
            return meta.get("option_B", None)
        if letter == "C":
            return meta.get("option_C", None)
        if letter == "D":
            return meta.get("option_D", None)
        return None

    def _match(pred_raw: str, meta):
        gold_raw = meta.get("label", "")

        # yes/no style shortcut
        gold_yn = _norm_yesno(gold_raw)
        pred_yn = _norm_yesno(pred_raw)
        if gold_yn is not None and pred_yn is not None:
            if gold_yn == pred_yn:
                return 1.0, pred_yn, gold_yn, "yesno"
            else:
                return 0.0, pred_yn, gold_yn, "yesno_mismatch"

        gold = _basic_norm(gold_raw)
        pred_norm = _basic_norm(pred_raw)

        # if model returns a letter, map to option text
        letter = _extract_choice_letter(pred_raw)
        if letter is not None:
            opt_txt = _option_text(meta, letter)
            if opt_txt is not None:
                pred_norm = _basic_norm(opt_txt)

        # exact
        if pred_norm != "" and pred_norm == gold:
            return 1.0, pred_norm, gold, "exact"

        # match against options too (model may output option text even if gt is same)
        opts = []
        for k in ["option_A", "option_B", "option_C", "option_D"]:
            v = meta.get(k, None)
            if v is not None and str(v).strip() != "":
                opts.append(_basic_norm(v))

        # substring
        if pred_norm and gold and (pred_norm in gold or gold in pred_norm):
            return 1.0, pred_norm, gold, "substring"

        # if pred matches an option that equals gold (after norm), count it
        # (covers cases like punctuation differences)
        for o in opts:
            if pred_norm == o and o == gold:
                return 1.0, pred_norm, gold, "option_exact"

        # fuzzy
        if _HAS_FUZZ and pred_norm and gold:
            r = fuzz.ratio(pred_norm, gold)
            if r >= FUZZY_TH:
                return 1.0, pred_norm, gold, f"fuzzy_{r}"

        return 0.0, pred_norm, gold, "mismatch"

    results_dict = {res["idx"]: res for res in results}
    meta_dict = {m["idx"]: m for m in meta_data}

    scores = []
    compare_logs = []

    for idx, meta in meta_dict.items():
        if idx in results_dict:
            pred_raw = results_dict[idx]["results"].get("final_answer", "")
        else:
            pred_raw = ""

        score, pred, gold, reason = _match(pred_raw, meta)
        scores.append(score)

        compare_logs.append({
            "idx": idx,
            "question_id": meta.get("question_id", None),
            "dataset": meta.get("dataset", None),
            "question_type": meta.get("question_type", None),

            # keep legacy keys for logger
            "gold": gold,
            "pred": pred,

            "gold_raw": str(meta.get("label", "")),
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
