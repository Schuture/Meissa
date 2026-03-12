from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger

import os
import re
import json
import string
from datasets import Dataset

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)


def load_dataset(file_path, num_samples=None):
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if num_samples is None:
        return Dataset.from_dict({"data": dataset})
    return Dataset.from_dict({"data": dataset[:num_samples]})


def _normalize_answer(s: str) -> str:
    """
    A light VQA-style normalization:
      - lowercase
      - strip
      - remove punctuation
      - remove articles: a, an, the
      - collapse whitespace
    """
    if s is None:
        return ""
    s = str(s).strip().lower()

    # unify some common variants
    s = s.replace("&", "and")

    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))

    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_data_function():
    """
    Expect SLAKE test.json is a JSON list.
    Each item contains at least:
      - question: str
      - answer: str
      - img_name: str, like "xmlab102/source.jpg", relative to image_root
      - qid: int
      - img_id: int
    """
    dataset_path = task_config["dataset_path"]
    image_root = task_config.get("image_root", None)
    num_samples = task_config.get("num_sample", None)

    if image_root is None:
        raise ValueError("slake config.yaml must provide image_root")

    dataset = load_dataset(dataset_path, num_samples)

    meta_data = []
    for i, row in enumerate(dataset):
        item = row["data"]
        q_lang = str(item.get("q_lang", "")).strip().lower()
        if q_lang != "en":
            continue

        item_id = f"slake_{i}"

        rel_img = str(item.get("img_name", "")).strip()
        if len(rel_img) == 0:
            continue

        image_path = os.path.join(image_root, rel_img)
        if not os.path.exists(image_path):
            # skip missing image to avoid crashing the whole run
            logger.warning(f"Missing image: {image_path}")
            continue

        text = str(item.get("question", "")).strip()
        label = str(item.get("answer", "")).strip()

        meta = {
            "idx": item_id,
            "text": text,
            "label": label,
            "image_path": image_path,
            # keep extra fields for debug/analysis
            "qid": item.get("qid", None),
            "img_id": item.get("img_id", None),
            "q_lang": item.get("q_lang", None),
            "modality": item.get("modality", None),
            "location": item.get("location", None),
            "answer_type": item.get("answer_type", None),
            "content_type": item.get("content_type", None),
            "img_name": rel_img,
        }
        meta_data.append(meta)

    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def evaluate_function(results, meta_data):
    """
    SLAKE evaluation (practical, closer to common repos than strict exact match):

    Score = 1 if any of these holds (in order):
      1) normalized exact match after synonym canonicalization
      2) substring match (handles "ct scan" vs "ct", "left lung" vs "lung")
      3) fuzzy match (thefuzz ratio >= threshold)

    Notes:
      - This is not the original VQA multi-annotator soft accuracy formula,
        because SLAKE test.json usually has a single "answer" string.
      - You can tune FUZZY_TH to be stricter (85) or looser (75).
    """
    import re
    import string

    try:
        from thefuzz import fuzz
        _HAS_FUZZ = True
    except Exception:
        _HAS_FUZZ = False

    FUZZY_TH = 80

    # Keep this small and high-impact. Add more after you inspect compare_logs.
    # Map any variant to a canonical form.
    SYN_CANON = {
        # modality
        "computed tomography": "ct",
        "ct scan": "ct",
        "cta": "ct",  # sometimes appears loosely
        "magnetic resonance imaging": "mri",
        "mr": "mri",
        "mr image": "mri",
        "x ray": "xray",
        "x-ray": "xray",
        "xray image": "xray",
        "ultrasound": "us",
        "ultrasonography": "us",

        # body regions / common anatomy wording
        "thorax": "chest",
        "pulmonary": "lung",
        "lungs": "lung",
        "abdomen": "abdominal",  # optional; you may prefer canonical "abdomen"
        "abdominal": "abdomen",
        "kidneys": "kidney",
        "hepatic": "liver",

        # direction / laterality (often causes misses)
        "left side": "left",
        "right side": "right",
        "lt": "left",
        "rt": "right",

        # boolean-ish (some questions act like yes/no)
        "true": "yes",
        "false": "no",
    }

    # Sometimes answers are short and appear with extra words
    # This helps reduce false negatives without going too loose.
    STOPWORDS = set(["a", "an", "the", "of", "to", "in", "on", "and"])

    def _basic_norm(s: str) -> str:
        if s is None:
            return ""
        s = str(s).strip().lower()
        s = s.replace("&", "and")
        # remove punctuation
        s = s.translate(str.maketrans("", "", string.punctuation))
        # collapse spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _canonize(s: str) -> str:
        s = _basic_norm(s)
        if not s:
            return ""

        # remove articles/stopwords for short answers
        toks = [t for t in s.split() if t not in STOPWORDS]
        s2 = " ".join(toks).strip()

        # apply synonym mapping on whole string first
        if s2 in SYN_CANON:
            return SYN_CANON[s2]

        # apply mapping token-wise (helps "computed tomography scan" -> contains "computed tomography")
        # do a simple longest-phrase replacement
        # (keep it deterministic and easy to debug)
        s3 = s2
        # sort by length to replace longer phrases first
        for k in sorted(SYN_CANON.keys(), key=lambda x: len(x), reverse=True):
            if k in s3:
                s3 = s3.replace(k, SYN_CANON[k])

        s3 = re.sub(r"\s+", " ", s3).strip()

        # after replacements, if it becomes a known key again, map once more
        if s3 in SYN_CANON:
            return SYN_CANON[s3]

        return s3

    def _token_set(s: str):
        return set([t for t in _canonize(s).split() if t])

    def _is_match(pred_raw: str, gold_raw: str):
        pred = _canonize(pred_raw)
        gold = _canonize(gold_raw)

        if pred == gold and pred != "":
            return 1.0, "exact"

        if pred == "" or gold == "":
            return 0.0, "empty"

        # substring match (safe when answers are short)
        # example: "ct scan" vs "ct", "left lung" vs "lung"
        if pred in gold or gold in pred:
            return 1.0, "substring"

        # token containment (helps "left lung" vs "lung", but avoids too many false positives)
        pt = _token_set(pred_raw)
        gt = _token_set(gold_raw)
        if len(gt) > 0 and gt.issubset(pt):
            return 1.0, "token_subset_gold_in_pred"
        if len(pt) > 0 and pt.issubset(gt):
            return 1.0, "token_subset_pred_in_gold"

        # fuzzy match
        if _HAS_FUZZ:
            r = fuzz.ratio(pred, gold)
            if r >= FUZZY_TH:
                return 1.0, f"fuzzy_{r}"
            # also try partial ratio for cases like "ct abdomen" vs "ct"
            pr = fuzz.partial_ratio(pred, gold)
            if pr >= FUZZY_TH:
                return 1.0, f"partial_fuzzy_{pr}"

        return 0.0, "mismatch"

    results_dict = {res["idx"]: res for res in results}
    meta_dict = {m["idx"]: m for m in meta_data}

    scores = []
    compare_logs = []

    for idx, meta in meta_dict.items():
        if idx in results_dict:
            pred_raw = results_dict[idx]["results"].get("final_answer", "")
        else:
            pred_raw = ""

        gold_raw = meta.get("label", "")

        score, reason = _is_match(pred_raw, gold_raw)
        scores.append(score)

        gold_norm = _canonize(gold_raw)
        pred_norm = _canonize(pred_raw)

        compare_logs.append({
            "idx": idx,
            "qid": meta.get("qid", None),
            "img_id": meta.get("img_id", None),
            "content_type": meta.get("content_type", None),
            "answer_type": meta.get("answer_type", None),

            # keep the legacy keys that the logger expects
            "gold": gold_norm,
            "pred": pred_norm,

            # keep richer debug fields
            "gold_raw": str(gold_raw),
            "pred_raw": str(pred_raw),
            "gold_norm": gold_norm,
            "pred_norm": pred_norm,

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
