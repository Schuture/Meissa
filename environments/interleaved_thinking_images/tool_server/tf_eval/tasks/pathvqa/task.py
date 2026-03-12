from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger

import os
import re
import json
import unicodedata
from typing import Any, Dict, List, Optional

from datasets import Dataset
from thefuzz import fuzz

try:
    from math_verify import parse, verify
except ImportError:
    parse, verify = None, None


logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)


def _load_json_list(file_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {file_path}, got {type(data)}")
    if num_samples is None:
        return data
    return data[:num_samples]


def load_dataset(file_path: str, num_samples: Optional[int] = None) -> Dataset:
    data = _load_json_list(file_path, num_samples)
    return Dataset.from_dict({"data": data})


def _pick_gold(item: Dict[str, Any]) -> str:
    """
    New PathVQA json formats:
      - yes/no subset: {"question","label","image_path"}
      - full:         {"question","answer","image_path"}
    Prefer label if present, else answer.
    """
    if "label" in item and item["label"] is not None and str(item["label"]).strip() != "":
        return str(item["label"]).strip()
    if "answer" in item and item["answer"] is not None and str(item["answer"]).strip() != "":
        return str(item["answer"]).strip()
    return ""


def load_data_function():
    """
    Expect dataset_path points to a JSON list.
    Each item at least contains:
      - question: str
      - image_path: str (absolute path)
    And one of:
      - label: str (for yes/no subset), or
      - answer: str (for full set)
    Optional:
      - orig_idx: int
    """
    dataset_path = task_config["dataset_path"]
    num_samples = task_config.get("num_sample", None)

    dataset = load_dataset(dataset_path, num_samples)

    meta_data: List[Dict[str, Any]] = []
    skipped = 0

    for i, row in enumerate(dataset):
        item = row["data"]

        item_id = f"pathvqa_{i}"

        image_path = item.get("image_path", None)
        if image_path is None or (isinstance(image_path, str) and len(image_path.strip()) == 0):
            skipped += 1
            continue

        # optional: if you want to be strict that file exists
        # if not os.path.exists(image_path):
        #     skipped += 1
        #     continue

        text = str(item.get("question", "")).strip()
        gold = _pick_gold(item)

        data_item = {
            "idx": item_id,
            "text": text,
            "label": gold,          # keep key name "label" for downstream evaluator
            "image_path": image_path,
        }
        if "orig_idx" in item:
            data_item["orig_idx"] = item["orig_idx"]

        meta_data.append(data_item)

    logger.info(f"Total data number: {len(meta_data)} (skipped={skipped})")
    return meta_data


def evaluate_function(results, meta_data):
    """
    Evaluation policy:
      - If gold is yes/no (or can be normalized to yes/no), use yes/no exact match.
      - Else:
          1) math_verify (if available and parse-able) -> correct/incorrect
          2) normalized exact match
          3) containment / token-subset match (helps short correct answers)
          4) fuzzy (partial + token_set + ratio) >= threshold
    """
    fuzzy_threshold = int(task_config.get("fuzzy_threshold", 70))

    def _norm_text(s: Any) -> str:
        if s is None:
            return ""
        s = str(s)
        s = unicodedata.normalize("NFKC", s)
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    def _norm_yn(s: Any) -> str:
        s = _norm_text(s)

        if re.search(r"\byes\b", s):
            return "yes"
        if re.search(r"\bno\b", s):
            return "no"

        if re.search(r"\btrue\b", s):
            return "yes"
        if re.search(r"\bfalse\b", s):
            return "no"

        if s in ["y"]:
            return "yes"
        if s in ["n"]:
            return "no"

        return ""

    def _is_yesno_gold(gold_raw: Any) -> bool:
        return _norm_yn(gold_raw) in ("yes", "no")

    def _strip_explanations(s: str) -> str:
        """
        Make Gemini answers less verbose for matching:
        - drop everything after common separators
        - drop parentheses content
        """
        s = _norm_text(s)
        if s == "":
            return ""
        # remove parenthetical content
        s = re.sub(r"\([^)]*\)", "", s).strip()
        # cut at sentence-like separators (keep the first short answer chunk)
        s = re.split(r"[.;:\n\r]|\\n", s, maxsplit=1)[0].strip()
        # cut at " is " patterns (common verbose style)
        s = re.split(r"\bis\b|\bare\b", s, maxsplit=1)[0].strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def _simple_tokenize(s: str) -> List[str]:
        s = _norm_text(s)
        s = re.sub(r"[^a-z0-9\s\-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s == "":
            return []
        toks = s.split(" ")
        # crude singularization for common plurals: ducts -> duct, cells -> cell
        norm_toks = []
        for t in toks:
            if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
                norm_toks.append(t[:-1])
            else:
                norm_toks.append(t)
        return [t for t in norm_toks if t != ""]

    def _token_subset_match(pred: str, gold: str) -> bool:
        """
        Treat as correct if pred tokens are largely contained in gold tokens.
        This helps cases like:
          gold: "bile duct cells and canals of hering"
          pred: "bile ducts"
        """
        pt = set(_simple_tokenize(pred))
        gt = set(_simple_tokenize(gold))
        if not pt or not gt:
            return False
        # require at least 1 meaningful overlap, and most pred tokens covered
        inter = pt & gt
        if len(inter) == 0:
            return False
        return len(inter) / max(len(pt), 1) >= 0.6

    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}

    scores: List[float] = []
    compare_logs: List[Dict[str, Any]] = []

    for idx, meta in meta_dict.items():
        if idx in results_dict:
            pred_raw = results_dict[idx]["results"].get("final_answer", "")
        else:
            pred_raw = ""

        gold_raw = meta.get("label", "")
        meta["prediction"] = pred_raw

        # yes/no branch
        if _is_yesno_gold(gold_raw):
            gold = _norm_yn(gold_raw)
            pred = _norm_yn(pred_raw)
            score = 1.0 if pred == gold else 0.0
            scores.append(score)
            compare_logs.append(
                {
                    "idx": idx,
                    "orig_idx": meta.get("orig_idx", None),
                    "gold": gold,
                    "pred": pred,
                    "pred_raw": str(pred_raw),
                    "gold_raw": str(gold_raw),
                    "score": score,
                    "mode": "yesno",
                }
            )
            continue

        # open-ended branch
        gold_n = _norm_text(gold_raw)
        pred_n = _norm_text(pred_raw)

        # extra cleaned version for verbose outputs
        pred_c = _strip_explanations(pred_raw)
        gold_c = _strip_explanations(gold_raw)

        score = 0.0
        mode = "open"

        # 1) math_verify if available
        if parse is not None and verify is not None:
            try:
                g = parse(gold_raw)
                p = parse(pred_raw)
                if g is not None and p is not None:
                    ok = verify(p, g)
                    score = 1.0 if ok else 0.0
                    mode = "math_verify"
            except Exception:
                pass

        # 2) normalized exact match (also on cleaned)
        if score == 0.0:
            if gold_n != "" and pred_n == gold_n:
                score = 1.0
                mode = "open_exact"
            elif gold_c != "" and pred_c != "" and pred_c == gold_c:
                score = 1.0
                mode = "open_exact_clean"

        # 3) containment / subset match (handles short correct answers)
        if score == 0.0 and gold_n != "" and pred_n != "":
            # direct substring either way, try both raw-normalized and cleaned
            if (gold_n in pred_n) or (pred_n in gold_n):
                score = 1.0
                mode = "open_contain_norm"
            elif (gold_c != "" and pred_c != "" and ((gold_c in pred_c) or (pred_c in gold_c))):
                score = 1.0
                mode = "open_contain_clean"
            elif _token_subset_match(pred_n, gold_n) or _token_subset_match(pred_c, gold_n):
                score = 1.0
                mode = "open_token_subset"

        # 4) fuzzy match (use stronger variants)
        ratio = None
        if score == 0.0:
            if gold_n != "" and pred_n != "":
                r1 = fuzz.partial_ratio(pred_n, gold_n)
                r2 = fuzz.token_set_ratio(pred_n, gold_n)
                r3 = fuzz.ratio(pred_n, gold_n)
                # also try cleaned pred
                if pred_c != "" and pred_c != pred_n:
                    r1c = fuzz.partial_ratio(pred_c, gold_n)
                    r2c = fuzz.token_set_ratio(pred_c, gold_n)
                    r3c = fuzz.ratio(pred_c, gold_n)
                    ratio = max(r1, r2, r3, r1c, r2c, r3c)
                else:
                    ratio = max(r1, r2, r3)

                score = 1.0 if ratio >= fuzzy_threshold else 0.0
                mode = f"open_fuzzy_max_{fuzzy_threshold}"
            else:
                ratio = None

        compare_logs.append(
            {
                "idx": idx,
                "orig_idx": meta.get("orig_idx", None),
                "gold": gold_n,
                "pred": pred_n,
                "pred_clean": pred_c,
                "pred_raw": str(pred_raw),
                "gold_raw": str(gold_raw),
                "score": score,
                "mode": mode,
                "fuzzy_max": ratio,
            }
        )
        scores.append(score)

    accuracy = sum(scores) / len(scores) if len(scores) > 0 else 0.0
    return {
        "Acc": accuracy,
        "compare_logs": compare_logs,
        "results": results,
        "meta_data": meta_data,
    }

