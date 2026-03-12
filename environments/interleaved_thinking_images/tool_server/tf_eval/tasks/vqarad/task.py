from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger

import os
import re
import json
import string
import unicodedata
from typing import Any, Dict, List, Optional

from datasets import Dataset

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
    New VQA-RAD json formats:
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
      - image_path: str (absolute path recommended)
    And one of:
      - label: str (for yes/no subset), or
      - answer: str (for full set)
    Optional:
      - orig_idx: int
    """
    dataset_path = task_config["dataset_path"]
    num_samples = task_config.get("num_sample", None)

    ds = load_dataset(dataset_path, num_samples)

    meta_data: List[Dict[str, Any]] = []
    skipped = 0

    for i, row in enumerate(ds):
        item = row["data"]
        item_id = f"vqarad_{i}"

        q = str(item.get("question", "")).strip()

        gold = _pick_gold(item)

        image_path = item.get("image_path", None)
        if image_path is None or (isinstance(image_path, str) and len(image_path.strip()) == 0):
            skipped += 1
            continue

        # If you want to be strict that the file must exist, uncomment:
        # if not os.path.exists(image_path):
        #     skipped += 1
        #     continue

        meta = {
            "idx": item_id,
            "text": q,
            "label": str(gold).strip(),     # keep key name "label" for evaluator
            "image_path": image_path,
        }
        if "orig_idx" in item:
            meta["orig_idx"] = item["orig_idx"]

        meta_data.append(meta)

    logger.info(f"Total data number: {len(meta_data)} (skipped={skipped})")
    return meta_data



# =========================================================
# Enhanced Evaluation Logic
# =========================================================

def _basic_norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    # Replace common connectors
    s = s.replace("&", " and ").replace("/", " or ")
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_yesno(text: str) -> Optional[str]:
    """Strictly extract yes/no from a longer sentence."""
    t = _basic_norm(text)
    
    # Check for explicit "yes" or "no" at the beginning or as a distinct word
    # Prioritize exact match or start of sentence
    if t in ["yes", "no"]:
        return t
    if t.startswith("yes "):
        return "yes"
    if t.startswith("no "):
        return "no"
        
    # Check for strong keywords
    if re.search(r"\byes\b", t): return "yes"
    if re.search(r"\bno\b", t): return "no"
    if re.search(r"\btrue\b", t): return "yes"
    if re.search(r"\bfalse\b", t): return "no"
    
    # Clinical negatives
    if re.search(r"\b(absent|not present|not seen|no evidence)\b", t):
        return "no"
    if re.search(r"\b(present|seen|evident)\b", t):
        return "yes"
        
    return None

def evaluate_function(results, meta_data):
    try:
        from thefuzz import fuzz
        _HAS_FUZZ = True
    except Exception:
        _HAS_FUZZ = False

    FUZZY_TH = int(task_config.get("fuzzy_threshold", 85))

    # 1. Number Mapping
    NUM_MAP = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "zero": "0", "none": "0"
    }

    # 2. Strong Synonym / Concept Mapping
    # Key = Canonical (Gold usually uses these, or we map both to these)
    # Value = List of alternatives
    SYNONYM_MAP = {
        "male": ["man", "boy", "gentleman"],
        "female": ["woman", "girl", "lady"],
        "frontal": ["ap", "pa", "anterior posterior", "posterior anterior", "coronal"], # loosely grouping frontal views
        "lateral": ["lat", "side view"],
        "air": ["gas"],
        "left side": ["left hemisphere", "left lung", "left lobe"], # Context dependent, but helpful
        "right side": ["right hemisphere", "right lung", "right lobe"],
        "intestine": ["bowel", "colon", "small intestine", "large intestine"],
        "iv": ["intravenous"],
        "oral": ["po"],
    }
    
    # Reverse map for lookup: "man" -> "male"
    REVERSE_SYN = {}
    for canon, alts in SYNONYM_MAP.items():
        for a in alts:
            REVERSE_SYN[a] = canon
        REVERSE_SYN[canon] = canon

    def _canonize_concept(s: str) -> str:
        s = _basic_norm(s)
        # Number normalization
        if s in NUM_MAP:
            return NUM_MAP[s]
        # Synonym normalization
        if s in REVERSE_SYN:
            return REVERSE_SYN[s]
        return s

    def _check_containment(gold: str, pred: str) -> bool:
        """
        Check if gold concepts are covered by pred.
        Handle cases like gold="oral and iv", pred="iv contrast" -> False (missing oral)
        """
        g_toks = set(_basic_norm(gold).split())
        p_toks = set(_basic_norm(pred).split())
        
        # Remove stop words
        stop = {"and", "or", "with", "without", "the", "a", "an", "in", "on", "of", "to"}
        g_toks -= stop
        p_toks -= stop
        
        # If gold is complex (multiple concepts), strict subset required
        # e.g. Gold: "oral", "iv" -> Pred must have "oral", "iv"
        if len(g_toks) > 1:
            return g_toks.issubset(p_toks)
        
        # If gold is single concept, allow lenient match
        if len(g_toks) == 1:
            g_word = list(g_toks)[0]
            # Try synonym match
            g_canon = _canonize_concept(g_word)
            for p in p_toks:
                if _canonize_concept(p) == g_canon:
                    return True
            # Try substring
            if g_word in _basic_norm(pred):
                return True
                
        return False

    results_dict = {res["idx"]: res for res in results}
    meta_dict = {m["idx"]: m for m in meta_data}

    scores = []
    compare_logs = []

    for idx, meta in meta_dict.items():
        pred_raw = results_dict.get(idx, {}).get("results", {}).get("final_answer", "")
        gold_raw = meta.get("label", "")
        question = meta.get("text", "").lower()

        # Update meta for dumping
        meta["prediction"] = pred_raw

        gold_norm = _basic_norm(gold_raw)
        pred_norm = _basic_norm(pred_raw)
        
        score = 0.0
        reason = "mismatch"

        # --- Strategy 1: Yes/No Logic ---
        # If question is clearly Yes/No type, prioritize yes/no extraction
        is_yn_question = any(q_start in question for q_start in ["is ", "are ", "do ", "does ", "can ", "could ", "has ", "have "])
        
        gold_yn = _extract_yesno(gold_raw)
        
        # Case A: Gold is explicitly Yes/No
        if gold_yn:
            pred_yn = _extract_yesno(pred_raw)
            if pred_yn == gold_yn:
                score = 1.0
                reason = "yesno_match"
            else:
                # Fallback: Sometimes Gold is "Yes", Pred is "Yes, there is..."
                pass 
                
        # Case B: Question is Yes/No type, but Gold isn't explicitly "yes/no" (e.g. Gold="in the bowel")
        # And Pred is "Yes". This is tricky. 
        # Usually implies Question: "Is contrast in bowel?" Gold: "in the bowel" (Implicit Yes).
        # We can't easily score this as correct unless we infer Gold implies Yes.
        # But for your specific example: Question="is there contrast...?" Gold="in the bowel", Pred="yes"
        # Strictly speaking, if Gold is a location, the question might not be purely Y/N, or the dataset is messy.
        # Let's skip complex inference and move to semantic matching.

        # --- Strategy 2: Exact & Synonym Matching ---
        if score == 0.0:
            # Canonize both
            g_canon = _canonize_concept(gold_norm)
            p_canon = _canonize_concept(pred_norm)

            if g_canon == p_canon:
                score = 1.0
                reason = "exact_canon"
            elif _check_containment(gold_raw, pred_raw):
                score = 1.0
                reason = "containment"
            
            # Specific Fixes for your examples
            # Gold: "retrocardiac", Pred: "heart" -> Usually wrong, but maybe close? 
            # Actually retrocardiac means *behind* the heart. Pred "heart" is technically imprecise. 
            # I will assume Strict is better unless you want to force it.

            # Gold: "left hemisphere", Pred: "left side" -> Match "left" + "side/hemisphere"
            elif "left" in g_canon and "left" in p_canon:
                # Loose lateralization match
                score = 1.0
                reason = "lateralization_match"
            elif "right" in g_canon and "right" in p_canon:
                score = 1.0
                reason = "lateralization_match"

        # --- Strategy 3: Fuzzy Fallback ---
        if score == 0.0 and _HAS_FUZZ:
            # Use token sort ratio to handle "oral and iv" vs "iv and oral"
            r = fuzz.token_sort_ratio(gold_norm, pred_norm)
            if r >= FUZZY_TH:
                score = 1.0
                reason = f"fuzzy_{r}"

        scores.append(score)
        compare_logs.append({
            "idx": idx,
            "orig_idx": meta.get("orig_idx"),
            "question": question,
            "gold": gold_raw,
            "pred": pred_raw,
            "score": score,
            "reason": reason
        })

    acc = sum(scores) / len(scores) if scores else 0.0
    return {
        "Acc": acc,
        "compare_logs": compare_logs,
        "results": results,
        "meta_data": meta_data,
    }