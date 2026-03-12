#!/usr/bin/env python3
# filter_for_gemini_distill.py
#
# Select high-value samples for Gemini distillation from Qwen3 scan outputs (jsonl).
# Focus: agentic tool behavior (tool selection, tool-result parsing, retry/fallback).
#
# v3: two-stage selection:
#   (1) pick strong behavior errors by score >= min-score
#   (2) optionally top-up a limited number of single_expert_wrong (score==3) by per-template caps

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

FAILURE_MARKERS = [
    "error",
    "failed",
    "traceback",
    "exception",
    "invalid",
    "badrequest",
    "timeout",
    "connection refused",
    "maximum",
]

NEGATION_MARKERS = [
    "no",
    "none",
    "negative",
    "normal",
    "without",
    "absent",
    "not seen",
    "no evidence",
]

AFFIRM_MARKERS = [
    "present",
    "seen",
    "found",
    "there is",
    "there are",
    "identified",
    "demonstrates",
    "shows",
]

TOOL_SYNONYMS = {
    "ng tube": ["ng tube", "nasogastric", "enteric tube", "feeding tube", "dobhoff"],
    "pleural effusion": ["pleural effusion", "effusion"],
    "lung opacity": ["lung opacity", "opacity", "infiltrate", "consolidation"],
    "enlarged cardiac silhouette": ["enlarged cardiac silhouette", "cardiomegaly", "enlarged heart"],
}

BEHAVIOR_LIKE_CONTENT_TYPES = {
    "presence",
    "anatomy",
    "attribute",
    "plane",
    "gender",
    "size",
}


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                yield {"_parse_error": str(e), "_line_no": line_no, "_raw": line[:500]}


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_tool_name(name: Any) -> str:
    if not name:
        return ""
    return str(name).strip()


def extract_tools(tool_trace: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
    calls = []
    results = []
    if not tool_trace:
        return calls, results
    for ev in tool_trace:
        if ev.get("event") == "tool_call":
            tn = normalize_tool_name(ev.get("tool_name"))
            if tn:
                calls.append(tn)
        elif ev.get("event") == "tool_result":
            results.append(ev)
    return calls, results


def has_failure_marker_in_results(tool_results: List[Dict[str, Any]]) -> bool:
    for ev in tool_results:
        txt = ev.get("content_text") or ev.get("content_preview") or ""
        low = str(txt).lower()
        if any(m in low for m in FAILURE_MARKERS):
            return True
        if "{'error':" in low or '{"error":' in low:
            return True
        if ("status" in low) and (("status': 'error'" in low) or ('status": "error"' in low)):
            return True
        if ("response" in low) and (("response': ''" in low) or ('response": ""' in low)):
            return True
    return False


def tool_result_text_by_tool(tool_results: List[Dict[str, Any]]) -> Dict[str, str]:
    out = defaultdict(list)
    for ev in tool_results:
        tn = normalize_tool_name(ev.get("tool_name"))
        txt = ev.get("content_text") or ev.get("content_preview") or ""
        out[tn].append(str(txt))
    return {k: "\n".join(v) for k, v in out.items()}


def any_keyword_in(text: str, kws: List[str]) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in kws)


def contains_negation(tool_text: str) -> bool:
    low = (tool_text or "").lower()
    return any(m in low for m in NEGATION_MARKERS)


def contains_affirmation(ans_text: str) -> bool:
    low = (ans_text or "").lower()
    return any(m in low for m in AFFIRM_MARKERS)


def get_num_tools(entry: Dict[str, Any]) -> int:
    v = entry.get("num_tools", None)
    if isinstance(v, int):
        return v
    calls, _ = extract_tools(entry.get("tool_trace"))
    return len(calls)


def gt_list(entry: Dict[str, Any]) -> List[str]:
    gt = entry.get("gt_answer_norm") or entry.get("gt_answer_raw") or entry.get("gt_answer") or []
    if isinstance(gt, str):
        xs = [gt]
    elif isinstance(gt, (list, tuple)):
        xs = list(gt)
    else:
        xs = [safe_str(gt)]
    out = []
    for x in xs:
        s = str(x).strip().lower()
        if not s:
            continue
        out.append(s)
    return out


def keep_single_expert_wrong_behavior_like(entry: Dict[str, Any], abnormality_gt_len_thresh: int = 4) -> bool:
    ct = safe_str(entry.get("content_type", "unknown")).strip().lower()
    st = safe_str(entry.get("semantic_type", "unknown")).strip().lower()

    # keep verify (tool policy)
    if st == "verify":
        return True

    if ct in BEHAVIOR_LIKE_CONTENT_TYPES:
        return True

    if ct == "abnormality":
        gt = gt_list(entry)
        if gt == ["none"]:
            return True
        if len(gt) > abnormality_gt_len_thresh:
            return False
        return True

    return False


def likely_tool_limit(entry: Dict[str, Any], calls: List[str], tool_text_map: Dict[str, str]) -> bool:
    if entry.get("status") != "ok":
        return False
    if not isinstance(entry.get("is_correct"), bool):
        return False
    if entry.get("is_correct") is True:
        return False
    if len(set(calls)) != 1:
        return False

    only_tool = calls[0]
    if only_tool != "chest_xray_expert":
        return False

    gt = gt_list(entry)
    if not gt or gt == ["none"]:
        return False

    tool_txt = tool_text_map.get(only_tool, "")
    tool_low = tool_txt.lower()
    pred = safe_str(entry.get("assistant_final", ""))

    hit = 0
    for g in gt:
        if g in tool_low:
            hit += 1
            continue
        for k, syns in TOOL_SYNONYMS.items():
            if g == k and any(s in tool_low for s in syns):
                hit += 1
                break

    if hit >= max(1, int(0.6 * len(gt))):
        return False

    tool_len = len(tool_low)
    pred_low = pred.lower()
    tokens = re.findall(r"[a-zA-Z_]+", tool_low)[:200]
    uniq = list(dict.fromkeys(tokens))
    overlap = sum(1 for tok in uniq if tok and tok in pred_low)
    overlap_ratio = overlap / max(1, len(uniq))

    if tool_len < 2000 and overlap_ratio > 0.6:
        return True

    return False

def is_toxic_data(entry: Dict[str, Any]) -> bool:
    """
    Check if the data sample is likely 'toxic' (bad ground truth, ambiguous, or unsolvable).
    """
    # 1. Check for Empty GT Raw
    # Many 'hard' failures are actually just empty annotations.
    raw_gt = entry.get("gt_answer_raw")
    if isinstance(raw_gt, list) and len(raw_gt) == 0:
        return True
    
    # 2. Check for "None" GT in "Point out/Locate" questions
    # If the question asks to "Point out" or "Locate" and the answer is "none",
    # models often panic trying to find something that isn't there.
    norm_gt = entry.get("gt_answer_norm") or []
    if isinstance(norm_gt, str): norm_gt = [norm_gt]
    
    q_low = safe_str(entry.get("question", "")).lower()
    if "none" in [str(x).lower() for x in norm_gt]:
        # Toxic combination: "Where is the X?" -> "None"
        # These cause massive hallucinations in agentic models.
        if any(kw in q_low for kw in ["point out", "locate", "where is", "show me"]):
            return True

    return False

def analyze_behavior_tags(entry: Dict[str, Any]) -> Tuple[List[str], int, Dict[str, Any]]:
    tags: List[str] = []
    debug: Dict[str, Any] = {}

    calls, results = extract_tools(entry.get("tool_trace"))
    tool_text_map = tool_result_text_by_tool(results)
    num_tools = get_num_tools(entry)
    is_correct = entry.get("is_correct", None)
    wrong = (isinstance(is_correct, bool) and (is_correct is False))

    q = safe_str(entry.get("question", ""))
    qlow = q.lower()
    ct = safe_str(entry.get("content_type", "unknown")).strip().lower()
    st = safe_str(entry.get("semantic_type", "unknown")).strip().lower()

    has_failure = has_failure_marker_in_results(results)

    debug.update(
        {
            "num_tools": num_tools,
            "tools": list(calls),
            "has_failure": has_failure,
            "content_type": ct,
            "semantic_type": st,
            "exclude_tool_limit": None,
        }
    )

    # A1: segmentation failure
    if "chest_xray_segmentation" in calls:
        seg_txt = tool_text_map.get("chest_xray_segmentation", "")
        seg_failed = any_keyword_in(seg_txt, FAILURE_MARKERS) or any_keyword_in(seg_txt, ["error", "exception", "traceback"])
        if seg_failed:
            seg_call_count = sum(1 for t in calls if t == "chest_xray_segmentation")
            has_retry = seg_call_count >= 2
            has_fallback = any(t in calls for t in ["chest_xray_classifier", "chest_xray_report_generator", "chest_xray_expert"])
            if (not has_retry) and (not has_fallback) and wrong:
                tags.append("seg_failed_no_recovery")
            elif (not has_retry) and wrong:
                tags.append("seg_failed_no_retry")
            else:
                tags.append("seg_failed")

    # A2: multi-tool panic
    if num_tools >= 4 and wrong:
        tags.append("multi_tool_panic")

    # A3: tool-task mismatch for plane
    if (ct == "plane" or st == "choose") and ("ap" in qlow or "pa" in qlow or "perspective" in qlow):
        if any(t in calls for t in ["image_visualizer", "chest_xray_report_generator", "chest_xray_segmentation", "xray_phrase_grounding"]) and wrong:
            tags.append("tool_task_mismatch_plane")

    # A4: empty/none GT + grounding / negation misread
    gt_empty = entry.get("gt_is_empty_list", None)
    if isinstance(gt_empty, bool) and gt_empty is True:
        if any(t == "xray_phrase_grounding" for t in calls) and wrong:
            tags.append("grounding_on_none")
        combined_tool_text = "\n".join(tool_text_map.values())
        if contains_negation(combined_tool_text) and contains_affirmation(safe_str(entry.get("assistant_final", ""))) and wrong:
            tags.append("negation_misread")

    # A5: tool failure markers
    if has_failure and wrong:
        distinct = list(dict.fromkeys(calls))
        if len(distinct) <= 1:
            tags.append("tool_failure_no_fallback")
        else:
            tags.append("tool_failure_with_fallback_but_wrong")

    # B1: single expert only and wrong
    if wrong and len(set(calls)) == 1 and calls and calls[0] == "chest_xray_expert":
        tags.append("single_expert_wrong")

    exclude_tool_limit = likely_tool_limit(entry, calls, tool_text_map)
    debug["exclude_tool_limit"] = exclude_tool_limit

    # Score
    score = 0
    if "seg_failed_no_recovery" in tags:
        score += 10
    if "seg_failed_no_retry" in tags:
        score += 7
    if "multi_tool_panic" in tags:
        # 如果 GT 很丰富，说明真的是难为此模型了，给高分让老师教
        # 如果 GT 很单薄，说明可能是无病呻吟
        gt_len = len(gt_list(entry))
        if gt_len >= 1 and gt_list(entry) != ["none"]:
            score += 6  # Keep high score for valid hard cases
        else:
            score += 1  # Reduce score for panic on simple/empty cases
    if "tool_task_mismatch_plane" in tags:
        score += 6
    if "tool_failure_no_fallback" in tags:
        score += 6
    if "grounding_on_none" in tags:
        score += 5
    if "negation_misread" in tags:
        score += 5

    if "tool_failure_with_fallback_but_wrong" in tags:
        score += 2
    if "single_expert_wrong" in tags:
        score += 2
    if "seg_failed" in tags:
        score += 1

    if wrong:
        score += 1

    if exclude_tool_limit:
        score = -10**9

    return tags, score, debug


def bucket_key(entry: Dict[str, Any], tag: str) -> Tuple[str, str]:
    return (str(entry.get("template_program", "unknown")), str(tag))


def uid_of(entry: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(entry.get("idx")),
        str(entry.get("image_id", "")),
        str(entry.get("question", ""))[:80],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-total", type=int, default=2000)
    ap.add_argument("--per-bucket", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-score", type=int, default=3)
    ap.add_argument("--single-expert-abn-gt-len-thresh", type=int, default=4)

    # NEW: top-up single_expert_wrong after main selection
    ap.add_argument("--topup-single-expert", action="store_true",
                    help="After selecting score>=min-score, top-up some single_expert_wrong (score==3) with caps.")
    ap.add_argument("--topup-single-expert-total", type=int, default=200,
                    help="Max number of single_expert_wrong samples to top-up.")
    ap.add_argument("--topup-single-expert-per-template", type=int, default=5,
                    help="Max top-up single_expert_wrong per template_program.")
    ap.add_argument("--topup-single-expert-seed", type=int, default=None,
                    help="Random seed for top-up (default: same as --seed).")

    args = ap.parse_args()

    rng = random.Random(args.seed)
    topup_rng = random.Random(args.seed if args.topup_single_expert_seed is None else args.topup_single_expert_seed)

    total = 0
    parse_errors = 0

    # We will store all processed entries with tags/score for later top-up
    processed: List[Dict[str, Any]] = []

    for entry in read_jsonl(args.input):
        total += 1
        if "_parse_error" in entry:
            parse_errors += 1
            continue
        
        # [NEW] 核心过滤逻辑：直接丢弃脏数据
        if is_toxic_data(entry):
            continue

        tags, score, debug = analyze_behavior_tags(entry)
        if score < 0:
            continue
        if not tags:
            continue

        # split single_expert_wrong: keep only behavior-like half
        if "single_expert_wrong" in tags:
            keep = keep_single_expert_wrong_behavior_like(entry, abnormality_gt_len_thresh=args.single_expert_abn_gt_len_thresh)
            debug["single_expert_behavior_like"] = keep
            if not keep:
                tags = [t for t in tags if t != "single_expert_wrong"]
                if not tags:
                    continue
                score = max(0, score - 2)
        else:
            debug["single_expert_behavior_like"] = None

        out_entry = dict(entry)
        out_entry["distill_tags"] = tags
        out_entry["distill_score"] = score
        out_entry["distill_debug"] = {
            "num_tools": debug.get("num_tools"),
            "tools": debug.get("tools"),
            "has_failure": debug.get("has_failure"),
            "exclude_tool_limit": debug.get("exclude_tool_limit"),
            "single_expert_behavior_like": debug.get("single_expert_behavior_like"),
        }
        processed.append(out_entry)

    # Stage 1: select score>=min-score with bucket caps
    candidates = [e for e in processed if e.get("distill_score", -1) >= args.min_score]

    buckets = defaultdict(list)
    for e in candidates:
        for t in e["distill_tags"]:
            buckets[bucket_key(e, t)].append(e)

    bucket_items = []
    for bk, items in buckets.items():
        best = max(it.get("distill_score", -1) for it in items)
        bucket_items.append((best, bk, items))
    bucket_items.sort(key=lambda x: x[0], reverse=True)

    selected: List[Dict[str, Any]] = []
    selected_ids = set()

    for _, _, items in bucket_items:
        by_score = defaultdict(list)
        for it in items:
            by_score[it["distill_score"]].append(it)

        ordered = []
        for s in sorted(by_score.keys(), reverse=True):
            group = by_score[s]
            rng.shuffle(group)
            ordered.extend(group)

        take = 0
        for it in ordered:
            if len(selected) >= args.max_total:
                break
            uid = uid_of(it)
            if uid in selected_ids:
                continue
            selected_ids.add(uid)
            selected.append(it)
            take += 1
            if take >= args.per_bucket:
                break

        if len(selected) >= args.max_total:
            break

    # Stage 2: top-up single_expert_wrong score==3 with caps (optional)
    if args.topup_single_expert and len(selected) < args.max_total:
        remain = args.max_total - len(selected)
        budget = min(remain, args.topup_single_expert_total)

        # eligible: score==3 AND has only single_expert_wrong tag OR includes it
        elig = []
        for e in processed:
            if e.get("distill_score") != 3:
                continue
            if "single_expert_wrong" not in (e.get("distill_tags") or []):
                continue
            # avoid duplicates
            if uid_of(e) in selected_ids:
                continue
            elig.append(e)

        # group by template_program, then sample with per-template cap
        by_tmpl = defaultdict(list)
        for e in elig:
            by_tmpl[str(e.get("template_program", "unknown"))].append(e)

        tmpl_keys = list(by_tmpl.keys())
        topup_rng.shuffle(tmpl_keys)

        per_tmpl_count = Counter()
        added = 0

        for tmpl in tmpl_keys:
            if added >= budget:
                break
            items = by_tmpl[tmpl]
            topup_rng.shuffle(items)
            for e in items:
                if added >= budget:
                    break
                if per_tmpl_count[tmpl] >= args.topup_single_expert_per_template:
                    break
                uid = uid_of(e)
                if uid in selected_ids:
                    continue
                # mark as top-up for tracking
                if "single_expert_topup" not in e["distill_tags"]:
                    e["distill_tags"] = list(e["distill_tags"]) + ["single_expert_topup"]
                selected_ids.add(uid)
                selected.append(e)
                per_tmpl_count[tmpl] += 1
                added += 1

    selected.sort(key=lambda x: x.get("distill_score", 0), reverse=True)
    write_jsonl(args.output, selected)

    sel_tag_ctr = Counter()
    for e in selected:
        for t in e.get("distill_tags", []):
            sel_tag_ctr[t] += 1

    print("=" * 80)
    print("Filter for Gemini distillation candidates (two-stage)")
    print("Input:", args.input)
    print("Total lines:", total, "| parse_errors:", parse_errors)
    print("Processed (tagged):", len(processed))
    print("Stage1 candidates (score>=min_score):", len(candidates))
    print("Selected:", len(selected))
    print("Top tags in selected:")
    for k, v in sel_tag_ctr.most_common(20):
        print(f"  {k:28s} {v:6d}")
    print("=" * 80)
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
