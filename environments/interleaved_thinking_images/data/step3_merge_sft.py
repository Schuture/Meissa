import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm


THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
FINAL_RE = re.compile(r"\[FINAL\]\s*(.*)", re.S)
IMAGE_TAG_RE = re.compile(r"<image>")
OUTPUT_IMAGE_ID_RE = re.compile(r"\[Output Image ID:\s*img_round_\d+\]", re.S)

DEFAULT_TERMINATE_MESSAGE = "now you have the answer, output the message to human"
DEFAULT_FINAL_THINK = "I will answer now"

SYMPTOM_KEYWORDS = [
    "cough", "fever", "shortness of breath", "dyspnea",
    "chest pain", "symptom", "history of", "chills",
    "weight loss", "night sweats", "clinical history"
]


# --------------------------
# IO
# --------------------------
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_correct_idxs(per_sample_path: str) -> set:
    print(f"Loading scores from: {per_sample_path}")
    correct_idxs = set()
    total = 0
    try:
        with open(per_sample_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                try:
                    item = json.loads(line)
                    if float(item.get("score", 0.0)) == 1.0:
                        correct_idxs.add(item["idx"])
                except Exception as e:
                    print(f"Error parsing line in per_sample: {e}")
    except FileNotFoundError:
        print(f"Error: File not found {per_sample_path}")
        return set()

    print(f"  - Total samples scanned: {total}")
    print(f"  - Correct samples (score=1.0): {len(correct_idxs)}")
    return correct_idxs


def extract_sft_data(sft_path: str, target_idxs: set, source_name: str = "Unknown") -> List[Dict[str, Any]]:
    print(f"Extracting SFT data from: {sft_path}")
    extracted_data = []
    try:
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Scanning {source_name}"):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    idx = (item.get("meta", {}) or {}).get("idx")
                    if idx is not None and idx in target_idxs:
                        extracted_data.append(item)
                except Exception:
                    pass
    except FileNotFoundError:
        print(f"Error: File not found {sft_path}")
        return []

    print(f"  - Extracted valid SFT trajectories: {len(extracted_data)}")
    return extracted_data


# --------------------------
# Text helpers
# --------------------------
def normalize_yes_no(ans: str) -> str:
    if ans is None:
        return ""
    s = ans.strip()
    if not s:
        return ""
    low = s.lower()
    if low == "yes":
        return "Yes"
    if low == "no":
        return "No"
    return s


def parse_thought_and_answer(text: str) -> Tuple[str, str]:
    thought = ""
    answer = ""
    if not text:
        return thought, answer

    m_think = THINK_RE.search(text)
    if m_think:
        thought = m_think.group(1).strip()

    m_final = FINAL_RE.search(text)
    if m_final:
        answer = m_final.group(1).strip()
    else:
        answer = THINK_RE.sub("", text).strip()

    answer = normalize_yes_no(answer)
    return thought, answer


def make_final_gpt_text(answer: str, final_think: str = DEFAULT_FINAL_THINK) -> str:
    answer = normalize_yes_no(answer or "")
    final_think = (final_think or "").strip()
    return f"<think>\n{final_think}\n</think>\n[FINAL] {answer}".strip()


def maybe_replace_system_prompt(item: Dict[str, Any], new_system_prompt: Optional[str]) -> Dict[str, Any]:
    if new_system_prompt is not None:
        item["system"] = new_system_prompt
    return item


# --------------------------
# Parse tool json from function_call.value
# --------------------------
def find_last_json_object_in_text(s: str) -> Optional[str]:
    if not s:
        return None
    end = s.rfind("}")
    if end == -1:
        return None

    depth = 0
    start = None
    for i in range(end, -1, -1):
        ch = s[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                start = i
                break

    if start is None:
        return None

    cand = s[start:end + 1].strip()
    if cand.startswith("{") and cand.endswith("}"):
        return cand
    return None


def parse_function_call_value(value: str) -> Tuple[Optional[str], Dict[str, Any], str]:
    if not isinstance(value, str):
        return None, {}, ""

    thought = ""
    mt = THINK_RE.search(value)
    if mt:
        thought = mt.group(1).strip()

    js = find_last_json_object_in_text(value)
    if not js:
        return None, {}, thought

    try:
        obj = json.loads(js)
    except Exception:
        return None, {}, thought

    name = obj.get("name", None)
    args = obj.get("arguments", None)
    if not isinstance(name, str) or not name.strip():
        return None, {}, thought
    if not isinstance(args, dict):
        args = {}
    return name.strip(), args, thought


def get_tool_name_and_args(msg: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    if msg.get("from") != "function_call":
        return None, {}

    v = msg.get("value", None)
    if isinstance(v, str) and v.strip():
        name, args, _ = parse_function_call_value(v)
        if name:
            return name, args

    name = msg.get("name", None)
    args = msg.get("arguments", None)
    if isinstance(name, str) and name.strip():
        return name.strip(), args if isinstance(args, dict) else {}

    return None, {}


def to_value_style_function_call(tool_name: str, arguments: Dict[str, Any], thought: str = "") -> Dict[str, Any]:
    tool_name = tool_name.strip()
    payload = {"name": tool_name, "arguments": arguments if isinstance(arguments, dict) else {}}
    json_part = json.dumps(payload, ensure_ascii=False)

    thought = (thought or "").strip()
    if thought:
        value = f"<think>\n{thought}\n</think>\n\n{json_part}"
    else:
        value = f"{json_part}"
    return {"from": "function_call", "value": value}


# --------------------------
# Strict image alignment (your requested rule)
# --------------------------
def count_image_tags(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(IMAGE_TAG_RE.findall(text))


def normalize_human_image_tags(conv: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Enforce: first human message has exactly 1 <image> tag.
    """
    for i, m in enumerate(conv):
        if m.get("from") == "human":
            text = m.get("value", "") or ""
            # Remove all <image> tags, then add exactly one at the beginning.
            body = text.replace("<image>", "").lstrip()
            conv[i]["value"] = "<image>\n" + body
            return True, "ok"
    return False, "no_human"


def is_image_observation(msg: Dict[str, Any]) -> bool:
    """
    Observation is considered image-related if:
      - it already contains "<image>"
      - OR it includes "[Output Image ID: img_round_N]"
    """
    if msg.get("from") != "observation":
        return False
    v = msg.get("value", "")
    if not isinstance(v, str):
        return False
    if "<image>" in v:
        return True
    if OUTPUT_IMAGE_ID_RE.search(v):
        return True
    if "segmentation completed" in v.lower():
        return True
    return False


def ensure_n_image_tags_in_observation(msg: Dict[str, Any], n_tags: int) -> None:
    v = msg.get("value", "")
    if not isinstance(v, str):
        v = ""

    body = v.replace("<image>", "").lstrip()

    if n_tags <= 0:
        msg["value"] = body
        return

    prefix = "<image>\n" * n_tags
    msg["value"] = prefix + body


def align_images_strict(item: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Rule:
      - Exactly 1 <image> in first human message.
      - Remaining images must be represented by <image> tags placed in observation messages.
      - Total <image> tags across (human + observations) must equal len(images).
    """
    imgs = item.get("images", [])
    if not isinstance(imgs, list):
        item["images"] = []
        return item, "rejected_images_not_list"
    n_imgs = len(imgs)
    if n_imgs <= 0:
        return item, "rejected_no_images"

    conv = item.get("conversations", [])
    if not isinstance(conv, list) or not conv:
        return item, "rejected_empty_conversation"

    ok, why = normalize_human_image_tags(conv)
    if not ok:
        return item, f"rejected_{why}"

    need_obs_tags = n_imgs - 1
    obs_idxs = [i for i, m in enumerate(conv) if is_image_observation(m)]

    if len(obs_idxs) < need_obs_tags:
        return item, "rejected_not_enough_image_observations"

    for j in range(need_obs_tags):
        idx = obs_idxs[j]
        ensure_n_image_tags_in_observation(conv[idx], 1)

    for j in range(need_obs_tags, len(obs_idxs)):
        idx = obs_idxs[j]
        ensure_n_image_tags_in_observation(conv[idx], 0)

    total_tags = 0
    for m in conv:
        if m.get("from") in ("human", "observation"):
            total_tags += count_image_tags(m.get("value", ""))

    if total_tags != n_imgs:
        return item, "rejected_image_tag_count_still_mismatch"

    item["conversations"] = conv
    return item, "ok"


# --------------------------
# Enforce Terminate + final reply (move long think into Terminate)
# --------------------------
def ensure_terminate_and_final(
    item: Dict[str, Any],
    terminate_message: str = DEFAULT_TERMINATE_MESSAGE,
    final_think: str = DEFAULT_FINAL_THINK,
) -> Dict[str, Any]:
    conv = item.get("conversations", [])
    if not conv:
        return item

    # Normalize legacy Terminate (name/arguments) to value-style
    for i, m in enumerate(conv):
        if m.get("from") == "function_call" and m.get("name") == "Terminate":
            args = m.get("arguments", {}) or {}
            ans = normalize_yes_no(args.get("ans", "")) if isinstance(args, dict) else ""
            conv[i] = to_value_style_function_call("Terminate", {"ans": ans}, thought="")
    item["conversations"] = conv
    conv = item["conversations"]

    # If last is gpt: move its long think into Terminate
    if conv and conv[-1].get("from") == "gpt":
        text = conv[-1].get("value", "") or ""
        long_thought, ans = parse_thought_and_answer(text)
        if ans:
            conv = conv[:-1]
            conv.append(to_value_style_function_call("Terminate", {"ans": ans}, thought=long_thought))
            conv.append({"from": "observation", "value": terminate_message})
            conv.append({"from": "gpt", "value": make_final_gpt_text(ans, final_think=final_think)})
            item["conversations"] = conv
        return item

    # Else: ensure Terminate exists and has observation+gpt after it
    t_idx = -1
    t_args = {}
    t_thought = ""
    for i in range(len(conv) - 1, -1, -1):
        if conv[i].get("from") == "function_call":
            name, args = get_tool_name_and_args(conv[i])
            if name == "Terminate":
                t_idx = i
                v = conv[i].get("value", "")
                if isinstance(v, str):
                    _, a2, th2 = parse_function_call_value(v)
                    if a2:
                        args = a2
                    t_thought = th2 or ""
                t_args = args
                break

    if t_idx == -1:
        item["conversations"] = conv
        return item

    ans = normalize_yes_no(t_args.get("ans", ""))
    conv[t_idx] = to_value_style_function_call("Terminate", {"ans": ans}, thought=t_thought)

    insert_pos = t_idx + 1
    if insert_pos >= len(conv) or conv[insert_pos].get("from") != "observation":
        conv.insert(insert_pos, {"from": "observation", "value": terminate_message})
        insert_pos += 1
    else:
        conv[insert_pos]["value"] = terminate_message
        insert_pos += 1

    if insert_pos >= len(conv) or conv[insert_pos].get("from") != "gpt":
        conv.insert(insert_pos, {"from": "gpt", "value": make_final_gpt_text(ans, final_think=final_think)})
    else:
        conv[insert_pos]["value"] = make_final_gpt_text(ans, final_think=final_think)

    item["conversations"] = conv
    return item

def hard_template_no_tool_first_think(item, template="NO_TOOL The question can be answered directly without tools."):
    conv = item.get("conversations", [])
    if not conv:
        return item

    # 判定是不是 NO_TOOL
    if tool_call_count_non_terminate(item) > 0:
        return item  # USE_TOOL，不动

    for i, m in enumerate(conv):
        v = m.get("value", None)
        if not isinstance(v, str):
            continue
        if "<think>" not in v:
            continue

        def _repl(match):
            return f"<think>\n{template}\n</think>"

        new_v, n = THINK_RE.subn(_repl, v, count=1)
        if n > 0:
            conv[i]["value"] = new_v
            break

    item["conversations"] = conv
    return item


# --------------------------
# Filters / checks
# --------------------------
def canonical_tool_sig(name: str, arguments: Any) -> str:
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {"_raw": str(arguments)}
    arg_str = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    return f"{name}::{arg_str}"


def has_consecutive_duplicate_tool_calls(item: Dict[str, Any]) -> bool:
    conv = item.get("conversations", [])
    prev_sig = None
    for m in conv:
        if m.get("from") == "function_call":
            name, args = get_tool_name_and_args(m)
            if not name:
                prev_sig = None
                continue
            sig = canonical_tool_sig(name, args)
            if prev_sig is not None and sig == prev_sig:
                return True
            prev_sig = sig
        else:
            prev_sig = None
    return False


def extract_question_text(item: Dict[str, Any]) -> str:
    for m in item.get("conversations", []):
        if m.get("from") == "human":
            v = (m.get("value", "") or "").replace("<image>", "").strip()
            return v.lower()
    return ""


def contains_hallucinated_symptoms_in_gpt(item: Dict[str, Any]) -> bool:
    q = extract_question_text(item)

    def is_hallucination(text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        for kw in SYMPTOM_KEYWORDS:
            if kw in t and kw not in q:
                return True
        return False

    for m in item.get("conversations", []):
        if m.get("from") == "gpt":
            text = m.get("value", "") or ""
            pre = text.split("[FINAL]")[0] if "[FINAL]" in text else text
            mt = THINK_RE.search(pre)
            thought_part = mt.group(1) if mt else pre
            if is_hallucination(thought_part):
                return True
    return False


def total_char_len(item: Dict[str, Any]) -> int:
    total = 0
    for m in item.get("conversations", []):
        v = m.get("value", None)
        if isinstance(v, str):
            total += len(v)
    sys = item.get("system", None)
    if isinstance(sys, str):
        total += len(sys)
    return total


def get_terminate_ans(item: Dict[str, Any]) -> str:
    for m in item.get("conversations", []):
        if m.get("from") == "function_call":
            name, args = get_tool_name_and_args(m)
            if name == "Terminate":
                return normalize_yes_no(args.get("ans", ""))
    return ""


def strict_checks(item: Dict[str, Any]) -> Tuple[bool, str]:
    conv = item.get("conversations", [])
    if not conv:
        return False, "empty_conversation"

    term_positions = []
    for i, m in enumerate(conv):
        if m.get("from") == "function_call":
            name, _ = get_tool_name_and_args(m)
            if name == "Terminate":
                term_positions.append(i)
    if len(term_positions) != 1:
        return False, f"terminate_count_{len(term_positions)}"

    t_ans = get_terminate_ans(item)
    if not t_ans:
        return False, "terminate_ans_empty"

    gpt_msgs = [m for m in conv if m.get("from") == "gpt" and isinstance(m.get("value", None), str)]
    if not gpt_msgs:
        return False, "no_gpt_message"
    last_gpt_text = gpt_msgs[-1]["value"]
    _, final_ans = parse_thought_and_answer(last_gpt_text)
    if not final_ans:
        return False, "final_missing"
    if normalize_yes_no(final_ans) != normalize_yes_no(t_ans):
        return False, "final_not_match_terminate"

    # enforce terminate args only {"ans": ...} inside the json
    t_pos = term_positions[0]
    v = conv[t_pos].get("value", "")
    name, args, _ = parse_function_call_value(v if isinstance(v, str) else "")
    if name != "Terminate":
        return False, "terminate_parse_failed"
    if set(args.keys()) != {"ans"}:
        return False, "terminate_args_not_only_ans"

    # enforce (human+observation) <image> count equals len(images)
    imgs = item.get("images", [])
    if not isinstance(imgs, list) or len(imgs) == 0:
        return False, "no_images"
    total_tags = 0
    for m in conv:
        if m.get("from") in ("human", "observation"):
            total_tags += count_image_tags(m.get("value", ""))
    if total_tags != len(imgs):
        return False, "image_tag_count_mismatch"

    return True, "ok"


# --------------------------
# Tool tag insertion: USE_TOOL / NO_TOOL into the first <think> content
# --------------------------
def tool_call_names(item: Dict[str, Any]) -> List[str]:
    names = []
    for m in item.get("conversations", []):
        if m.get("from") == "function_call":
            name, _ = get_tool_name_and_args(m)
            if name:
                names.append(name)
    return names


def tool_call_count_non_terminate(item: Dict[str, Any]) -> int:
    return sum(1 for n in tool_call_names(item) if n != "Terminate")


def add_tool_use_tag_to_first_think(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert "USE_TOOL " or "NO_TOOL " at the beginning of the first <think>...</think> content
    across the whole conversation (in order).
    """
    conv = item.get("conversations", [])
    if not isinstance(conv, list) or not conv:
        return item

    tag = "USE_TOOL" if tool_call_count_non_terminate(item) > 0 else "NO_TOOL"

    for i, m in enumerate(conv):
        v = m.get("value", None)
        if not isinstance(v, str) or "<think>" not in v:
            continue

        def _repl(match: re.Match) -> str:
            inner = match.group(1) or ""
            inner_stripped = inner.lstrip()
            # Avoid double-tagging
            if inner_stripped.startswith("USE_TOOL") or inner_stripped.startswith("NO_TOOL"):
                return f"<think>{inner}</think>"
            # Keep original indentation after tag if any
            # Put tag at the very beginning of content
            new_inner = f"{tag} {inner_stripped}"
            # Preserve leading whitespace/newlines by replacing stripped version
            prefix_len = len(inner) - len(inner_stripped)
            prefix = inner[:prefix_len] if prefix_len > 0 else ""
            return f"<think>{prefix}{new_inner}</think>"

        new_v, n = THINK_RE.subn(_repl, v, count=1)
        if n > 0:
            conv[i]["value"] = new_v
            break

    item["conversations"] = conv
    return item


# --------------------------
# Stats
# --------------------------
def tool_pattern(item: Dict[str, Any]) -> str:
    names = tool_call_names(item)
    if not names:
        return "no_tool"
    return " -> ".join(names)


def print_stats(items: List[Dict[str, Any]]):
    ans_counter = Counter()
    tool_count_counter = Counter()
    pattern_counter = Counter()

    for it in items:
        t_ans = get_terminate_ans(it)
        if t_ans:
            ans_counter[t_ans] += 1
        tool_count_counter[tool_call_count_non_terminate(it)] += 1
        pattern_counter[tool_pattern(it)] += 1

    print("\n=== Stats: Answer Distribution (Top 10) ===")
    for ans, c in ans_counter.most_common(10):
        print(f"{ans}\t{c}")

    print("\n=== Stats: Tool Call Count Distribution (non-Terminate) ===")
    for k in sorted(tool_count_counter.keys()):
        print(f"{k}\t{tool_count_counter[k]}")

    print("\n=== Stats: Tool Call Pattern (Top 5, includes Terminate) ===")
    for p, c in pattern_counter.most_common(5):
        print(f"{p}\t{c}")


# --------------------------
# Source handling (NEW): multiple qwen/gemini sources with per-source limit
# --------------------------
def parse_sources_from_args(args) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """
    Returns:
      qwen_sources:   list of (score_path, sft_path, limit)
      gemini_sources: list of (score_path, sft_path, limit)

    Supports new flags:
      --qwen SCORE SFT LIMIT   (repeatable)
      --gemini SCORE SFT LIMIT (repeatable)

    Also supports legacy single-source flags for backward compatibility:
      --qwen_score_path --qwen_sft_path --qwen_limit
      --gemini_score_path --gemini_sft_path --gemini_limit
    """
    qwen_sources: List[Tuple[str, str, int]] = []
    gemini_sources: List[Tuple[str, str, int]] = []

    # New style
    if args.qwen:
        for score_path, sft_path, limit_str in args.qwen:
            qwen_sources.append((score_path, sft_path, int(limit_str)))
    if args.gemini:
        for score_path, sft_path, limit_str in args.gemini:
            gemini_sources.append((score_path, sft_path, int(limit_str)))

    # Legacy fallback (only if no new sources provided)
    if not qwen_sources:
        if args.qwen_score_path and args.qwen_sft_path:
            qwen_sources.append((args.qwen_score_path, args.qwen_sft_path, int(args.qwen_limit)))
    if not gemini_sources:
        if args.gemini_score_path and args.gemini_sft_path:
            gemini_sources.append((args.gemini_score_path, args.gemini_sft_path, int(args.gemini_limit)))

    return qwen_sources, gemini_sources


def load_and_limit_one_source(score_path: str, sft_path: str, limit: int, source_name: str, seed: int) -> List[Dict[str, Any]]:
    correct = load_correct_idxs(score_path)
    data = extract_sft_data(sft_path, correct, source_name)
    if limit > 0 and len(data) > limit:
        random.Random(seed).shuffle(data)
        data = data[:limit]
    return data


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Merge Qwen/Gemini SFT (multi-source); strict image alignment; normalize Terminate; add tool-use tag; filter; stats."
    )

    # New multi-source arguments
    parser.add_argument(
        "--qwen",
        action="append",
        nargs=3,
        metavar=("SCORE_PATH", "SFT_PATH", "LIMIT"),
        help="Add one Qwen3 source: SCORE_PATH SFT_PATH LIMIT. Repeatable. LIMIT=-1 means no limit.",
    )
    parser.add_argument(
        "--gemini",
        action="append",
        nargs=3,
        metavar=("SCORE_PATH", "SFT_PATH", "LIMIT"),
        help="Add one Gemini source: SCORE_PATH SFT_PATH LIMIT. Repeatable. LIMIT=-1 means no limit.",
    )

    # Legacy single-source arguments (kept for backward compatibility)
    parser.add_argument("--qwen_score_path", type=str, default="")
    parser.add_argument("--qwen_sft_path", type=str, default="")
    parser.add_argument("--qwen_limit", type=int, default=-1)

    parser.add_argument("--gemini_score_path", type=str, default="")
    parser.add_argument("--gemini_sft_path", type=str, default="")
    parser.add_argument("--gemini_limit", type=int, default=-1)

    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--system_prompt_path", type=str, default="")
    parser.add_argument("--terminate_message", type=str, default=DEFAULT_TERMINATE_MESSAGE)
    parser.add_argument("--final_think", type=str, default=DEFAULT_FINAL_THINK)

    parser.add_argument("--max_chars", type=int, default=8000)
    parser.add_argument("--template_simple_think", action=argparse.BooleanOptionalAction, default=False,
                        help="Replace NO_TOOL samples' first <think> with a fixed template (default: False). Use --no_template_simple_think to disable.")

    args = parser.parse_args()
    random.seed(args.seed)

    qwen_sources, gemini_sources = parse_sources_from_args(args)

    if not qwen_sources:
        raise ValueError("No Qwen sources provided. Use --qwen ... or legacy --qwen_score_path/--qwen_sft_path.")
    if not gemini_sources:
        raise ValueError("No Gemini sources provided. Use --gemini ... or legacy --gemini_score_path/--gemini_sft_path.")

    new_system_prompt = None
    if args.system_prompt_path:
        if not os.path.exists(args.system_prompt_path):
            raise FileNotFoundError(f"system_prompt_path not found: {args.system_prompt_path}")
        new_system_prompt = load_text(args.system_prompt_path)

    fix_counter = Counter()

    def fix_all(data: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
        fixed = []
        local_ctr = Counter()
        for it in tqdm(data, desc=f"Fixing schema ({tag})"):
            it = maybe_replace_system_prompt(it, new_system_prompt)

            it, status = align_images_strict(it)
            local_ctr[status] += 1
            if status != "ok":
                continue

            it = ensure_terminate_and_final(
                it,
                terminate_message=args.terminate_message,
                final_think=args.final_think,
            )

            # Optionally template NO_TOOL samples' first <think>, then add USE_TOOL/NO_TOOL tag
            if args.template_simple_think:
                it = hard_template_no_tool_first_think(it)
            it = add_tool_use_tag_to_first_think(it)

            fixed.append(it)

        print(f"\n[{tag}] Image alignment stats:")
        for k, v in local_ctr.most_common():
            print(f"{k}\t{v}")
        fix_counter.update(local_ctr)
        return fixed

    # Load multi-sources
    qwen_all: List[Dict[str, Any]] = []
    print("\n=== Processing Qwen sources ===")
    for si, (score_path, sft_path, limit) in enumerate(qwen_sources):
        name = f"Qwen[{si}]"
        print(f"\n--- {name} ---")
        print(f"score: {score_path}")
        print(f"sft:   {sft_path}")
        print(f"limit: {limit}")
        q_data = load_and_limit_one_source(score_path, sft_path, limit, name, seed=args.seed + si)
        q_data = fix_all(q_data, name)
        qwen_all.extend(q_data)

    gemini_all: List[Dict[str, Any]] = []
    print("\n=== Processing Gemini sources ===")
    for si, (score_path, sft_path, limit) in enumerate(gemini_sources):
        name = f"Gemini[{si}]"
        print(f"\n--- {name} ---")
        print(f"score: {score_path}")
        print(f"sft:   {sft_path}")
        print(f"limit: {limit}")
        g_data = load_and_limit_one_source(score_path, sft_path, limit, name, seed=args.seed + 1000 + si)
        g_data = fix_all(g_data, name)
        gemini_all.extend(g_data)

    merged = qwen_all + gemini_all
    random.shuffle(merged)

    reasons = Counter()
    kept = []

    for it in tqdm(merged, desc="Filtering"):
        ok, reason = strict_checks(it)
        if not ok:
            reasons[reason] += 1
            continue

        if has_consecutive_duplicate_tool_calls(it):
            reasons["dup_consecutive_tool_call"] += 1
            continue

        if contains_hallucinated_symptoms_in_gpt(it):
            reasons["hallucinated_symptoms"] += 1
            continue

        if total_char_len(it) > args.max_chars:
            reasons["too_long"] += 1
            continue

        kept.append(it)

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for it in tqdm(kept, desc="Writing"):
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\nSaved: {args.output_path}")
    print(f"Before filtering: {len(merged)}")
    print(f"After filtering:  {len(kept)}")

    print("\n=== Fix/Alignment Summary (All sources) ===")
    for k, v in fix_counter.most_common():
        print(f"{k}\t{v}")

    print("\n=== Filter Rejection Reasons ===")
    for k, v in reasons.most_common():
        print(f"{k}\t{v}")

    print_stats(kept)


if __name__ == "__main__":
    main()
