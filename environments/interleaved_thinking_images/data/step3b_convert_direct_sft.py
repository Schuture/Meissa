"""
Convert agentic SFT data (with tool calls) to direct VQA format.

For each sample in all_merged.jsonl:
  - Keep only the first human message (with <image> and question)
  - Extract the final answer (from Terminate args or last gpt [FINAL])
  - Output a simple 2-turn conversation: human + gpt
  - Keep only the first image (original)
  - Replace system prompt with a simple VQA prompt
  - Remove tools field
"""

import argparse
import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
FINAL_RE = re.compile(r"\[FINAL\]\s*(.*)", re.S)

DIRECT_SYSTEM_PROMPT = (
    "You are a medical visual question answering assistant. "
    "Answer the question based on the given image. Keep your answer concise."
)


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


def parse_function_call_value(value: str) -> Tuple[Optional[str], Dict[str, Any]]:
    if not isinstance(value, str):
        return None, {}
    end = value.rfind("}")
    if end == -1:
        return None, {}
    depth = 0
    start = None
    for i in range(end, -1, -1):
        ch = value[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                start = i
                break
    if start is None:
        return None, {}
    try:
        obj = json.loads(value[start:end + 1])
    except Exception:
        return None, {}
    name = obj.get("name")
    args = obj.get("arguments", {})
    if not isinstance(name, str) or not name.strip():
        return None, {}
    if not isinstance(args, dict):
        args = {}
    return name.strip(), args


def get_terminate_ans(item: Dict[str, Any]) -> str:
    for m in item.get("conversations", []):
        if m.get("from") == "function_call":
            name, args = parse_function_call_value(m.get("value", ""))
            if name == "Terminate":
                return normalize_yes_no(args.get("ans", ""))
            # legacy format
            if m.get("name") == "Terminate":
                a = m.get("arguments", {})
                if isinstance(a, dict):
                    return normalize_yes_no(a.get("ans", ""))
    return ""


def get_final_answer_from_gpt(item: Dict[str, Any]) -> str:
    gpt_msgs = [m for m in item.get("conversations", [])
                 if m.get("from") == "gpt" and isinstance(m.get("value"), str)]
    if not gpt_msgs:
        return ""
    last_text = gpt_msgs[-1]["value"]
    m_final = FINAL_RE.search(last_text)
    if m_final:
        return normalize_yes_no(m_final.group(1).strip())
    # fallback: strip <think> tags and return remaining
    cleaned = THINK_RE.sub("", last_text).strip()
    return normalize_yes_no(cleaned)


def get_first_human_message(item: Dict[str, Any]) -> Optional[str]:
    for m in item.get("conversations", []):
        if m.get("from") == "human":
            return m.get("value", "")
    return None


def convert_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    human_text = get_first_human_message(item)
    if not human_text:
        return None

    # Ensure exactly one <image> tag
    body = human_text.replace("<image>", "").lstrip()
    human_text = "<image>\n" + body

    # Extract final answer: prefer Terminate, fallback to last gpt
    answer = get_terminate_ans(item)
    if not answer:
        answer = get_final_answer_from_gpt(item)
    if not answer:
        return None

    # Keep only the first image (original)
    images = item.get("images", [])
    if not images:
        return None
    first_image = images[0]

    return {
        "conversations": [
            {"from": "human", "value": human_text},
            {"from": "gpt", "value": answer},
        ],
        "images": [first_image],
        "system": DIRECT_SYSTEM_PROMPT,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert agentic SFT data to direct VQA format (single-turn)."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input jsonl (e.g. all_merged.jsonl)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output jsonl")
    args = parser.parse_args()

    converted = []
    skipped = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Converting"):
            if not line.strip():
                continue
            item = json.loads(line)
            result = convert_item(item)
            if result:
                converted.append(result)
            else:
                skipped += 1

    with open(args.output, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted: {len(converted)}")
    print(f"Skipped (no answer or no image): {skipped}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
