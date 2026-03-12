"""
merge_sft_results.py — Merge step1 (Qwen3) and step2 (Gemini) chunk results
into final SFT training JSONL files.

Includes quality filters:
- Strip format instruction leakage from function_call values
- Filter think-loop trash (excessively long function_call)
- Filter samples with consecutive repeated same-test requests

Usage:
    python merge_sft_results.py \
        --dataset MedQA \
        --step1_dir outputs/MedQA/Qwen3-VL-8B-Instruct \
        --step2_dir outputs/MedQA/gemini-3-flash-preview \
        --output_dir training_data/MedQA
"""

import os
import sys
import re
import json
import glob
import random
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_utils import format_final_sft_sample

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Max allowed length for a single function_call value (chars).
# Think-loop samples typically exceed 5000 chars.
MAX_FUNCTION_CALL_LEN = 5000

# Max total character length for a sample (system + tools + conversations).
# Text-only: ~20k chars ≈ 5700 tokens, fits in cutoff_len=6144.
# With images: Qwen3-VL adds 256-1280 vision tokens per image, so text budget is lower.
MAX_SAMPLE_CHARS = 20000
MAX_SAMPLE_CHARS_IMAGE = 18000  # ~5100 text tokens + ~350 vision tokens (512px images) < 6144

# Max allowed consecutive identical test requests before flagging as trash
MAX_REPEAT_TESTS = 3


def load_all_chunks(results_dir):
    """Load all chunk_*.jsonl files from a directory. Returns list of records."""
    records = []
    jsonl_files = sorted(glob.glob(os.path.join(results_dir, "chunk_*.jsonl")))
    if not jsonl_files:
        logger.warning(f"No chunk files found in {results_dir}")
        return records

    for jf in jsonl_files:
        try:
            with open(jf, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to load {jf}: {e}")

    logger.info(f"Loaded {len(records)} records from {results_dir} ({len(jsonl_files)} files)")
    return records


def _fix_function_call_json(val):
    """
    Fix function_call value for LLaMA-Factory compatibility.
    LLaMA-Factory's FunctionFormatter strips <think>...</think> then json.loads() the rest.
    This function ensures:
    1. <think> tags are properly closed
    2. Stray </think> / code-block markers outside think block are removed
    3. Multiple JSON objects are wrapped in an array [...]
    """
    # Step 1: Close unclosed <think> — insert </think> before first JSON line
    if '<think>' in val and '</think>' not in val:
        lines = val.split('\n')
        json_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                json_start = i
                break
        if json_start >= 0:
            lines.insert(json_start, '</think>')
            val = '\n'.join(lines)
        else:
            # No JSON found at all — just close the think tag
            val = val + '\n</think>'

    # Step 2: Strip <think>...</think> to isolate JSON part
    if '<think>' in val and '</think>' in val:
        json_part = re.sub(r'<think>[\s\S]*?</think>', '', val).strip()
        think_match = re.search(r'<think>[\s\S]*?</think>', val)
        think_block = think_match.group(0) if think_match else ''
    else:
        json_part = val.strip()
        think_block = ''

    # Step 3: Clean stray markers from json_part
    json_part = json_part.replace('</think>', '')
    json_part = re.sub(r'```json\s*', '', json_part)
    json_part = re.sub(r'```\s*', '', json_part)
    json_part = json_part.strip()

    if not json_part:
        return val  # Nothing to fix, will be caught by quality filter

    # Step 4: Try parsing as-is
    try:
        json.loads(json_part)
        # Valid JSON — reconstruct with think block
        return f"{think_block}\n{json_part}".strip() if think_block else json_part
    except json.JSONDecodeError:
        pass

    # Step 5: Try wrapping line-by-line JSON objects into an array
    objs = []
    for line in json_part.split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # Skip non-JSON lines
    if objs:
        array_json = json.dumps(objs, ensure_ascii=False)
        return f"{think_block}\n{array_json}".strip() if think_block else array_json

    return val  # Unfixable — quality filter will catch it


def clean_conversations(conversations):
    """
    Clean conversations in-place:
    - Strip [END OF FORMAT INSTRUCTIONS] from function_call values
    - Wrap Thinking model output with proper <think> tags
    - Fix function_call JSON for LLaMA-Factory compatibility
    Returns the cleaned conversations list.
    """
    for conv in conversations:
        if conv.get("from") == "function_call":
            val = conv["value"]
            val = re.sub(r'\s*\[END OF FORMAT INSTRUCTIONS\]\s*$', '', val)
            # Wrap Thinking model output: has </think> but no <think>
            if '</think>' in val and '<think>' not in val:
                think_end = val.find('</think>')
                reasoning = val[:think_end].strip()
                action = val[think_end + len('</think>'):].strip()
                if reasoning and action:
                    val = f"<think>\n{reasoning}\n</think>\n{action}"
            # Fix JSON format for LLaMA-Factory
            val = _fix_function_call_json(val)
            conv["value"] = val.strip()
    return conversations


_THINK_GOAL_LINE = "You must reason step-by-step using <think>...</think> tags before each tool call."
_THINK_FORMAT = (
    "[BEGIN OF FORMAT INSTRUCTIONS]\n"
    "Your output must follow this format:\n"
    "<think>\n"
    "your clinical reasoning here\n"
    "</think>\n"
    "\n"
    "{\"name\": \"action_name\", \"arguments\": {\"arg\": \"value\"}}\n"
    "[END OF FORMAT INSTRUCTIONS]"
)
_NO_THINK_FORMAT = (
    "[BEGIN OF FORMAT INSTRUCTIONS]\n"
    "Your output must follow this format:\n"
    "{\"name\": \"action_name\", \"arguments\": {\"arg\": \"value\"}}\n"
    "[END OF FORMAT INSTRUCTIONS]"
)


def align_system_think(sample):
    """
    Ensure the system prompt's think instructions are consistent with the
    actual conversations: if no function_call turn has a <think> block,
    strip think requirements from the system prompt (and vice versa).
    Modifies sample in-place and returns it.
    """
    conv_has_think = any(
        t.get("from") == "function_call" and re.match(r"<think>", t.get("value", ""))
        for t in sample.get("conversations", [])
    )
    sys = sample.get("system", "")
    sys_has_think = bool(re.search(re.escape(_THINK_GOAL_LINE), sys))

    if sys_has_think and not conv_has_think:
        sys = sys.replace(_THINK_GOAL_LINE + "\n", "").replace(_THINK_GOAL_LINE, "")
        sys = sys.replace(_THINK_FORMAT, _NO_THINK_FORMAT)
        sample["system"] = sys
    return sample


def check_quality(record):
    """
    Check if a record passes quality filters.
    Returns (passes, reason) tuple.
    """
    conversations = record.get("conversations", [])
    if not conversations:
        return False, "empty_conversations"

    # Check total sample length to prevent OOM during training
    total_chars = sum(len(c.get("value", "")) for c in conversations)
    total_chars += len(record.get("system", "")) + len(record.get("tools", ""))
    char_limit = MAX_SAMPLE_CHARS_IMAGE if record.get("images") else MAX_SAMPLE_CHARS
    if total_chars > char_limit:
        return False, "too_long"

    for conv in conversations:
        if conv.get("from") == "function_call":
            val = conv.get("value", "")
            # Strip <think>...</think> before checking length —
            # legitimate reasoning content should not trigger think-loop filter
            val_no_think = re.sub(r'<think>[\s\S]*?</think>', '', val).strip()
            if len(val_no_think) > MAX_FUNCTION_CALL_LEN:
                return False, "think_loop"
            # Detect truly repetitive content (real think-loop indicator)
            if len(val) > 3000:
                sentences = [s.strip() for s in val.split('.') if len(s.strip()) > 20]
                if len(sentences) > 10:
                    unique = set(s.lower() for s in sentences)
                    if len(unique) < len(sentences) * 0.3:
                        return False, "think_loop"

    # Check for repeated same-test requests (consecutive)
    prev_test = None
    repeat_count = 0
    for conv in conversations:
        if conv.get("from") == "function_call":
            test_match = re.search(r'"test"\s*:\s*"([^"]+)"', conv.get("value", ""))
            if test_match:
                test_name = test_match.group(1)
                if test_name == prev_test:
                    repeat_count += 1
                    if repeat_count >= MAX_REPEAT_TESTS:
                        return False, "repeated_test"
                else:
                    repeat_count = 0
                prev_test = test_name
            else:
                prev_test = None
                repeat_count = 0

    # Validate function_call JSON is parseable and structurally valid
    # LLaMA-Factory's FunctionFormatter does json.loads() then accesses tc["name"]/tc["arguments"]
    for conv in conversations:
        if conv.get("from") == "function_call":
            val = conv.get("value", "")
            json_part = re.sub(r'<think>[\s\S]*?</think>', '', val).strip()
            if not json_part:
                return False, "invalid_json"
            try:
                parsed = json.loads(json_part)
                if not isinstance(parsed, list):
                    parsed = [parsed]
                for tc in parsed:
                    if not isinstance(tc, dict) or "name" not in tc or "arguments" not in tc:
                        return False, "invalid_json"
            except (json.JSONDecodeError, TypeError):
                return False, "invalid_json"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Merge SFT results from step1 and step2")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--step1_dir", type=str, required=True, help="Step 1 (Qwen3) results dir")
    parser.add_argument("--step2_dir", type=str, default=None, help="Step 2 (Gemini) results dir (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for final SFT JSONL")
    parser.add_argument("--qwen_ratio", type=float, default=None,
                        help="Max Qwen3 samples as a multiple of Gemini count. "
                             "E.g. 1.0 = keep at most 1x Gemini count of Qwen3 samples. "
                             "Default: None (keep all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")

    args = parser.parse_args()

    # Load step1 results
    step1_records = load_all_chunks(args.step1_dir)

    # Load step2 results (optional)
    step2_records = []
    if args.step2_dir and os.path.isdir(args.step2_dir):
        step2_records = load_all_chunks(args.step2_dir)

    # Collect correct samples with quality filtering — separately for Qwen3 and Gemini
    qwen_samples = []
    gemini_samples = []
    seen_ids = set()
    filter_stats = {"think_loop": 0, "repeated_test": 0, "empty_conversations": 0, "invalid_json": 0, "too_long": 0}

    # Step 1: Qwen3 correct answers
    qwen_total = len(step1_records)
    for record in step1_records:
        sid = record.get("scenario_id")
        if record.get("is_correct", False) and record.get("conversations"):
            record["conversations"] = clean_conversations(record["conversations"])
            passes, reason = check_quality(record)
            if not passes:
                filter_stats[reason] = filter_stats.get(reason, 0) + 1
                logger.debug(f"Filtered scenario {sid}: {reason}")
                continue
            qwen_samples.append(align_system_think(format_final_sft_sample(record)))
            seen_ids.add(sid)

    # Step 2: Gemini corrected answers (only for scenarios not already correct)
    gemini_total = len(step2_records)
    for record in step2_records:
        sid = record.get("scenario_id")
        if sid in seen_ids:
            continue  # Already have correct answer from Qwen3
        if record.get("is_correct", False) and record.get("conversations"):
            record["conversations"] = clean_conversations(record["conversations"])
            passes, reason = check_quality(record)
            if not passes:
                filter_stats[reason] = filter_stats.get(reason, 0) + 1
                logger.debug(f"Filtered scenario {sid}: {reason}")
                continue
            gemini_samples.append(align_system_think(format_final_sft_sample(record)))
            seen_ids.add(sid)

    qwen_correct = len(qwen_samples)
    gemini_correct = len(gemini_samples)

    # Apply qwen_ratio subsampling
    qwen_kept = qwen_correct
    if args.qwen_ratio is not None and gemini_correct > 0:
        max_qwen = int(gemini_correct * args.qwen_ratio)
        if qwen_correct > max_qwen:
            rng = random.Random(args.seed)
            qwen_samples = rng.sample(qwen_samples, max_qwen)
            qwen_kept = len(qwen_samples)
            logger.info(f"Qwen3 subsampled: {qwen_correct} -> {qwen_kept} (ratio={args.qwen_ratio}, base={gemini_correct})")

    sft_samples = qwen_samples + gemini_samples

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "sft_train.jsonl")

    with open(output_file, 'w') as f:
        for sample in sft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Print summary
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Step 1 (Qwen3): {qwen_correct}/{qwen_total} correct ({100*qwen_correct/max(qwen_total,1):.1f}%)")
    if qwen_kept != qwen_correct:
        logger.info(f"  -> Subsampled to {qwen_kept} (ratio={args.qwen_ratio}x of {gemini_correct} Gemini)")
    if gemini_total > 0:
        logger.info(f"Step 2 (Gemini): {gemini_correct}/{gemini_total} corrected ({100*gemini_correct/max(gemini_total,1):.1f}%)")
    if any(v > 0 for v in filter_stats.values()):
        logger.info(f"Quality filters applied: {filter_stats}")
    logger.info(f"Total SFT samples: {len(sft_samples)} (Qwen3: {qwen_kept}, Gemini: {gemini_correct})")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
