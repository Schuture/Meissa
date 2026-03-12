#!/usr/bin/env python3
"""
generate_retrospective_data.py — Replace <think> blocks in SFT traces with hindsight reasoning.

Reads validated samples from training_data/{dataset}/sft_train.jsonl,
calls Gemini to generate retrospective reasoning for each function_call turn
(replacing the original forward-looking <think> block),
and writes enriched samples to outputs/{dataset}/{model}_retrospective/chunk_{idx}.jsonl.

Usage (quick test):
    python generate_retrospective_data.py \
        --input_file training_data/MedQA_Train/sft_train.jsonl \
        --output_dir outputs/MedQA_Train/gemini-3-flash-preview_retrospective \
        --total_chunks 1 --chunk_idx 0 --num_samples 3

Usage (SLURM):
    sbatch run_retrospective_gemini.sbatch
"""

import os
import sys
import json
import re
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from sft_utils import configure_gemini, query_gemini_with_retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================

def strip_think(value):
    """Remove leading <think>...</think> block from a function_call value.
    Returns the bare JSON string."""
    return re.sub(r'^<think>.*?</think>\s*', '', value, flags=re.DOTALL).strip()


def extract_ground_truth(conversations):
    """Extract the ground truth diagnosis from the Terminate call in conversations."""
    for turn in conversations:
        if turn['from'] == 'function_call':
            bare = strip_think(turn['value'])
            try:
                parsed = json.loads(bare)
                if parsed.get('name') == 'Terminate':
                    return parsed.get('arguments', {}).get('diagnosis', '')
            except Exception:
                continue
    # Fallback: try gpt turn "[FINAL] ..."
    for turn in conversations:
        if turn['from'] == 'gpt':
            m = re.search(r'\[FINAL\]\s*(.+)', turn['value'])
            if m:
                return m.group(1).strip()
    return ''


# ============================================================
# Hindsight Prompt Building
# ============================================================

HINDSIGHT_SYSTEM = (
    "You are assisting a medical AI agent in generating retrospective reasoning. "
    "You will be given a complete diagnostic trajectory and asked to write hindsight "
    "reasoning for a specific step — reasoning that reflects knowing the outcome and "
    "all future observations."
)


def format_trajectory(conversations, ground_truth):
    """Format all turns into a readable numbered trajectory string."""
    lines = []
    step = 0
    i = 0
    while i < len(conversations):
        turn = conversations[i]
        if turn['from'] == 'human':
            i += 1
            continue
        elif turn['from'] == 'function_call':
            step += 1
            bare = strip_think(turn['value'])
            try:
                tool_json = json.loads(bare)
                tool_name = tool_json.get('name', '?')
                args = tool_json.get('arguments', {})
                args_str = ', '.join(f'{k}="{v}"' for k, v in args.items())
            except Exception:
                tool_name = '?'
                args_str = bare[:80]
            obs = ''
            if i + 1 < len(conversations) and conversations[i + 1]['from'] == 'observation':
                obs_full = conversations[i + 1]['value']
                obs = obs_full[:500] + ('...' if len(obs_full) > 500 else '')
            lines.append(f'Step {step}: {tool_name}({args_str})')
            if obs:
                lines.append(f'  → {obs}')
        elif turn['from'] == 'gpt':
            lines.append(f'Final: {turn["value"]}')
        i += 1
    lines.append(f'\nCorrect Diagnosis: {ground_truth}')
    return '\n'.join(lines)


def build_hindsight_prompt(sample, fc_index, fc_step_num, ground_truth):
    """Build the Gemini user prompt for hindsight reasoning at function_call index fc_index."""
    conversations = sample['conversations']

    human_value = next(
        (t['value'] for t in conversations if t['from'] == 'human'), ''
    )

    # Current tool call — bare JSON
    bare_json = strip_think(conversations[fc_index]['value'])
    try:
        tool_json = json.loads(bare_json)
        tool_name = tool_json.get('name', '?')
        args = tool_json.get('arguments', {})
        args_str = ', '.join(f'{k}="{v}"' for k, v in args.items())
        current_call_str = f'{tool_name}({args_str})'
    except Exception:
        tool_name = '?'
        current_call_str = bare_json

    # Future context: turns after fc_index
    future_lines = []
    for t in conversations[fc_index + 1:]:
        if t['from'] == 'observation':
            obs_val = t['value']
            obs_short = obs_val[:600] + ('...' if len(obs_val) > 600 else '')
            future_lines.append(f'Observation: {obs_short}')
        elif t['from'] == 'function_call':
            bare = strip_think(t['value'])
            try:
                tj = json.loads(bare)
                tn = tj.get('name', '?')
                ta = tj.get('arguments', {})
                ta_str = ', '.join(f'{k}="{v}"' for k, v in ta.items())
                future_lines.append(f'Next call: {tn}({ta_str})')
            except Exception:
                future_lines.append(f'Next call: {bare[:100]}')
        elif t['from'] == 'gpt':
            future_lines.append(f'Final diagnosis: {t["value"]}')

    future_text = '\n'.join(future_lines) if future_lines else '(none — this is the final step)'
    trajectory = format_trajectory(conversations, ground_truth)

    return f"""[CORRECT DIAGNOSIS — for your reference only, do NOT mention it explicitly in your output]
{ground_truth}

[PATIENT PRESENTATION]
{human_value}

[FULL DIAGNOSTIC TRAJECTORY]
{trajectory}

[CURRENT STEP: Step {fc_step_num}]
The agent called: {current_call_str}

The observations and actions that followed this step were:
{future_text}

Task: Write the reasoning for Step {fc_step_num} ({tool_name}).
Write in first person as the AI diagnostic agent (use "I"), in present tense, as if you are the agent
in the moment deciding to take this action — NOT as a retrospective summary of what happened.
Use your hindsight knowledge (the full trajectory and correct diagnosis) to write reasoning that
is more focused and clinically precise than naive forward-looking speculation would be, but
the voice must sound like real-time thinking, not a case report written after the fact.

The reasoning should:
1. State why I am choosing this specific action right now, given what I know so far
2. Describe what I expect this step to reveal and why it matters for narrowing the differential
3. Do NOT describe the results of this step (those come in the observation that follows)
4. Do NOT name or hint at the final diagnosis — reason about differentials and clinical logic
5. Be medically precise and concise (3–5 sentences)

Output ONLY the reasoning text (no <think> tags, no JSON, no headers)."""


# ============================================================
# Per-sample Processing
# ============================================================

def add_hindsight_thoughts(sample, genai, gemini_model, max_tokens):
    """
    Replace <think> blocks in all function_call turns with hindsight reasoning.
    Returns a new sample dict with updated conversations.
    """
    conversations = sample['conversations']
    ground_truth = extract_ground_truth(conversations)
    if not ground_truth:
        logger.warning(f"Scenario {sample.get('meta', {}).get('scenario_id')}: could not extract ground truth, skipping.")
        return None

    new_conversations = []
    fc_step_num = 0

    for i, turn in enumerate(conversations):
        if turn['from'] != 'function_call':
            new_conversations.append(turn)
            continue

        fc_step_num += 1
        bare_json = strip_think(turn['value'])

        # Only process turns with a valid single tool-call dict
        try:
            parsed = json.loads(bare_json)
            if not isinstance(parsed, dict) or 'name' not in parsed:
                raise ValueError("not a tool call dict")
        except (json.JSONDecodeError, ValueError):
            logger.debug(
                f"Scenario {sample.get('meta', {}).get('scenario_id')} step {fc_step_num}: "
                f"not a valid tool call, keeping original."
            )
            new_conversations.append(turn)
            continue

        try:
            prompt = build_hindsight_prompt(sample, i, fc_step_num, ground_truth)
            messages = [{'role': 'user', 'parts': [prompt]}]
            hindsight_text = query_gemini_with_retry(
                genai, gemini_model, HINDSIGHT_SYSTEM, messages,
                max_tokens=max_tokens, temperature=0.7, max_retries=8,
                fatal_on_429=True,
            )
            hindsight_text = hindsight_text.strip()
            new_value = f'<think>\n{hindsight_text}\n</think>\n{bare_json}'
        except Exception as e:
            logger.warning(
                f"Scenario {sample.get('meta', {}).get('scenario_id')} step {fc_step_num}: "
                f"Gemini failed ({e}). Keeping original."
            )
            new_value = turn['value']

        new_conversations.append({'from': 'function_call', 'value': new_value})

    result = {
        'conversations': new_conversations,
        'system': sample['system'],
        'tools': sample['tools'],
        'meta': {**sample['meta'], 'retrospective': True},
    }
    if sample.get('images'):
        result['images'] = sample['images']
    return result


# ============================================================
# I/O helpers
# ============================================================

def load_sft_file(input_file, filter_source=None):
    """Load samples from a single JSONL file, optionally filtering by meta.source."""
    samples = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if filter_source and s.get('meta', {}).get('source') != filter_source:
                continue
            samples.append(s)
    src_msg = f" (source={filter_source})" if filter_source else ""
    logger.info(f"Loaded {len(samples)} samples from {input_file}{src_msg}")
    return samples


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate retrospective traces by replacing <think> blocks with hindsight reasoning via Gemini"
    )
    parser.add_argument('--input_file', required=True,
                        help='Path to validated SFT JSONL (e.g. training_data/MedQA_Train/sft_train.jsonl)')
    parser.add_argument('--output_dir', required=True,
                        help='Where to write retrospective chunk files')
    parser.add_argument('--filter_source', default='gemini-3-flash-preview',
                        help='Only process samples with this meta.source (empty = all)')
    parser.add_argument('--gemini_model', default='gemini-3-flash-preview',
                        help='Gemini model for hindsight generation')
    parser.add_argument('--total_chunks', type=int, default=1,
                        help='Total SLURM array size')
    parser.add_argument('--chunk_idx', type=int, default=0,
                        help='This job\'s 0-based chunk index')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Parallel Gemini threads per chunk')
    parser.add_argument('--num_samples', type=int, default=999999,
                        help='Cap on samples to process (useful for testing)')
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='Max output tokens per Gemini hindsight call')
    args = parser.parse_args()

    genai = configure_gemini()
    logger.info(f"Gemini model: {args.gemini_model}")

    filter_source = args.filter_source if args.filter_source else None
    all_samples = load_sft_file(args.input_file, filter_source=filter_source)
    all_samples = all_samples[:args.num_samples]

    # SLURM chunk split
    chunk_size = (len(all_samples) + args.total_chunks - 1) // args.total_chunks
    start = args.chunk_idx * chunk_size
    end = min(start + chunk_size, len(all_samples))
    chunk = all_samples[start:end]
    logger.info(
        f"Chunk {args.chunk_idx}/{args.total_chunks}: "
        f"samples {start}–{end-1} ({len(chunk)} samples)"
    )

    if not chunk:
        logger.info("Empty chunk, nothing to do.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"chunk_{args.chunk_idx}.jsonl")

    # Resume: collect already-processed scenario_ids from existing output file
    processed_ids = set()
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        sid = record.get('meta', {}).get('scenario_id')
                        if sid is not None:
                            processed_ids.add(sid)
                    except json.JSONDecodeError:
                        pass
        logger.info(f"Resuming: {len(processed_ids)} scenarios already processed, skipping them")

    todo = [s for s in chunk if s.get('meta', {}).get('scenario_id') not in processed_ids]
    logger.info(f"Remaining: {len(todo)} samples to process")

    if not todo:
        logger.info("All samples already processed.")
        return

    done = 0
    failed = 0
    skipped = 0
    write_lock = threading.Lock()

    def process_one(sample):
        return add_hindsight_thoughts(sample, genai, args.gemini_model, args.max_tokens)

    with open(out_path, 'a') as out_f:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_one, s): s for s in todo}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is None:
                        skipped += 1
                    else:
                        # Write immediately after each sample completes
                        with write_lock:
                            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                            out_f.flush()
                    done += 1
                    if done % 10 == 0 or done == len(todo):
                        logger.info(f"Progress: {done}/{len(todo)} done, {failed} failed, {skipped} skipped")
                except SystemExit:
                    logger.critical("Stopping due to Gemini 429 rate limit.")
                    raise
                except Exception as e:
                    failed += 1
                    logger.error(f"Sample failed: {e}")

    logger.info(f"Chunk {args.chunk_idx} finished. {done} processed, {failed} failed, {skipped} skipped → {out_path}")


if __name__ == '__main__':
    main()
