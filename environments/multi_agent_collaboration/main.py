import os
import sys
# agents/utils.py contains all framework logic; add it to path before importing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents'))

import json
import random
import argparse
from tqdm import tqdm
from termcolor import cprint
from pptree import print_tree
from prettytable import PrettyTable
from utils import (
    Agent, Group, parse_hierarchy, parse_group_info, setup_model,
    load_data, create_question, determine_difficulty,
    process_basic_query, process_intermediate_query, process_advanced_query,
    reset_trace, get_trace, filter_wrong_samples,
    build_exampler_cache,
    _check_correct, generate_hindsight_recap,
    APIResourceExhausted
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='medqa')
parser.add_argument('--output-dir', type=str, default='outputs')
parser.add_argument('--model', type=str, default='Qwen3-VL-4B-Instruct')
parser.add_argument('--difficulty', type=str, default='adaptive')
parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                    help='Which data split to run inference on')
parser.add_argument('--max_difficulty', type=str, default='advanced',
                    choices=['basic', 'intermediate', 'advanced'],
                    help='Cap adaptive difficulty (e.g., intermediate = no advanced)')
parser.add_argument('--filter_wrong_from', type=str, default=None,
                    help='Directory containing previous result JSONs; only re-run wrong samples')
parser.add_argument('--num_samples', type=int, default=999999)
parser.add_argument('--total_chunks', type=int, default=1, help='Total number of chunks to split the dataset into')
parser.add_argument('--chunk_idx', type=int, default=0, help='The index of the current chunk (0 to total_chunks-1)')
parser.add_argument('--use-think-format', action='store_true',
                    help='Use <think>...</think> format for SFT-trained models (default: Thought:/Answer: for Gemini)')
parser.add_argument('--exampler-cache-path', type=str, default=None,
                    help='Path to a pre-generated exemplar cache JSON file (e.g., from base model). '
                         'Avoids degraded exemplars from fine-tuned models.')
parser.add_argument('--demo', action='store_true',
                    help='Run a single demo question (no dataset required)')

args = parser.parse_args()

random.seed(1)

if args.demo:
    # PubMedQA sample 23848044 — correct answer: yes
    # Source: PubMedQA (open access). Clear medical question about drug safety in children.
    _demo_question = (
        "Context: This study represents a subset of a complete data set, considering only those "
        "children aged admitted to the Pediatric Surgery and Pediatric Nephrology Clinics during "
        "the period January 2011 to July 2012. In this study, we have determined that the QT "
        "interval changes significantly depending on the use of oxybutynin. The QT changes "
        "increased cardiac arrhythmia in children.\n\n"
        "Question: Does oxybutynin hydrochloride cause arrhythmia in children with bladder "
        "dysfunction?"
    )
    print("\n=== Meissa Demo: Multi-Agent Collaboration (Framework III) ===")
    print(f"Model: {args.model}")
    print(f"\nQuestion:\n{_demo_question}\n")

    class _DemoArgs:
        dataset = "pubmedqa"
        use_think_format = args.use_think_format
        max_difficulty = "advanced"

    # Force advanced for demo — this shows the full MDT (Multidisciplinary Team)
    # collaboration pipeline with multiple groups and internal discussions.
    # (Intermediate depends on agents saying "yes" to participate, which is unreliable.)
    difficulty = "advanced"
    print(f"[Routing] \u2192 Difficulty: {difficulty.upper()} (forced for demo)")
    print()

    if difficulty == "basic":
        answer = process_basic_query(
            _demo_question, [], args.model, _DemoArgs(), cached_examplers=[]
        )
    elif difficulty == "intermediate":
        answer = process_intermediate_query(
            _demo_question, [], args.model, _DemoArgs(), cached_examplers=[]
        )
    else:
        answer = process_advanced_query(_demo_question, args.model, _DemoArgs())

    # process_*_query returns nested dict: {'majority': {0.0: content}}
    # Flatten to extract the actual text answer.
    while isinstance(answer, dict):
        answer = next(iter(answer.values()), str(answer))

    # Strip <think> tags for clean display
    import re as _re
    _answer_clean = _re.sub(r'<think>.*?</think>', '', str(answer), flags=_re.DOTALL).strip()
    print(f"\nAnswer: {_answer_clean}")
    print("\nExpected: yes")
    sys.exit(0)

model, client = setup_model(args.model)
test_qa, examplers = load_data(args.dataset, split=args.split)

if args.filter_wrong_from:
    test_qa = filter_wrong_samples(test_qa, args.filter_wrong_from, args.dataset)

random.shuffle(test_qa)
print(f"[INFO] Shuffled {len(test_qa)} samples with seed=1")

total_data_len = len(test_qa)
if args.total_chunks > 1:
    chunk_size = total_data_len // args.total_chunks
    start_idx = args.chunk_idx * chunk_size
    if args.chunk_idx == args.total_chunks - 1:
        end_idx = total_data_len
    else:
        end_idx = start_idx + chunk_size
    
    test_qa_chunk = test_qa[start_idx:end_idx]
    print(f"[INFO] Running chunk {args.chunk_idx + 1}/{args.total_chunks}")
    print(f"[INFO] Original Data slice: {start_idx} to {end_idx} (Total: {len(test_qa_chunk)})")
else:
    test_qa_chunk = test_qa
    start_idx = 0
    print(f"[INFO] Running full dataset ({len(test_qa_chunk)} samples)")

if args.num_samples < len(test_qa_chunk):
    print(f"[INFO] limiting execution to first {args.num_samples} samples per chunk as requested.")
    test_qa_chunk = test_qa_chunk[:args.num_samples]

agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
random.shuffle(agent_emoji)

path = os.path.join(os.getcwd(), args.output_dir)
os.makedirs(path, exist_ok=True)

# Pre-generate and cache exemplar reasonings (PubMedQA / MedQA only)
# First run: generates reasoning via API and saves to disk
# Subsequent runs: loads from cache, no API calls needed
if args.exampler_cache_path:
    # Load pre-generated exemplar cache directly (e.g., from base model)
    import json as _json
    with open(args.exampler_cache_path, 'r', encoding='utf-8') as _f:
        cached_examplers = _json.load(_f)[:3]
    print(f"[INFO] Loaded {len(cached_examplers)} exemplars from external cache: {args.exampler_cache_path}")
else:
    cached_examplers = build_exampler_cache(examplers, args.model, args.dataset, path, num_examplers=3,
                                               use_think_format=args.use_think_format)

output_filename = f'{args.output_dir}/{args.model}_{args.dataset}_{args.split}_{args.difficulty}_chunk_{args.chunk_idx}.json'
trace_filename = f'{args.output_dir}/{args.model}_{args.dataset}_{args.split}_{args.difficulty}_chunk_{args.chunk_idx}_traces.jsonl'

results = []
if os.path.exists(output_filename):
    try:
        with open(output_filename, 'r') as f:
            results = json.load(f)
        print(f"[INFO] Loaded {len(results)} existing results from {output_filename}, skipping them.")
    except json.JSONDecodeError:
        print("[WARN] Output file exists but is corrupted, starting fresh.")

processed_count = len(results)

try:
    for i, sample in enumerate(tqdm(test_qa_chunk)):

        if processed_count >= len(test_qa_chunk):
            print("[INFO] All samples in this chunk (or limit) have been processed.")
            break

        if i < processed_count:
            continue

        global_idx = start_idx + i
        print(f"\n[INFO] Global Index: {global_idx} | Chunk Index: {i}")

        # Reset trace collector for this sample
        reset_trace()

        question, img_path = create_question(sample, args.dataset)

        try:
            difficulty = determine_difficulty(question, args.difficulty, args.model, img_path=img_path)

            difficulty_order = ['basic', 'intermediate', 'advanced']
            max_idx = difficulty_order.index(args.max_difficulty)
            cur_idx = difficulty_order.index(difficulty)
            if cur_idx > max_idx:
                print(f"[INFO] Capping difficulty: {difficulty} -> {args.max_difficulty}")
                difficulty = args.max_difficulty

            print(f"difficulty: {difficulty}")

            if difficulty == 'basic':
                final_decision = process_basic_query(question, examplers, args.model, args, img_path=img_path, cached_examplers=cached_examplers)
            elif difficulty == 'intermediate':
                final_decision = process_intermediate_query(question, examplers, args.model, args, img_path=img_path, cached_examplers=cached_examplers)
            elif difficulty == 'advanced':
                final_decision = process_advanced_query(question, args.model, args, img_path=img_path)

            result_entry = {
                'id': sample.get('id', global_idx),
                'question': question,
                'response': final_decision,
                'difficulty': difficulty
            }

            if args.dataset == 'medqa':
                result_entry['label'] = sample['answer_idx']
                result_entry['answer'] = sample['answer']
                result_entry['options'] = sample['options']
            elif args.dataset == 'pubmedqa':
                result_entry['label'] = sample['final_decision'] # yes/no/maybe
                result_entry['long_answer'] = sample.get('LONG_ANSWER', '')
            elif args.dataset in ['pathvqa', 'mimic-cxr-vqa']:
                result_entry['label'] = sample['answer']

            results.append(result_entry)

            with open(output_filename, 'w') as file:
                json.dump(results, file, indent=4)

            if args.split != 'test':
                is_correct = _check_correct(result_entry, args.dataset)
                if is_correct:
                    generate_hindsight_recap(
                        question=question, difficulty=difficulty,
                        result_entry=result_entry, model=args.model,
                        dataset=args.dataset, img_path=img_path
                    )

            trace_entry = {
                'id': sample.get('id', global_idx),
                'question': question,
                'difficulty': difficulty,
                'agents': get_trace()
            }
            if img_path:
                trace_entry['img_path'] = img_path
            with open(trace_filename, 'a') as f:
                f.write(json.dumps(trace_entry, ensure_ascii=False) + '\n')

        except Exception as e:
            print(f"[ERROR] Failed at Global Index {global_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

except (SystemExit, APIResourceExhausted) as e:
    print(f"\n{'='*60}")
    print(f"[FATAL] Program stopped due to consecutive API failures:")
    print(f"  {e}")
    print(f"{'='*60}")
    print(f"[INFO] Processed {len(results)} samples before stopping.")
    print(f"[INFO] Results saved to: {output_filename}")
    print(f"[INFO] Traces saved to: {trace_filename}")
    print(f"\nPlease check:")
    print(f"  1. API keys are valid and not expired")
    print(f"  2. API quotas/credits are not exhausted")
    print(f"  3. API service is operational")
    import sys
    sys.exit(1)

print(f"[INFO] Chunk {args.chunk_idx} finished.")