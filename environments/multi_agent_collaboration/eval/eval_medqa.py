import os
import json
import re
import glob
import argparse
from termcolor import cprint
from prettytable import PrettyTable
from eval_helpers import (
    check_correct, extract_response_text, _extract_answer_from_response
)


def clean_answer(text):
    """
    Extract option letter A/B/C/D/E from model response text.
    """
    if not isinstance(text, str):
        return None

    patterns = [
        r"Answer:\s*\(?([A-E])\)?",
        r"The correct answer is\s*\(?([A-E])\)?",
        r"Final Answer:\s*\(?([A-E])\)?",
        r"Correct Answer:\s*\(?([A-E])\)?",
        r"\*\*Answer:\s*\(?([A-E])\)?\*\*",
        r"\(([A-E])\)",
        r"(?:^|\n)\s*([A-E])\)\s",
        r"^([A-E])$"
    ]

    found_answers = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            found_answers.extend(matches)

    if found_answers:
        return found_answers[-1].upper()

    return None


def is_result_file(filepath):
    """Check if a JSON file is a result file (not cache/trace)."""
    basename = os.path.basename(filepath)
    if '_traces' in basename:
        return False
    if 'exampler_cache' in basename:
        return False
    if basename.startswith('.'):
        return False
    return True


def evaluate(output_dir):
    json_files = sorted([
        f for f in glob.glob(os.path.join(output_dir, "*.json"))
        if is_result_file(f)
    ])
    if not json_files:
        cprint(f"[Error] No result JSON files found in {output_dir}", "red")
        return

    total_samples = 0
    correct_count = 0
    correct_count_utils = 0  # using utils.py logic
    extraction_errors = 0

    stats = {
        'basic': {'total': 0, 'correct': 0, 'correct_utils': 0},
        'intermediate': {'total': 0, 'correct': 0, 'correct_utils': 0},
        'advanced': {'total': 0, 'correct': 0, 'correct_utils': 0},
        'unknown': {'total': 0, 'correct': 0, 'correct_utils': 0}
    }

    all_data = []

    print(f"Loading files from: {output_dir}")
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                    print(f"  - Loaded {len(data)} samples from {os.path.basename(jf)}")
        except Exception as e:
            print(f"  [Warn] Failed to load {jf}: {e}")

    print("-" * 60)

    for item in all_data:
        true_label = item.get('label')
        if not true_label:
            continue

        true_label = true_label.upper().strip()
        difficulty = item.get('difficulty', 'unknown')
        if difficulty not in stats:
            difficulty = 'unknown'

        # Method 1: Regex extraction
        raw_response = item.get('response', {})
        response_text = extract_response_text(raw_response, difficulty)
        pred_label = clean_answer(response_text)

        # Method 2: utils.py _check_correct logic (containment-based)
        is_correct_utils = check_correct(item, 'medqa')

        total_samples += 1
        stats[difficulty]['total'] += 1

        if pred_label:
            if pred_label == true_label:
                correct_count += 1
                stats[difficulty]['correct'] += 1
        else:
            extraction_errors += 1
            if extraction_errors <= 5:
                print(f"[Extraction Fail] ID: {item.get('id')} | Diff: {difficulty}")
                print(f"  Raw Text Tail: {response_text[-100:] if response_text else 'None'}")

        if is_correct_utils:
            correct_count_utils += 1
            stats[difficulty]['correct_utils'] += 1

    # --- Output table ---
    table = PrettyTable()
    table.field_names = ["Category", "Total", "Correct (regex)", "Acc-regex (%)",
                         "Correct (utils)", "Acc-utils (%)"]

    acc_regex = (correct_count / total_samples * 100) if total_samples > 0 else 0
    acc_utils = (correct_count_utils / total_samples * 100) if total_samples > 0 else 0
    table.add_row(["Overall", total_samples, correct_count, f"{acc_regex:.2f}",
                    correct_count_utils, f"{acc_utils:.2f}"])

    for diff in ['basic', 'intermediate', 'advanced']:
        s = stats[diff]
        if s['total'] > 0:
            d_acc_regex = (s['correct'] / s['total'] * 100)
            d_acc_utils = (s['correct_utils'] / s['total'] * 100)
            table.add_row([diff.capitalize(), s['total'], s['correct'], f"{d_acc_regex:.2f}",
                           s['correct_utils'], f"{d_acc_utils:.2f}"])

    print(f"\nMedQA Evaluation Results: {output_dir}")
    print(table)

    if extraction_errors > 0:
        cprint(f"\n[Warning] Failed to extract answers from {extraction_errors} samples "
               f"({extraction_errors/total_samples*100:.1f}%).", "yellow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='output',
                        help='Directory containing result JSON files')
    args = parser.parse_args()

    evaluate(args.dir)
