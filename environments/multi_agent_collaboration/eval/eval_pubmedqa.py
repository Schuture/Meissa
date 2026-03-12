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


def clean_answer_pubmedqa(text):
    """
    Extract yes/no/maybe from PubMedQA response text.
    """
    if not isinstance(text, str):
        return "unknown"

    text_lower = text.lower()

    # 1. Explicit format patterns
    patterns = [
        r"answer:\s*(yes|no|maybe)",
        r"final answer:\s*(yes|no|maybe)",
        r"conclusion:\s*(yes|no|maybe)",
        r"\*\*(yes|no|maybe)\*\*"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            return matches[-1]

    # 2. Check last 100 chars for keyword presence
    last_part = text_lower[-100:]
    scores = {'yes': -1, 'no': -1, 'maybe': -1}
    for label in ['yes', 'no', 'maybe']:
        idx = last_part.rfind(label)
        if idx != -1:
            scores[label] = idx

    best_label = max(scores, key=scores.get)
    if scores[best_label] != -1:
        return best_label

    return "unknown"


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
        cprint(f"[Error] No JSON files found in {output_dir}", "red")
        return

    total = 0
    correct_regex = 0
    correct_utils = 0
    stats = {
        'basic': {'t': 0, 'c_regex': 0, 'c_utils': 0},
        'intermediate': {'t': 0, 'c_regex': 0, 'c_utils': 0},
        'advanced': {'t': 0, 'c_regex': 0, 'c_utils': 0},
        'unknown': {'t': 0, 'c_regex': 0, 'c_utils': 0}
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
        true_label = item.get('label')  # yes/no/maybe
        if not true_label:
            continue

        true_label = str(true_label).lower().strip()
        difficulty = item.get('difficulty', 'unknown')
        if difficulty not in stats:
            difficulty = 'unknown'

        # Method 1: Regex extraction
        resp_text = extract_response_text(item.get('response'), difficulty)
        pred_label = clean_answer_pubmedqa(resp_text)

        # Method 2: utils.py logic (containment-based)
        is_correct_u = check_correct(item, 'pubmedqa')

        total += 1
        stats[difficulty]['t'] += 1

        if pred_label == true_label:
            correct_regex += 1
            stats[difficulty]['c_regex'] += 1

        if is_correct_u:
            correct_utils += 1
            stats[difficulty]['c_utils'] += 1

    table = PrettyTable(['Category', 'Total', 'Correct (regex)', 'Acc-regex (%)',
                          'Correct (utils)', 'Acc-utils (%)'])
    acc_regex = (correct_regex / total * 100) if total else 0
    acc_utils = (correct_utils / total * 100) if total else 0
    table.add_row(['Overall', total, correct_regex, f"{acc_regex:.2f}",
                    correct_utils, f"{acc_utils:.2f}"])

    for diff in ['basic', 'intermediate', 'advanced']:
        s = stats[diff]
        if s['t'] > 0:
            d_acc_regex = (s['c_regex'] / s['t'] * 100)
            d_acc_utils = (s['c_utils'] / s['t'] * 100)
            table.add_row([diff, s['t'], s['c_regex'], f"{d_acc_regex:.2f}",
                           s['c_utils'], f"{d_acc_utils:.2f}"])

    print(f"\nPubMedQA Evaluation Results: {output_dir}")
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='output')
    args = parser.parse_args()
    evaluate(args.dir)
