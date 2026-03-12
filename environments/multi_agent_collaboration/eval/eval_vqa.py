import os
import json
import re
import glob
import argparse
from termcolor import cprint
from prettytable import PrettyTable
from eval_helpers import (
    check_correct, extract_response_text, _extract_answer_from_response,
    _check_correct_pathvqa, _check_correct_mimic
)


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


def evaluate(output_dir, dataset_filter="pathvqa"):
    search_pattern = os.path.join(output_dir, "*.json")
    json_files = sorted([
        f for f in glob.glob(search_pattern)
        if is_result_file(f)
    ])

    if not json_files:
        cprint(f"[Error] No JSON files found in {output_dir}", "red")
        return

    print(f"[INFO] Found {len(json_files)} result files to evaluate.")

    # Determine dataset type for check_correct
    if 'mimic' in dataset_filter.lower():
        dataset_name = 'mimic-cxr-vqa'
    else:
        dataset_name = 'pathvqa'

    def _new_bucket():
        return {'t': 0, 'c': 0, 'yn_t': 0, 'yn_c': 0, 'open_t': 0, 'open_c': 0}

    overall = _new_bucket()
    per_diff = {d: _new_bucket() for d in ['basic', 'intermediate', 'advanced', 'unknown']}

    all_data = []
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                    print(f"  - Loaded {len(data)} samples from {os.path.basename(jf)}")
                else:
                    print(f"[WARN] File {jf} is not a list, skipping.")
        except Exception as e:
            print(f"[WARN] Error reading {jf}: {e}")

    print("-" * 60)

    for item in all_data:
        raw_label = item.get('label')
        if raw_label is None or (isinstance(raw_label, str) and not raw_label.strip()):
            continue

        difficulty = item.get('difficulty', 'unknown')
        if difficulty not in per_diff:
            difficulty = 'unknown'

        is_correct = check_correct(item, dataset_name)

        label_str = str(raw_label).lower().strip()
        is_yn = label_str in ('yes', 'no')

        for bucket in [overall, per_diff[difficulty]]:
            bucket['t'] += 1
            if is_yn:
                bucket['yn_t'] += 1
            else:
                bucket['open_t'] += 1
            if is_correct:
                bucket['c'] += 1
                if is_yn:
                    bucket['yn_c'] += 1
                else:
                    bucket['open_c'] += 1

    # --- Output table ---
    def _pct(n, d):
        return f"{n/d*100:.2f}" if d else "—"

    table = PrettyTable(['Category', 'Total', 'Overall (%)',
                          'YN Total', 'YN (%)', 'Open Total', 'Open (%)'])
    table.align = 'r'
    table.align['Category'] = 'l'

    def _add_row(label, b):
        table.add_row([label, b['t'], _pct(b['c'], b['t']),
                        b['yn_t'], _pct(b['yn_c'], b['yn_t']),
                        b['open_t'], _pct(b['open_c'], b['open_t'])])

    _add_row('Overall', overall)
    for diff in ['basic', 'intermediate', 'advanced']:
        if per_diff[diff]['t'] > 0:
            _add_row(diff.capitalize(), per_diff[diff])

    ds_label = dataset_name.upper().replace('-', '_')
    print(f"\n{ds_label} Evaluation Results: {output_dir}")
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True,
                        help="Path to the output directory containing JSON files")
    parser.add_argument('--dataset', type=str, default='pathvqa',
                        help="Dataset type: pathvqa or mimic-cxr-vqa")
    args = parser.parse_args()

    evaluate(args.dir, args.dataset)
