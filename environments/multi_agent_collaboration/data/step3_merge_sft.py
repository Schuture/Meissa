#!/usr/bin/env python3
"""
Merge SFT files with three-tier sampling control:

  Tier 1 (Gemini intermediate): Keep ALL samples
  Tier 2 (Gemini basic):        Downsample Type 1/5 to gemini_basic_ratio * intermediate count
  Tier 3 (Qwen3 basic correct): Downsample total to qwen3_ratio * total Gemini kept

Usage:
    python merge_sft.py \
        --input-dir training_data/medqa/ \
        --gemini-prefix medqa_gemini \
        --qwen3-prefix medqa_qwen3 \
        --output training_data/medqa/medqa_merged.jsonl \
        --gemini-basic-ratio 2.0 \
        --qwen3-ratio 0.5
"""

import json
import argparse
import glob
import os
import random
from collections import defaultdict


# ── Type categories ──────────────────────────────────────────
# Types that contain both basic and intermediate samples (apply ratio)
MIXED_DIFFICULTY_TYPES = [
    'type1_difficulty', 'type5_synthesis',
    'type1r_difficulty_recap', 'type5r_synthesis_recap',
]

# Types that are intermediate only (always keep all)
INTERMEDIATE_ONLY_TYPES = ['type2_recruitment', 'type3_expert_analysis', 'type4_debate']

# Recap types that are intermediate only (always keep all)
RECAP_TYPES = ['type2r_recruitment_recap']

# Qwen3 only produces these two types
QWEN3_TYPES = ['type1_difficulty', 'type5_synthesis']

ALL_GEMINI_SUFFIXES = MIXED_DIFFICULTY_TYPES + INTERMEDIATE_ONLY_TYPES + RECAP_TYPES


def load_jsonl(filepath):
    """Load JSONL file, return list of dicts."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_if_exists(input_dir, prefix, suffix):
    """Load a JSONL file if it exists, return (samples, filepath_or_None)."""
    fp = os.path.join(input_dir, f"{prefix}_{suffix}.jsonl")
    if os.path.exists(fp):
        samples = load_jsonl(fp)
        return samples, fp
    return [], None


def get_difficulty(sample):
    """Extract difficulty from sample meta."""
    return sample.get('meta', {}).get('difficulty', 'unknown')


def split_by_difficulty(samples):
    """Split samples into basic and intermediate lists."""
    basic, intermediate = [], []
    for s in samples:
        d = get_difficulty(s)
        if d == 'intermediate':
            intermediate.append(s)
        else:
            basic.append(s)
    return basic, intermediate


def downsample(samples, target, rng):
    """Downsample list to target size. If already <= target, return as-is."""
    if len(samples) <= target:
        return samples
    return rng.sample(samples, target)


def main():
    parser = argparse.ArgumentParser(
        description='Merge SFT files with three-tier sampling control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Three-tier logic:
  1. Gemini intermediate samples: keep ALL
  2. Gemini basic samples (Type 1/5): keep gemini_basic_ratio * intermediate_count per type
  3. Qwen3 basic samples (Type 1/5): keep qwen3_ratio * intermediate_count per type

Example:
  python merge_sft.py \\
    --input-dir training_data/medqa/ \\
    --gemini-prefix medqa_gemini \\
    --qwen3-prefix medqa_qwen3 \\
    --output training_data/medqa/medqa_merged.jsonl \\
    --gemini-basic-ratio 2.0 \\
    --qwen3-ratio 0.5
        """,
    )
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing SFT JSONL files')
    parser.add_argument('--output', required=True,
                        help='Output merged JSONL file path')
    parser.add_argument('--gemini-prefix', required=True,
                        help='Prefix for Gemini files (e.g., medqa_gemini)')
    parser.add_argument('--qwen3-prefix', required=True,
                        help='Prefix for Qwen3 files (e.g., medqa_qwen3)')
    parser.add_argument('--gemini-basic-ratio', type=float, default=2.0,
                        help='For Type 1/5: Gemini basic count = ratio * intermediate count. '
                             '(default: 2.0)')
    parser.add_argument('--qwen3-ratio', type=float, default=0.5,
                        help='For Type 1/5: Qwen3 count = ratio * intermediate count per type. '
                             '(default: 0.5)')
    parser.add_argument('--type1-max-basic-ratio', type=float, default=2.0,
                        help='Max basic:intermediate ratio for Type 1 difficulty types. '
                             'Overrides --gemini-basic-ratio for type1_difficulty and '
                             'type1r_difficulty_recap to prevent class imbalance. '
                             '(default: 2.0, meaning at most 2x basic vs intermediate)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Do not shuffle the final output')

    args = parser.parse_args()
    rng = random.Random(args.seed)

    print("=" * 60)
    print("SFT Three-Tier Merge")
    print("=" * 60)
    print(f"  Input dir:           {args.input_dir}")
    print(f"  Gemini prefix:       {args.gemini_prefix}")
    print(f"  Qwen3 prefix:        {args.qwen3_prefix}")
    print(f"  Gemini basic ratio:  {args.gemini_basic_ratio}x intermediate")
    print(f"  Type1 max basic:     {args.type1_max_basic_ratio}x intermediate (cap for difficulty types)")
    print(f"  Qwen3 ratio:         {args.qwen3_ratio}x intermediate per type")
    print(f"  Seed:                {args.seed}")
    print()

    # ═══════════════════════════════════════════════════════════
    #  Step 1: Process Gemini files
    # ═══════════════════════════════════════════════════════════
    print("[Step 1] Processing Gemini files...")

    gemini_kept = []  # all Gemini samples that survive filtering
    report_rows = []  # for final report
    inter_count_by_type = {}  # track intermediate count per type for Qwen3

    # --- Type 1/5: split by difficulty, apply ratio ---
    for suffix in MIXED_DIFFICULTY_TYPES:
        samples, fp = load_if_exists(args.input_dir, args.gemini_prefix, suffix)
        if not samples:
            continue

        basic, intermediate = split_by_difficulty(samples)

        # Keep ALL intermediate, record count for Qwen3 step
        kept_inter = intermediate
        inter_count_by_type[suffix] = len(intermediate)

        # Downsample basic to ratio * intermediate count
        # For Type 1 difficulty types, apply stricter cap to prevent class imbalance
        if suffix in ('type1_difficulty', 'type1r_difficulty_recap'):
            effective_ratio = min(args.gemini_basic_ratio, args.type1_max_basic_ratio)
        else:
            effective_ratio = args.gemini_basic_ratio
        target_basic = int(len(intermediate) * effective_ratio)
        kept_basic = downsample(basic, target_basic, rng)

        kept = kept_basic + kept_inter
        gemini_kept.extend(kept)

        report_rows.append({
            'source': 'Gemini',
            'type': suffix,
            'orig': len(samples),
            'basic_orig': len(basic),
            'basic_kept': len(kept_basic),
            'inter_orig': len(intermediate),
            'inter_kept': len(kept_inter),
            'final': len(kept),
        })
        print(f"  {os.path.basename(fp)}: {len(samples)} → {len(kept)} "
              f"(basic {len(kept_basic)}/{len(basic)}, inter {len(kept_inter)}/{len(intermediate)})")

    # --- Type 2/3/4: intermediate only, keep all ---
    for suffix in INTERMEDIATE_ONLY_TYPES:
        samples, fp = load_if_exists(args.input_dir, args.gemini_prefix, suffix)
        if not samples:
            continue

        gemini_kept.extend(samples)
        report_rows.append({
            'source': 'Gemini',
            'type': suffix,
            'orig': len(samples),
            'final': len(samples),
        })
        print(f"  {os.path.basename(fp)}: {len(samples)} (all kept)")

    # --- Recap types: keep all ---
    for suffix in RECAP_TYPES:
        samples, fp = load_if_exists(args.input_dir, args.gemini_prefix, suffix)
        if not samples:
            continue

        gemini_kept.extend(samples)
        report_rows.append({
            'source': 'Gemini',
            'type': suffix,
            'orig': len(samples),
            'final': len(samples),
        })
        print(f"  {os.path.basename(fp)}: {len(samples)} (all kept)")

    total_gemini = len(gemini_kept)
    print(f"\n  Total Gemini kept: {total_gemini}")

    # ═══════════════════════════════════════════════════════════
    #  Step 2: Process Qwen3 files
    # ═══════════════════════════════════════════════════════════
    print(f"\n[Step 2] Processing Qwen3 files...")

    # Per-type: Qwen3 kept = qwen3_ratio * intermediate count for that type
    qwen3_kept = []
    for suffix in QWEN3_TYPES:
        samples, fp = load_if_exists(args.input_dir, args.qwen3_prefix, suffix)
        if not samples:
            continue

        inter_count = inter_count_by_type.get(suffix, 0)
        target = int(inter_count * args.qwen3_ratio)
        kept = downsample(samples, target, rng)
        qwen3_kept.extend(kept)

        report_rows.append({
            'source': 'Qwen3',
            'type': suffix,
            'orig': len(samples),
            'final': len(kept),
        })
        print(f"  {os.path.basename(fp)}: {len(samples)} → {len(kept)} "
              f"({args.qwen3_ratio} × {inter_count} inter = {target})")

    print(f"\n  Total Qwen3 kept: {len(qwen3_kept)}")

    # ═══════════════════════════════════════════════════════════
    #  Step 3: Merge and write
    # ═══════════════════════════════════════════════════════════
    all_merged = gemini_kept + qwen3_kept

    if not args.no_shuffle:
        rng.shuffle(all_merged)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for s in all_merged:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    # ═══════════════════════════════════════════════════════════
    #  Report
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Merge Report")
    print("=" * 60)

    header = f"  {'Source':<8s} {'Type':<30s} {'Original':>8s} {'Basic':>14s} {'Inter':>14s} {'Final':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for row in report_rows:
        src = row['source']
        typ = row['type']
        orig = row['orig']
        final = row['final']

        if 'basic_orig' in row:
            b_str = f"{row['basic_kept']}/{row['basic_orig']}"
            i_str = f"{row['inter_kept']}/{row['inter_orig']}"
        else:
            b_str = "—"
            i_str = "—"

        print(f"  {src:<8s} {typ:<30s} {orig:>8d} {b_str:>14s} {i_str:>14s} {final:>8d}")

    print("  " + "-" * (len(header) - 2))
    print(f"  {'':8s} {'Gemini subtotal':<30s} {'':>8s} {'':>14s} {'':>14s} {total_gemini:>8d}")
    print(f"  {'':8s} {'Qwen3 subtotal':<30s} {'':>8s} {'':>14s} {'':>14s} {len(qwen3_kept):>8d}")
    print(f"  {'':8s} {'TOTAL':<30s} {'':>8s} {'':>14s} {'':>14s} {len(all_merged):>8d}")

    print(f"\nOutput: {args.output} ({len(all_merged)} samples)")
    print("[DONE]")


if __name__ == '__main__':
    main()
