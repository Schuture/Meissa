"""
Extract direct-answer VQA training data from all_merged.jsonl.

Filters Type 5 (synthesis) samples and strips <think> reasoning,
producing clean question→answer pairs for direct VQA SFT training.

Usage:
    python extract_direct_vqa.py \
        --input training_data/all_merged.jsonl \
        --output training_data/all_direct_vqa.jsonl
"""

import argparse
import json
import re
import sys


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> and any trailing newline from gpt response."""
    cleaned = re.sub(r'<think>.*?</think>\n?', '', text, flags=re.DOTALL)
    return cleaned.strip()


def clean_basic_human_prompt(text: str) -> str:
    """Remove format instruction block from basic Type 5 human prompts.

    Removes patterns like:
      'Provide your response in the following format:\n<think>...'
      'Provide your analysis in the following format:\n<think>...'
    """
    # 找到 format instruction 的起始位置并截断
    patterns = [
        r'\n\nProvide your response in the following format:.*',
        r'\n\nProvide your analysis in the following format:.*',
    ]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.DOTALL)
    return text.strip()


def clean_intermediate_human_prompt(text: str) -> str:
    """Extract only the question portion from intermediate Type 5 human prompts.

    Intermediate prompts contain expert reports after:
      'Here are some reports from different medical domain experts.'
    We keep everything before that marker.
    """
    marker = "\n\nHere are some reports from different medical domain experts."
    idx = text.find(marker)
    if idx != -1:
        text = text[:idx]
    # 也去掉可能残留的 format instruction（不太可能，但以防万一）
    text = clean_basic_human_prompt(text)
    return text.strip()


def clean_system_prompt(text: str) -> str:
    """Remove 'First think step by step, then' from system prompts."""
    if not text:
        return text
    text = text.replace("First think step by step, then provide", "Provide")
    text = text.replace("First think step by step, then p", "P")
    return text.strip()


def process_sample(sample: dict) -> dict | None:
    """Process a single synthesis sample into direct VQA format.

    Returns None if the sample should be skipped.
    """
    meta = sample.get("meta", {})
    if meta.get("type") != "synthesis":
        return None

    convs = sample.get("conversations", [])
    if len(convs) < 2:
        return None

    human_msg = convs[0]
    gpt_msg = convs[1]

    if human_msg.get("from") != "human" or gpt_msg.get("from") != "gpt":
        return None

    # 根据 difficulty 清理 human prompt
    difficulty = meta.get("difficulty", "basic")
    if difficulty == "intermediate":
        cleaned_human = clean_intermediate_human_prompt(human_msg["value"])
    else:
        cleaned_human = clean_basic_human_prompt(human_msg["value"])

    # 去除 gpt 回复中的 <think> 推理
    cleaned_gpt = strip_think_tags(gpt_msg["value"])

    if not cleaned_gpt:
        return None

    # 构建输出 sample
    result = {
        "conversations": [
            {"from": "human", "value": cleaned_human},
            {"from": "gpt", "value": cleaned_gpt},
        ],
    }

    # 保留 images 字段（如果存在且非空）
    images = sample.get("images", [])
    if images:
        result["images"] = images

    # 更新 meta
    new_meta = dict(meta)
    new_meta["type"] = "direct_vqa"
    result["meta"] = new_meta

    # 清理 system prompt
    system = sample.get("system", "")
    if system:
        result["system"] = clean_system_prompt(system)

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract direct-answer VQA data from all_merged.jsonl")
    parser.add_argument("--input", type=str, default="training_data/all_merged.jsonl",
                        help="Input JSONL file (all_merged.jsonl)")
    parser.add_argument("--output", type=str, default="training_data/all_direct_vqa.jsonl",
                        help="Output JSONL file for direct VQA data")
    args = parser.parse_args()

    total = 0
    kept = 0
    basic_count = 0
    intermediate_count = 0
    skipped_empty = 0

    # 按 dataset 统计
    dataset_counts = {}

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            sample = json.loads(line)
            result = process_sample(sample)

            if result is None:
                continue

            if not result["conversations"][1]["value"]:
                skipped_empty += 1
                continue

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            kept += 1

            difficulty = sample["meta"].get("difficulty", "basic")
            if difficulty == "intermediate":
                intermediate_count += 1
            else:
                basic_count += 1

            # 根据 source_file 推断 dataset（长名优先避免子串误匹配）
            source = sample["meta"].get("source_file", "")
            for ds in ["mimic-cxr-vqa", "pubmedqa", "pathvqa", "medqa"]:
                if ds in source:
                    dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
                    break

    print(f"\n=== Direct VQA Extraction Summary ===")
    print(f"Total samples in input:     {total}")
    print(f"Type 5 (synthesis) kept:    {kept}")
    print(f"  - basic:                  {basic_count}")
    print(f"  - intermediate:           {intermediate_count}")
    print(f"Skipped (empty answer):     {skipped_empty}")
    print(f"\nPer-dataset breakdown:")
    for ds in ["medqa", "pubmedqa", "pathvqa", "mimic-cxr-vqa"]:
        print(f"  {ds:20s}: {dataset_counts.get(ds, 0)}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
