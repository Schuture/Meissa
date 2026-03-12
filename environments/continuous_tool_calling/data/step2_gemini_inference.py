import json
import random
from collections import defaultdict
import argparse

def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def is_easy_sample(x):
    if x.get("status") != "ok":
        return False
    if not x.get("is_correct", False):
        return False
    if x.get("tool_failures", {}).get("count", 0) > 0:
        return False
    if x.get("num_tools", 0) > 1:
        return False
    if not x.get("assistant_final"):
        return False
    return True


def select_easy_samples(
    qwen_data,
    max_total,
    per_template,
    seed,
):
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for x in qwen_data:
        if is_easy_sample(x):
            tmpl = x.get("template_program", "unknown")
            buckets[tmpl].append(x)

    selected = []
    for tmpl, items in buckets.items():
        rng.shuffle(items)
        take = items[:per_template]
        selected.extend(take)

    rng.shuffle(selected)
    return selected[:max_total]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini-jsonl", required=True)
    parser.add_argument("--qwen-jsonl", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--max-easy", type=int, default=800)
    parser.add_argument("--per-template", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    gemini = read_jsonl(args.gemini_jsonl)
    qwen = read_jsonl(args.qwen_jsonl)

    easy = select_easy_samples(
        qwen_data=qwen,
        max_total=args.max_easy,
        per_template=args.per_template,
        seed=args.seed,
    )

    merged = []
    for x in gemini:
        y = dict(x)
        y["source"] = "gemini_hard"
        merged.append(y)

    for x in easy:
        y = dict(x)
        y["source"] = "qwen_easy"
        merged.append(y)

    print("================================================")
    print(f"Gemini hard samples: {len(gemini)}")
    print(f"Qwen easy samples:   {len(easy)}")
    print(f"Total merged:        {len(merged)}")
    print("================================================")

    write_jsonl(args.output, merged)


if __name__ == "__main__":
    main()

