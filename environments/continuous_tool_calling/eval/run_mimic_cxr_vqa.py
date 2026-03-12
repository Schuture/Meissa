"""
eval/run_mimic_cxr_vqa.py — Evaluate Meissa (Framework I: Continuous Tool Calling)
on the MIMIC-CXR-VQA benchmark.

MIMIC data requires PhysioNet credentialing. See eval/README.md for setup instructions.

Usage:
    python run_mimic_cxr_vqa.py \
        --model Qwen/Qwen3-VL-4B-Instruct \
        --data_path /path/to/mimic_cxr_vqa_test.json \
        --image_dir /path/to/mimic-cxr-jpg/files \
        --output results/ctc_mimic_cxr_vqa.json

Environment variables:
    MEISSA_MODEL   — model path / HuggingFace ID (overrides --model)
"""

import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.agent import Agent


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_correct(prediction: str, label) -> bool:
    if isinstance(label, list):
        return any(normalize(l) in normalize(prediction) for l in label)
    return normalize(str(label)) in normalize(prediction)


def load_dataset(path: str, limit: int | None = None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def run_eval(args):
    model_id = os.environ.get("MEISSA_MODEL", args.model)
    logger.info(f"Model: {model_id}")
    logger.info(f"Dataset: {args.data_path}  |  Image dir: {args.image_dir}")

    agent = Agent(model=model_id)
    dataset = load_dataset(args.data_path, args.limit)
    logger.info(f"Loaded {len(dataset)} samples")

    results = []
    correct = 0

    for i, sample in enumerate(dataset):
        question = sample.get("question", "")
        label = sample.get("answer", sample.get("label", ""))
        image_path = os.path.join(args.image_dir, sample.get("image_path", ""))

        try:
            response = agent.run(question=question, image_path=image_path)
        except Exception as e:
            logger.warning(f"Sample {i} failed: {e}")
            response = ""

        is_correct = check_correct(response, label)
        correct += int(is_correct)

        results.append({
            "id": sample.get("id", i),
            "question": question,
            "label": label,
            "response": response,
            "correct": is_correct,
        })

        if (i + 1) % 50 == 0:
            logger.info(f"[{i+1}/{len(dataset)}] Running acc: {correct/(i+1):.4f}")

    accuracy = correct / len(dataset) if dataset else 0.0
    logger.info(f"Final accuracy: {accuracy:.4f} ({correct}/{len(dataset)})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"accuracy": accuracy, "n_correct": correct,
                       "n_total": len(dataset), "results": results}, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate Meissa on MIMIC-CXR-VQA")
    parser.add_argument("--model", default=os.environ.get("MEISSA_MODEL", "CYX1998/Meissa-4B"),
                        help="Model path or HuggingFace ID")
    parser.add_argument("--data_path", required=True,
                        help="Path to MIMIC-CXR-VQA test JSON file")
    parser.add_argument("--image_dir", required=True,
                        help="Root directory of MIMIC-CXR-JPG images")
    parser.add_argument("--output", default=None,
                        help="Path to save JSON results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
