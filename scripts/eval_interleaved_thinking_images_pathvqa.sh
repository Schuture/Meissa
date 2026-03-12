#!/usr/bin/env bash
# Evaluate Framework II (Interleaved Thinking with Images) on PathVQA
#
# Required env vars:
#   MEISSA_MODEL: path or HuggingFace ID (default: CYX1998/Meissa-4B)
#
# Optional env vars:
#   PATHVQA_DATA: path to PathVQA test set (default: data/sft_samples/interleaved_thinking_images/pathvqa_test.jsonl)
#   OUTPUT_DIR:   where to save results (default: results/iti_pathvqa/)
#   GPU_IDS:      comma-separated GPU IDs (default: 0)

set -euo pipefail

MEISSA_MODEL="${MEISSA_MODEL:-CYX1998/Meissa-4B}"
PATHVQA_DATA="${PATHVQA_DATA:-environments/interleaved_thinking_images/eval/pathvqa_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/iti_pathvqa}"
GPU_IDS="${GPU_IDS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}"

echo "Running Framework II (Interleaved Thinking with Images) on PathVQA"
echo "Model:  ${MEISSA_MODEL}"
echo "Data:   ${PATHVQA_DATA}"
echo "Output: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python \
    "${ROOT_DIR}/environments/interleaved_thinking_images/eval/run_pathvqa.py" \
    --model "${MEISSA_MODEL}" \
    --data "${ROOT_DIR}/${PATHVQA_DATA}" \
    --output_dir "${ROOT_DIR}/${OUTPUT_DIR}" \
    --tool_server_url "http://localhost:8080"

echo "Results saved to ${OUTPUT_DIR}"
