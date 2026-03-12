#!/usr/bin/env bash
# Evaluate Framework III (Multi-Agent Collaboration) on PubMedQA
#
# Required env vars:
#   MEISSA_MODEL: path or HuggingFace ID (default: CYX1998/Meissa-4B)
#
# Optional env vars:
#   PUBMEDQA_DATA: path to PubMedQA test set (default: data/sft_samples/multi_agent_collaboration/pubmedqa_test.jsonl)
#   OUTPUT_DIR:    where to save results (default: results/mac_pubmedqa/)
#   GPU_IDS:       comma-separated GPU IDs (default: 0)

set -euo pipefail

MEISSA_MODEL="${MEISSA_MODEL:-CYX1998/Meissa-4B}"
PUBMEDQA_DATA="${PUBMEDQA_DATA:-environments/multi_agent_collaboration/eval/pubmedqa_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/mac_pubmedqa}"
GPU_IDS="${GPU_IDS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}"

echo "Running Framework III (Multi-Agent Collaboration) on PubMedQA"
echo "Model:  ${MEISSA_MODEL}"
echo "Data:   ${PUBMEDQA_DATA}"
echo "Output: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python \
    "${ROOT_DIR}/environments/multi_agent_collaboration/eval/eval_pubmedqa.py" \
    --model "${MEISSA_MODEL}" \
    --data "${ROOT_DIR}/${PUBMEDQA_DATA}" \
    --output_dir "${ROOT_DIR}/${OUTPUT_DIR}"

echo "Results saved to ${OUTPUT_DIR}"
