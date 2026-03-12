#!/usr/bin/env bash
# Evaluate Framework I (Continuous Tool Calling) on MIMIC-CXR-VQA
#
# REQUIRES: PhysioNet credentialed access to MIMIC-CXR-VQA and MIMIC-CXR-JPG.
# See docs/data.md for access instructions.
#
# Required env vars:
#   MEISSA_MODEL:       path or HuggingFace ID (default: CYX1998/Meissa-4B)
#   MIMIC_CXR_VQA_ROOT: path to MIMIC-CXR-VQA dataset root
#   MIMIC_CXR_JPG_ROOT: path to MIMIC-CXR-JPG dataset root
#
# Optional env vars:
#   CHEXAGENT_SERVER_URL: CheXagent tool server URL (default: http://127.0.0.1:19101)
#   MAIRA2_SERVER_URL:    Maira-2 tool server URL (default: http://127.0.0.1:19102)
#   OUTPUT_DIR:           where to save results (default: results/ctc_mimic_cxr_vqa/)
#   GPU_IDS:              comma-separated GPU IDs (default: 0)

set -euo pipefail

if [[ -z "${MIMIC_CXR_VQA_ROOT:-}" ]]; then
    echo "ERROR: MIMIC_CXR_VQA_ROOT is not set."
    echo "This benchmark requires PhysioNet credentialed access."
    echo "See docs/data.md for instructions."
    exit 1
fi

if [[ -z "${MIMIC_CXR_JPG_ROOT:-}" ]]; then
    echo "ERROR: MIMIC_CXR_JPG_ROOT is not set."
    exit 1
fi

MEISSA_MODEL="${MEISSA_MODEL:-CYX1998/Meissa-4B}"
CHEXAGENT_SERVER_URL="${CHEXAGENT_SERVER_URL:-http://127.0.0.1:19101}"
MAIRA2_SERVER_URL="${MAIRA2_SERVER_URL:-http://127.0.0.1:19102}"
OUTPUT_DIR="${OUTPUT_DIR:-results/ctc_mimic_cxr_vqa}"
GPU_IDS="${GPU_IDS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}"

echo "Running Framework I (Continuous Tool Calling) on MIMIC-CXR-VQA"
echo "Model:  ${MEISSA_MODEL}"
echo "Data:   ${MIMIC_CXR_VQA_ROOT}"
echo "Output: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
CHEXAGENT_SERVER_URL="${CHEXAGENT_SERVER_URL}" \
MAIRA2_SERVER_URL="${MAIRA2_SERVER_URL}" \
python \
    "${ROOT_DIR}/environments/continuous_tool_calling/eval/medrax_agent_bench.py" \
    --model "${MEISSA_MODEL}" \
    --data_root "${MIMIC_CXR_VQA_ROOT}" \
    --image_root "${MIMIC_CXR_JPG_ROOT}" \
    --output_dir "${ROOT_DIR}/${OUTPUT_DIR}"

echo "Results saved to ${OUTPUT_DIR}"
