#!/usr/bin/env bash
# Evaluate Framework IV (Multi-turn Clinical Simulation) on MIMIC-IV
#
# REQUIRES: PhysioNet credentialed access to MIMIC-IV and MIMIC-CXR-JPG.
# See docs/data.md for access instructions.
#
# Required env vars:
#   MEISSA_MODEL:       path or HuggingFace ID (default: CYX1998/Meissa-4B)
#   MIMIC_IV_ROOT:      path to MIMIC-IV dataset root
#   MIMIC_CXR_JPG_ROOT: path to MIMIC-CXR-JPG dataset root (for multimodal cases)
#
# Optional env vars:
#   OUTPUT_DIR: where to save results (default: results/mcs_mimic_iv/)
#   GPU_IDS:    comma-separated GPU IDs (default: 0)

set -euo pipefail

if [[ -z "${MIMIC_IV_ROOT:-}" ]]; then
    echo "ERROR: MIMIC_IV_ROOT is not set."
    echo "This benchmark requires PhysioNet credentialed access."
    echo "See docs/data.md for instructions."
    exit 1
fi

MEISSA_MODEL="${MEISSA_MODEL:-CYX1998/Meissa-4B}"
MIMIC_CXR_JPG_ROOT="${MIMIC_CXR_JPG_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/mcs_mimic_iv}"
GPU_IDS="${GPU_IDS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}"

echo "Running Framework IV (Multi-turn Clinical Simulation) on MIMIC-IV"
echo "Model:  ${MEISSA_MODEL}"
echo "Data:   ${MIMIC_IV_ROOT}"
echo "Output: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python \
    "${ROOT_DIR}/environments/clinical_simulation/medsim/main.py" \
    --model "${MEISSA_MODEL}" \
    --mimic_iv_root "${MIMIC_IV_ROOT}" \
    --mimic_cxr_root "${MIMIC_CXR_JPG_ROOT}" \
    --output_dir "${ROOT_DIR}/${OUTPUT_DIR}" \
    --mode eval

echo "Results saved to ${OUTPUT_DIR}"
