#!/usr/bin/env bash
# Demo: Framework IV — Multi-turn Clinical Simulation
#
# Runs Meissa on a sample MedQA OSCE scenario (no MIMIC access required).
#
# Required env vars:
#   MEISSA_MODEL: path or HuggingFace ID (default: CYX1998/Meissa-4B)
#
# Optional env vars:
#   GPU_IDS: GPU IDs (default: 0)

set -euo pipefail

MEISSA_MODEL="${MEISSA_MODEL:-CYX1998/Meissa-4B}"
GPU_IDS="${GPU_IDS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "=== Meissa Demo: Multi-turn Clinical Simulation ==="
echo "Model: ${MEISSA_MODEL}"
echo ""

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python \
    "${ROOT_DIR}/environments/clinical_simulation/medsim/main.py" \
    --model "${MEISSA_MODEL}" \
    --mode demo
