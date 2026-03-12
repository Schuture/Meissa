#!/usr/bin/env bash
# Demo: Framework III — Multi-Agent Collaboration
#
# Runs Meissa on a sample PubMedQA question with adaptive multi-agent routing.
# No external data access required.
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

echo "=== Meissa Demo: Multi-Agent Collaboration ==="
echo "Model: ${MEISSA_MODEL}"
echo ""

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python \
    "${ROOT_DIR}/environments/multi_agent_collaboration/main.py" \
    --model "${MEISSA_MODEL}" \
    --demo
