#!/usr/bin/env bash
# Demo: Framework I — Continuous Tool Calling
#
# Runs Meissa on a sample chest X-ray question using the 8-tool radiology agent.
# No MIMIC access required for this demo (uses a bundled example image).
#
# Required env vars:
#   MEISSA_MODEL: path or HuggingFace ID (default: CYX1998/Meissa-4B)
#
# Optional env vars:
#   CHEXAGENT_SERVER_URL: CheXagent server URL (default: http://127.0.0.1:19101)
#   MAIRA2_SERVER_URL:    Maira-2 server URL (default: http://127.0.0.1:19102)
#   GPU_IDS:              GPU IDs (default: 0)

set -euo pipefail

MEISSA_MODEL="${MEISSA_MODEL:-CYX1998/Meissa-4B}"
CHEXAGENT_SERVER_URL="${CHEXAGENT_SERVER_URL:-http://127.0.0.1:19101}"
MAIRA2_SERVER_URL="${MAIRA2_SERVER_URL:-http://127.0.0.1:19102}"
GPU_IDS="${GPU_IDS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "=== Meissa Demo: Continuous Tool Calling ==="
echo "Model: ${MEISSA_MODEL}"
echo ""

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
CHEXAGENT_SERVER_URL="${CHEXAGENT_SERVER_URL}" \
MAIRA2_SERVER_URL="${MAIRA2_SERVER_URL}" \
python "${ROOT_DIR}/environments/continuous_tool_calling/main.py" \
    --model "${MEISSA_MODEL}" \
    --demo
