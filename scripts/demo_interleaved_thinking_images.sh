#!/usr/bin/env bash
# Demo: Framework II — Interleaved Thinking with Images
#
# Runs Meissa on a sample pathology image with multi-round visual tool calls.
# Uses a bundled PathVQA sample image.
#
# Required env vars:
#   MEISSA_MODEL: path or HuggingFace ID (default: CYX1998/Meissa-4B)
#
# Optional env vars:
#   TOOL_SERVER_URL: visual tool server URL (default: http://localhost:8080)
#   GPU_IDS:         GPU IDs (default: 0)

set -euo pipefail

MEISSA_MODEL="${MEISSA_MODEL:-CYX1998/Meissa-4B}"
TOOL_SERVER_URL="${TOOL_SERVER_URL:-http://localhost:8080}"
GPU_IDS="${GPU_IDS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "=== Meissa Demo: Interleaved Thinking with Images ==="
echo "Model:       ${MEISSA_MODEL}"
echo "Tool server: ${TOOL_SERVER_URL}"
echo ""
echo "Note: Start the tool server first with:"
echo "  python environments/interleaved_thinking_images/tool_server/tf_eval/run_server.py --port 8080"
echo ""

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python \
    "${ROOT_DIR}/environments/interleaved_thinking_images/tool_server/tf_eval/run_inference.py" \
    --model "${MEISSA_MODEL}" \
    --tool_server_url "${TOOL_SERVER_URL}" \
    --demo
