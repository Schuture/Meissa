#!/usr/bin/env bash
# test_demo_all.sh — Start vLLM, then run demos for all 4 Meissa frameworks.
#
# Recommended: submit as Slurm job:
#   cd /path/to/Meissa
#   sbatch scripts/test_demo_all.sh
#
#SBATCH --job-name=meissa-demo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=logs/demo_%j.log

set -euo pipefail
mkdir -p logs

# ─── Paths & model ────────────────────────────────────────────────────────
# IMPORTANT: Set ROOT_DIR to the absolute path of the Meissa project.
# SLURM copies scripts to a temp dir, so BASH_SOURCE-based resolution won't work.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MEISSA_MODEL:-CYX1998/Meissa-4B}"
MODEL_DIR="${MEISSA_MODEL_DIR:-/model-weights}"
VLLM_PORT=8877

# Conda environments — adjust these paths to your local installation
ENV_VLLM="${ENV_VLLM:-python}"
ENV_MEDRAX="${ENV_MEDRAX:-python}"
ENV_TOOLSERVER="${ENV_TOOLSERVER:-python}"
ENV_MDAGENTS="${ENV_MDAGENTS:-python}"
ENV_MEDAGENTSIM="${ENV_MEDAGENTSIM:-python}"

echo "=============================="
echo "Meissa Demo Test — All 4 Frameworks"
echo "Model: ${MODEL}"
echo "vLLM port: ${VLLM_PORT}"
echo "=============================="
echo ""

# ─── 1. Start vLLM server ─────────────────────────────────────────────────
echo "[1/5] Starting vLLM server..."
${ENV_VLLM} -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${VLLM_PORT}" \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM to start..."
READY=0
for i in $(seq 1 90); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM is ready (${i}s)."
        READY=1
        break
    fi
    sleep 3
done
if [ "${READY}" -eq 0 ]; then
    echo "ERROR: vLLM did not start within 270s. Check logs." >&2
    kill "${VLLM_PID}" 2>/dev/null || true
    exit 1
fi

# Shared env vars for all frameworks
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_API_KEY="dummy"
export openai_api_key="dummy"          # lowercase — used by multi_agent_collaboration
export MEISSA_MODEL="${MODEL}"

# ─── 2. Framework I: Continuous Tool Calling ──────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo "[2/5] Framework I: Continuous Tool Calling"
echo "────────────────────────────────────────"
cd "${ROOT_DIR}/environments/continuous_tool_calling"
# XRayVQATool (CheXagent) and XRayPhraseGroundingTool (MAIRA-2) need external servers.
# Set these if the servers are running; otherwise only local tools are used.
export CHEXAGENT_SERVER_URL="${CHEXAGENT_SERVER_URL:-}"
export MAIRA2_SERVER_URL="${MAIRA2_SERVER_URL:-}"
${ENV_MEDRAX} main.py \
    --model "${MODEL}" \
    --model_dir "${MODEL_DIR}" \
    --temp_dir "${ROOT_DIR}/environments/continuous_tool_calling/temp" \
    --demo

# ─── 3. Framework II: Interleaved Thinking with Images ───────────────────
echo ""
echo "────────────────────────────────────────"
echo "[3/5] Framework II: Interleaved Thinking with Images"
echo "────────────────────────────────────────"
cd "${ROOT_DIR}/environments/interleaved_thinking_images"
${ENV_TOOLSERVER} tool_server/tf_eval/run_inference.py \
    --model "${MODEL}" \
    --demo

# ─── 4. Framework III: Multi-Agent Collaboration ─────────────────────────
echo ""
echo "────────────────────────────────────────"
echo "[4/5] Framework III: Multi-Agent Collaboration"
echo "────────────────────────────────────────"
cd "${ROOT_DIR}/environments/multi_agent_collaboration"
${ENV_MDAGENTS} main.py \
    --model "${MODEL}" \
    --demo \
    --use-think-format

# ─── 5. Framework IV: Multi-turn Clinical Simulation ─────────────────────
echo ""
echo "────────────────────────────────────────"
echo "[5/5] Framework IV: Multi-turn Clinical Simulation"
echo "────────────────────────────────────────"
cd "${ROOT_DIR}/environments/clinical_simulation"
${ENV_MEDAGENTSIM} medsim/main.py \
    --model "${MODEL}" \
    --mode demo \
    --total_inferences 5

# ─── Cleanup ──────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "All demos completed."
echo "=============================="
echo "Stopping vLLM server (PID ${VLLM_PID})..."
kill "${VLLM_PID}" 2>/dev/null || true
wait "${VLLM_PID}" 2>/dev/null || true
echo "Done."
