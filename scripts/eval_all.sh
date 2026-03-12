#!/usr/bin/env bash
# Meissa: Run all evaluations
#
# Usage:
#   bash scripts/eval_all.sh --quick        # Open benchmarks only (no PhysioNet required)
#   bash scripts/eval_all.sh --full         # All benchmarks (requires PhysioNet access)
#
# Required env vars:
#   MEISSA_MODEL: path or HuggingFace ID of Meissa-4B (default: CYX1998/Meissa-4B)
#
# For --full mode, also set:
#   MIMIC_CXR_VQA_ROOT: path to MIMIC-CXR-VQA dataset
#   MIMIC_CXR_JPG_ROOT: path to MIMIC-CXR-JPG dataset
#   MIMIC_IV_ROOT:      path to MIMIC-IV dataset

set -euo pipefail

MEISSA_MODEL="${MEISSA_MODEL:-CYX1998/Meissa-4B}"
MODE="${1:---quick}"

echo "=== Meissa Evaluation ==="
echo "Model: ${MEISSA_MODEL}"
echo "Mode:  ${MODE}"
echo ""

# Framework II: Interleaved Thinking with Images — PathVQA
echo "[1/4] Framework II: PathVQA"
bash scripts/eval_interleaved_thinking_images_pathvqa.sh
echo ""

# Framework III: Multi-Agent Collaboration — PubMedQA
echo "[2/4] Framework III: PubMedQA"
bash scripts/eval_multi_agent_collaboration_pubmedqa.sh
echo ""

if [[ "${MODE}" == "--full" ]]; then
    # Framework I: Continuous Tool Calling — MIMIC-CXR-VQA
    echo "[3/4] Framework I: MIMIC-CXR-VQA"
    if [[ -z "${MIMIC_CXR_VQA_ROOT:-}" ]]; then
        echo "ERROR: MIMIC_CXR_VQA_ROOT is not set. Skipping."
        echo "See docs/data.md for PhysioNet access instructions."
    else
        bash scripts/eval_continuous_tool_calling_mimic_cxr_vqa.sh
    fi
    echo ""

    # Framework IV: Clinical Simulation — MIMIC-IV
    echo "[4/4] Framework IV: MIMIC-IV"
    if [[ -z "${MIMIC_IV_ROOT:-}" ]]; then
        echo "ERROR: MIMIC_IV_ROOT is not set. Skipping."
        echo "See docs/data.md for PhysioNet access instructions."
    else
        bash scripts/eval_clinical_simulation_mimic_iv.sh
    fi
else
    echo "[3/4] Framework I: MIMIC-CXR-VQA — SKIPPED (use --full for MIMIC benchmarks)"
    echo "[4/4] Framework IV: MIMIC-IV — SKIPPED (use --full for MIMIC benchmarks)"
fi

echo ""
echo "=== Evaluation complete ==="
