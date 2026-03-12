# Evaluation

This document describes how to run evaluations for all four frameworks.

---

## Requirements

- Meissa-4B model downloaded (see README for HuggingFace link)
- For MIMIC benchmarks: PhysioNet access (see [data/README.md](../data/README.md))
- GPU: 24GB+ VRAM (single A100 for inference)

Set the model path:
```bash
export MEISSA_MODEL=/path/to/Meissa-4B
# or use HuggingFace:
export MEISSA_MODEL=CYX1998/Meissa-4B
```

---

## Framework I: Continuous Tool Calling

**Benchmark**: MIMIC-CXR-VQA
**Requires**: PhysioNet credentialed access to MIMIC-CXR-JPG and MIMIC-CXR-VQA

```bash
export MIMIC_CXR_VQA_ROOT=/path/to/mimic-cxr-vqa/1.0.0
export MIMIC_CXR_JPG_ROOT=/path/to/mimic-cxr-jpg/2.1.0

bash scripts/eval_continuous_tool_calling_mimic_cxr_vqa.sh
```

**Expected GPU**: 32GB (tools load additional specialist models)
**Estimated time**: ~4 hours on A100

The evaluation script runs the 8-tool agent on the MIMIC-CXR-VQA test set and computes soft-match accuracy with medical synonym normalization.

---

## Framework II: Interleaved Thinking with Images

**Benchmark**: PathVQA
**Requires**: No restricted data (PathVQA is freely available)

```bash
# PathVQA test set is included in this repo
bash scripts/eval_interleaved_thinking_images_pathvqa.sh
```

**Expected GPU**: 24GB
**Estimated time**: ~2 hours on A100

Additional benchmarks (require setup):
```bash
# SLAKE
bash scripts/eval_interleaved_thinking_images_slake.sh

# VQArad
bash scripts/eval_interleaved_thinking_images_vqarad.sh

# MIMIC-CXR-VQA (requires PhysioNet)
bash scripts/eval_interleaved_thinking_images_mimic.sh
```

---

## Framework III: Multi-Agent Collaboration

**Benchmark**: PubMedQA
**Requires**: No restricted data (PubMedQA is freely available)

```bash
# PubMedQA test set is included in this repo
bash scripts/eval_multi_agent_collaboration_pubmedqa.sh
```

**Expected GPU**: 24GB
**Estimated time**: ~1 hour on A100

Additional benchmarks:
```bash
# MedQA (USMLE) — freely downloadable
bash scripts/eval_multi_agent_collaboration_medqa.sh

# PathVQA (multimodal reasoning)
bash scripts/eval_multi_agent_collaboration_pathvqa.sh
```

---

## Framework IV: Multi-turn Clinical Simulation

**Benchmark**: MIMIC-IV
**Requires**: PhysioNet credentialed access to MIMIC-IV and MIMIC-CXR-JPG

```bash
export MIMIC_IV_ROOT=/path/to/mimiciv/3.1
export MIMIC_CXR_JPG_ROOT=/path/to/mimic-cxr-jpg/2.1.0

bash scripts/eval_clinical_simulation_mimic_iv.sh
```

**Expected GPU**: 24GB
**Estimated time**: ~3 hours on A100

---

## Run All (Quick Mode)

Evaluates on open benchmarks only (no PhysioNet required):

```bash
bash scripts/eval_all.sh --quick
```

This runs PathVQA (Framework II) and PubMedQA (Framework III).

Run all including MIMIC benchmarks:
```bash
bash scripts/eval_all.sh --full
```

---

## GPU Requirements Summary

| Benchmark | Framework | GPU Memory | Estimated Time |
|-----------|-----------|-----------|----------------|
| MIMIC-CXR-VQA | I | 32GB | ~4h |
| PathVQA | II | 24GB | ~2h |
| SLAKE | II | 24GB | ~1h |
| VQArad | II | 24GB | ~30min |
| PubMedQA | III | 24GB | ~1h |
| MedQA | III | 24GB | ~2h |
| MIMIC-IV | IV | 24GB | ~3h |

---

## Evaluation Metrics

| Framework | Benchmark | Metric |
|-----------|-----------|--------|
| I | MIMIC-CXR-VQA | Soft-match accuracy (synonym normalization) |
| II | PathVQA | Overall / Yes-No / Open-ended accuracy |
| II | SLAKE / VQArad | Accuracy |
| III | PubMedQA | yes/no/maybe classification accuracy |
| III | MedQA | MCQ accuracy (A/B/C/D letter format) |
| IV | MIMIC-IV | Diagnosis accuracy (LLM-based semantic comparison) |

---

## Comparing with Baseline Models

To reproduce baseline comparisons (GPT-4V, Gemini Flash, etc.), replace `MEISSA_MODEL` with the respective API endpoint and set the appropriate API key. See per-environment eval scripts for `--model_backend` options.
