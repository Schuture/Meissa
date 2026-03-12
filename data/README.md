# Data

This document explains which data is freely available, which requires access requests, and how to reconstruct the full training set.

---

## Overview

**Total SFT samples: 43,210**

| Framework | Dataset | Samples | Availability |
|-----------|---------|---------|-------------|
| I: Continuous Tool Calling | MIMIC-CXR-VQA | 4,898 | PhysioNet required |
| II: Interleaved Thinking | PathVQA | 7,674 | Free |
| II: Interleaved Thinking | MIMIC-CXR-VQA | 4,684 | PhysioNet required |
| II: Interleaved Thinking | SLAKE | 1,698 | Free |
| II: Interleaved Thinking | VQA-RAD | 1,155 | Free |
| III: Multi-Agent Collaboration | MIMIC-CXR-VQA | 7,344 | PhysioNet required |
| III: Multi-Agent Collaboration | PathVQA | 4,616 | Free |
| III: Multi-Agent Collaboration | MedQA | 2,575 | Free |
| III: Multi-Agent Collaboration | PubMedQA | 892 | Free |
| IV: Clinical Simulation | MedQA | 6,408 | Free |
| IV: Clinical Simulation | MIMIC-CXR | 1,266 | PhysioNet required |
| | **Open-source total** | **25,018** | |
| | **PhysioNet-restricted total** | **18,192** | |

**Note**: Raw Gemini-generated trajectory files are not released due to Google Terms of Service. The SFT data released here is in the final processed format.

---

## Freely Available Data

The non-MIMIC SFT subset (25,018 samples) is hosted on HuggingFace:

```
https://huggingface.co/datasets/CYX1998/Meissa-SFT
```

Download via CLI:
```bash
huggingface-cli download CYX1998/Meissa-SFT --repo-type dataset --local-dir data/sft_data
```

Or load directly in Python:
```python
from datasets import load_dataset
ds = load_dataset("CYX1998/Meissa-SFT")
```

All files are in ShareGPT format compatible with LLaMA-Factory.

---

## PhysioNet-Restricted Data (MIMIC)

MIMIC-CXR and MIMIC-IV require credentialed access through PhysioNet.

### Step 1: Apply for PhysioNet Access

1. Create an account at https://physionet.org/register/
2. Complete the CITI training (required): https://physionet.org/about/citi-course/
3. Apply for access to:
   - **MIMIC-CXR-JPG**: https://physionet.org/content/mimic-cxr-jpg/
   - **MIMIC-CXR-VQA**: https://physionet.org/content/mimic-cxr-vqa/
   - **MIMIC-IV**: https://physionet.org/content/mimiciv/
4. Download the datasets after approval (typically 1–2 weeks).

### Step 2: Set Environment Variables

```bash
export DATA_ROOT=/path/to/your/physionet/files
export MIMIC_CXR_ROOT=${DATA_ROOT}/mimic-cxr-jpg/2.1.0
export MIMIC_CXR_VQA_ROOT=${DATA_ROOT}/mimic-cxr-vqa/1.0.0
export MIMIC_IV_ROOT=${DATA_ROOT}/mimiciv/3.1
```

### Step 3: Reconstruct SFT Data

For Framework I (Continuous Tool Calling):
```bash
python environments/continuous_tool_calling/eval/medrax_mimic_cxr_vqa_distill.py \
    --data_root ${MIMIC_CXR_VQA_ROOT} \
    --image_root ${MIMIC_CXR_ROOT} \
    --output data/sft_samples/continuous_tool_calling/
```

For Framework II and III (MIMIC-CXR-VQA portions):
```bash
# These follow the same distillation process as Framework I
# but with different agent configurations
```

For Framework IV (Clinical Simulation):
```bash
python environments/clinical_simulation/data/step0_preprocess_mimic.py \
    --output data/sft_samples/clinical_simulation/mimic_sft.jsonl
```

---

## Data Format

All SFT data follows the ShareGPT format used by LLaMA-Factory:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat abnormality is visible in this chest X-ray?"
    },
    {
      "from": "function_call",
      "value": "<think>I need to classify the chest X-ray findings first.</think>\n{\"name\": \"ChestXRayClassifier\", \"arguments\": {\"image_path\": \"...\"}}"
    },
    {
      "from": "observation",
      "value": "{\"findings\": [\"pleural effusion\", \"cardiomegaly\"], \"confidence\": 0.92}"
    },
    {
      "from": "gpt",
      "value": "<think>The classifier confirms pleural effusion and cardiomegaly.</think>\n[FINAL] The chest X-ray shows pleural effusion and cardiomegaly."
    }
  ],
  "system": "You are a medical assistant with access to radiology tools...",
  "tools": "[{\"name\": \"ChestXRayClassifier\", ...}]",
  "images": ["images/pathvqa/sample_001.jpg"],
  "meta": {
    "source": "mimic_cxr_vqa",
    "framework": "continuous_tool_calling"
  }
}
```

Key fields:
- `from: "function_call"` — model's tool invocation with `<think>` reasoning
- `from: "observation"` — tool execution result (may include `<image>` for Framework II)
- `[FINAL]` — marks the final answer in the last `gpt` turn
