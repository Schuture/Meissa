# Training

This document describes how to reproduce the Meissa training from scratch.

---

## Prerequisites

1. **LLaMA-Factory** (>=0.9.4): https://github.com/hiyouga/LLaMA-Factory
2. **Base model**: `Qwen/Qwen3-VL-4B-Instruct` (auto-downloaded from HuggingFace)
3. **GPU**: 8× A6000 48GB recommended; minimum 4× A5000 24GB for reduced batch size and context length
4. **Data**: Prepare training data following [data/README.md](../data/README.md)

Install LLaMA-Factory:
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

---

## Dataset Preparation

### Step 1: Download Open-Source SFT Data

Download the open-source subset (25,018 samples) from HuggingFace:

```bash
# Download from HuggingFace
huggingface-cli download CYX1998/Meissa-SFT --repo-type dataset --local-dir data/sft_data
```

### Step 2: (Optional) Reconstruct MIMIC Data

The remaining 18,192 samples require PhysioNet credentialed access to MIMIC-CXR and MIMIC-IV. See [data/README.md](../data/README.md) for reconstruction instructions.

### Step 3: Register Datasets in LLaMA-Factory

```bash
# Copy dataset_info.json to your LLaMA-Factory data directory
cp training/configs/dataset_info.json /path/to/LLaMA-Factory/data/

# Update the "file_name" fields in dataset_info.json to point to
# the downloaded JSONL files from Step 1 (and Step 2 if applicable)
```

---

## Training Configs

| Config | Framework | Description |
|--------|-----------|-------------|
| `train_qwen3vl_meissa_combined.yaml` | All (combined) | **Main config**: trains on all four frameworks |

---

## Run Training

```bash
llamafactory-cli train training/configs/train_qwen3vl_meissa_combined.yaml
```

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `Qwen/Qwen3-VL-4B-Instruct` | Downloaded automatically |
| LoRA rank | 16–32 | 32 for Framework II |
| LoRA alpha | 32–64 | 2× rank |
| Learning rate | 1e-4 to 5e-5 | See individual configs |
| Training epochs | 3 | |
| Precision | bf16 | |
| Max sequence length | 4096–6144 | 6144 for long trajectories |
| Batch size | 1 per device | Use gradient accumulation |
| Gradient accumulation | 8–16 | |

---

## Merge LoRA Weights

After training, merge LoRA adapters into the base model:

```bash
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --adapter_name_or_path /path/to/lora/checkpoint \
    --template qwen3_vl \
    --finetuning_type lora \
    --export_dir /path/to/merged/model \
    --export_size 2 \
    --export_device cpu
```

The pre-merged model is available at HuggingFace: `CYX1998/Meissa-4B`

---

## Compute Budget

| Resource | Amount |
|----------|--------|
| GPUs | 8× A6000 48GB |
| Training time | ~12 hours |
| Training samples | ~43K |
| GPU memory | ~30GB per GPU (bf16 + LoRA) |

