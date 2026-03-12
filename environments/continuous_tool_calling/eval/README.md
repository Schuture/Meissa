# Evaluation: Framework I — Continuous Tool Calling

## Benchmark: MIMIC-CXR-VQA

MIMIC-CXR-VQA requires PhysioNet credentialing. You cannot run this evaluation without approved access.

### Step 1 — Apply for MIMIC access

1. Create a PhysioNet account at https://physionet.org/register/
2. Complete the CITI "Data or Specimens Only Research" training
3. Apply for access to:
   - [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) (images)
   - [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) (reports)
4. Download and organize the data locally

### Step 2 — Download MIMIC-CXR-VQA annotations

The VQA annotation files are available from the dataset authors:
https://github.com/baeseongsu/mimic-cxr-vqa

### Step 3 — Run evaluation

```bash
export MEISSA_MODEL=CYX1998/Meissa-4B   # or local path

python eval/run_mimic_cxr_vqa.py \
    --model $MEISSA_MODEL \
    --data_path /path/to/mimic_cxr_vqa_test.json \
    --image_dir /path/to/mimic-cxr-jpg/files \
    --output results/ctc_mimic_cxr_vqa.json
```

### Expected Results

| Model | MIMIC-CXR-VQA Acc. |
|-------|-------------------|
| Meissa-4B | See paper Table 2 |
| Qwen3-VL-4B (base) | See paper Table 2 |
