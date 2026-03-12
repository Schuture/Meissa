# Continuous Tool Calling — SFT Data

The training data for Framework I (Continuous Tool Calling) is derived from **MIMIC-CXR-VQA**, which requires PhysioNet credentialed access and **cannot be redistributed**.

## Data Statistics

| Source | Samples | Description |
|--------|---------|-------------|
| MIMIC-CXR-VQA | 4,898 | Chest X-ray VQA with tool-calling trajectories |

## Reconstruct the Data

After obtaining PhysioNet access (see [data/README.md](../../README.md)):

```bash
export MIMIC_CXR_VQA_ROOT=/path/to/mimic-cxr-vqa/1.0.0
export MIMIC_CXR_JPG_ROOT=/path/to/mimic-cxr-jpg/2.1.0

python environments/continuous_tool_calling/eval/medrax_mimic_cxr_vqa_distill.py \
    --data_root ${MIMIC_CXR_VQA_ROOT} \
    --image_root ${MIMIC_CXR_JPG_ROOT} \
    --output_dir data/sft_samples/continuous_tool_calling/ \
    --mode tier3
```
