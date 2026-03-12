# Clinical Simulation — SFT Data

Framework IV (Multi-turn Clinical Simulation) uses two data sources:

- **MedQA** (open): Available on HuggingFace ([CYX1998/Meissa-SFT](https://huggingface.co/datasets/CYX1998/Meissa-SFT))
- **MIMIC-CXR** (restricted): Requires PhysioNet credentialed access and cannot be redistributed.

## Data Statistics

| Source | Samples | Availability |
|--------|---------|-------------|
| MedQA | 6,408 | [CYX1998/Meissa-SFT](https://huggingface.co/datasets/CYX1998/Meissa-SFT) |
| MIMIC-CXR | 1,266 | PhysioNet access required |
| **Total** | **7,674** | |

## Reconstruct MIMIC Data

After obtaining PhysioNet access (see [data/README.md](../../README.md)):

```bash
export MIMIC_IV_DIR=/path/to/mimiciv/3.1
export MIMIC_CXR_DIR=/path/to/mimic-cxr-jpg/2.1.0
export MIMIC_NOTE_DIR=/path/to/mimic-iv-note/2.2/note

python environments/clinical_simulation/data/step0_preprocess_mimic.py \
    --output data/sft_samples/clinical_simulation/mimic_sft.jsonl
```
