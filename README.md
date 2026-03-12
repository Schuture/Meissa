# Meissa: Multi-modal Medical Agentic Intelligence

<p align="center">
  <a href="https://arxiv.org/abs/2603.09018"><img src="https://img.shields.io/badge/arXiv-2603.09018-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/CYX1998/Meissa-4B"><img src="https://img.shields.io/badge/HuggingFace-Meissa--4B-yellow" alt="HuggingFace Model"></a>
  <a href="https://huggingface.co/datasets/CYX1998/Meissa-SFT"><img src="https://img.shields.io/badge/HuggingFace-Meissa--SFT-blue" alt="HuggingFace Dataset"></a>
  <a href="https://github.com/Schuture/Meissa/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
</p>

> [Paper](https://arxiv.org/abs/2603.09018) | [Model (HuggingFace)](https://huggingface.co/CYX1998/Meissa-4B) | [Dataset (HuggingFace)](https://huggingface.co/datasets/CYX1998/Meissa-SFT)

---

## Overview

Multi-modal large language models (MM-LLMs) have shown strong performance in medical image understanding and clinical reasoning. Recent medical agent systems extend them with tool use and multi-agent collaboration, enabling complex decision-making beyond single-pass inference. However, these systems rely on proprietary frontier models (GPT, Gemini), whose API-based deployment incurs high cost, high latency, and privacy risks.

**Meissa** is a lightweight **4B-parameter** medical MM-LLM that brings full agentic capability **offline**. Instead of imitating static answers, Meissa learns both *when* to engage external interaction (strategy selection) and *how* to execute multi-step interaction (strategy execution) by distilling structured trajectories from frontier agent systems.

**Three core contributions:**
1. **Unified trajectory modeling**: trajectories (reasoning and action traces) are represented within a single state–action–observation formalism, allowing one model to generalize across heterogeneous medical environments.
2. **Three-tier stratified supervision**: the model's own errors trigger progressive escalation from direct reasoning to tool-augmented and multi-agent interaction, explicitly learning difficulty-aware strategy selection.
3. **Prospective–retrospective supervision**: pairing exploratory forward traces with hindsight-rationalized execution traces enables stable learning of effective interaction policies.

Trained on ~40K curated trajectories, Meissa **matches or exceeds proprietary frontier agents in 10 of 16 evaluation settings** across 13 medical benchmarks spanning radiology, pathology, and clinical reasoning. With **~25× fewer parameters** than Gemini-3, Meissa operates fully offline with **~22× lower end-to-end latency** compared to API-based deployment.

<!-- ![Overview](assets/fig1_overview.png) -->

---

## News

- **[2026-03-12]** Code, model, and data released.

---

## Agent Environments

Meissa unifies four heterogeneous medical agent paradigms:

| Framework | Paradigm | Key Benchmark | Related Work |
|-----------|----------|---------------|--------------|
| **I. Continuous Tool Calling** | Sequential invocation of 8 radiology tools | MIMIC-CXR-VQA | MedRAX (ICML 2025) |
| **II. Interleaved Thinking with Images** | Multi-round visual tool use with image injection | PathVQA | Ophiuchus, OpenThinkIMG |
| **III. Multi-Agent Collaboration** | Adaptive difficulty routing + multi-agent debate | PubMedQA | MDAgents (NeurIPS 2024 Oral) |
| **IV. Multi-turn Clinical Simulation** | Doctor-patient simulation with exam/test ordering | MIMIC-IV | AgentClinic, MedAgentSim (MICCAI 2025) |

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/Schuture/Meissa.git
cd Meissa
```

Each framework requires its own conda environment due to conflicting dependencies:

```bash
# Framework I: Continuous Tool Calling
conda create -n medrax python=3.10 && conda activate medrax
pip install -r environments/continuous_tool_calling/requirements.txt

# Framework II: Interleaved Thinking with Images
conda create -n tool-server python=3.10 && conda activate tool-server
pip install -r environments/interleaved_thinking_images/requirements.txt

# Framework III: Multi-Agent Collaboration
conda create -n mdagents python=3.10 && conda activate mdagents
pip install -r environments/multi_agent_collaboration/requirements.txt

# Framework IV: Multi-turn Clinical Simulation
conda create -n medagentsim python=3.10 && conda activate medagentsim
pip install -r environments/clinical_simulation/requirements.txt

# vLLM (shared model server for all frameworks)
conda create -n vllm python=3.10 && conda activate vllm
pip install vllm
```

### 2. Download the Model

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("CYX1998/Meissa-4B", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("CYX1998/Meissa-4B", trust_remote_code=True)
```

### 3. Serve the Model with vLLM

Meissa uses [vLLM](https://github.com/vllm-project/vllm) as the inference backend. The `--tool-call-parser hermes` flag is **required** to enable the model's tool-calling ability.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model CYX1998/Meissa-4B \
    --port 8877 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# Set the endpoint for all frameworks
export OPENAI_BASE_URL="http://127.0.0.1:8877/v1"
export OPENAI_API_KEY="dummy"
```

### 4. (Framework I only) Start External Tool Servers

Framework I's XRayVQATool and XRayPhraseGroundingTool call [CheXagent](https://huggingface.co/StanfordAIMI/CheXagent-2-3b) and [MAIRA-2](https://huggingface.co/microsoft/maira-2) as remote HTTP servers. These are large models that run on separate GPUs. You can deploy them using any standard model serving framework (e.g., vLLM, FastAPI), then point Meissa to them:

```bash
export CHEXAGENT_SERVER_URL="http://<host>:19101"
export MAIRA2_SERVER_URL="http://<host>:19102"
```

If these environment variables are not set, Framework I will fall back to loading the models locally (requires sufficient GPU memory).

### 5. Run Demos

```bash
# Run all 4 frameworks end-to-end (starts vLLM automatically, requires SLURM)
sbatch scripts/test_demo_all.sh

# Or run individual frameworks:
bash scripts/demo_continuous_tool_calling.sh       # Framework I
bash scripts/demo_interleaved_thinking_images.sh   # Framework II
bash scripts/demo_multi_agent_collaboration.sh     # Framework III
bash scripts/demo_clinical_simulation.sh           # Framework IV
```

---

## Repository Structure

```
Meissa/
├── meissa/                          # Python package
│   └── __init__.py
│
├── environments/
│   ├── continuous_tool_calling/     # Framework I (8 radiology tools)
│   ├── interleaved_thinking_images/ # Framework II (vision tool server)
│   ├── multi_agent_collaboration/   # Framework III (multi-agent debate)
│   └── clinical_simulation/         # Framework IV (doctor-patient sim)
│
├── data/
│   └── sft_samples/                 # MIMIC reconstruction instructions (READMEs)
│       ├── continuous_tool_calling/ # MIMIC-CXR-VQA reconstruction
│       └── clinical_simulation/     # MIMIC-IV reconstruction
│
├── training/
│   └── configs/                     # LLaMA-Factory training configs
│
├── scripts/                         # Demo and evaluation shell scripts
│
│   # Each directory contains its own README.md with detailed documentation

```

---

## Main Results

> Full results with standard deviations and per-category breakdowns are in the [paper](https://arxiv.org/abs/2603.09018).

### Framework I: Continuous Tool Calling

| Model | MIMIC-CXR-VQA | ChestAgentBench |
|-------|:---:|:---:|
| GPT-4o (Direct) | 40.0 | 56.4 |
| Gemini-3-flash (Direct) | 43.6 | 76.2 |
| GPT-4o + MedRAX | 55.6 | 63.1 |
| Gemini-3-flash + MedRAX | 65.0 | 72.7 |
| Qwen3-VL-4B (Agent) | 51.4 | 46.6 |
| **Meissa (Ours)** | **65.2** | **62.8** |

### Framework II: Interleaved Thinking with Images

| Model | PathVQA | SLAKE | VQA-RAD | OmniMedVQA | MedXpertQA |
|-------|:---:|:---:|:---:|:---:|:---:|
| GPT-5 (Direct) | 60.0 | 73.2 | 64.5 | 75.4 | 40.4 |
| Gemini-3-flash (Direct) | 64.3 | 77.7 | 58.8 | 78.0 | 54.4 |
| Gemini-3-flash + OpenThinkIMG | 74.3 | 73.9 | 52.0 | 77.4 | 69.2 |
| Ophiuchus-7B (Agent) | 74.3 | 83.9 | 73.6 | 78.6 | 39.3 |
| Qwen3-VL-4B (Agent) | 65.3 | 55.6 | 51.8 | 38.1 | 23.9 |
| **Meissa (Ours)** | **78.2** | **82.0** | **70.1** | **82.8** | **36.0** |

### Framework III: Multi-Agent Collaboration

| Model | MedQA | PubMedQA | PathVQA | MIMIC-CXR-VQA |
|-------|:---:|:---:|:---:|:---:|
| GPT-4V (Direct) | 75.0 | 61.5 | 57.9 | 40.0 |
| Gemini-3-flash (Direct) | 75.5 | 66.7 | 64.3 | 43.6 |
| GPT-4V + MDAgents | 88.7 | 75.0 | 65.3 | 55.9 |
| Gemini-3-flash + MDAgents | 75.5 | 71.9 | 56.3 | 64.0 |
| Qwen3-VL-4B (Agent) | 59.8 | 57.3 | 65.5 | 54.9 |
| **Meissa (Ours)** | **57.2** | **77.9** | **67.9** | **59.4** |

### Framework IV: Multi-turn Clinical Simulation

| Model | NEJM | NEJM Ext. | MedQA | MedQA Ext. | MIMIC-IV |
|-------|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 26.7 | 25.8 | 52.8 | 52.3 | 34.4 |
| Gemini-3-flash | 40.0 | 33.3 | 97.9 | 92.3 | 70.6 |
| Llama-3.3-70B | 20.0 | 24.2 | 54.7 | 53.3 | 36.8 |
| Qwen3-VL-4B | 40.0 | 20.8 | 50.5 | 50.0 | 61.1 |
| **Meissa (Ours)** | **46.7** | **23.3** | **49.5** | **46.7** | **84.4** |

---

## Reproduce Main Results

See [scripts/EVALUATION.md](scripts/EVALUATION.md) for full instructions. Quick commands:

```bash
# Evaluate Framework II on PathVQA (no restricted data needed)
bash scripts/eval_interleaved_thinking_images_pathvqa.sh

# Evaluate Framework III on PubMedQA (no restricted data needed)
bash scripts/eval_multi_agent_collaboration_pubmedqa.sh

# For MIMIC-CXR-VQA and MIMIC-IV: see data/README.md for PhysioNet access
bash scripts/eval_continuous_tool_calling_mimic_cxr_vqa.sh   # requires MIMIC-CXR
bash scripts/eval_clinical_simulation_mimic_iv.sh              # requires MIMIC-IV
```

**GPU requirements:**

| Component | GPU Memory | Notes |
|-----------|-----------|-------|
| vLLM serving (Meissa-4B) | ~16GB | 1 GPU, shared across all frameworks |
| CheXagent server | ~20GB | 1 GPU, Framework I only |
| MAIRA-2 server | ~28GB | 1 GPU, Framework I only |
| Training (LoRA SFT) | 8×48GB | ~12h on A6000 |

---

## Data

See [data/README.md](data/README.md) for complete instructions.

### Evaluation Benchmarks

| Dataset | Domain | Link | Access |
|---------|--------|------|--------|
| [MIMIC-CXR-VQA](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/) | Radiology VQA | PhysioNet | Credentialed |
| [ChestAgentBench](https://github.com/bowang-lab/MedRAX) | Radiology (multi-task) | GitHub | Open |
| [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | Pathology VQA | HuggingFace | Open |
| [SLAKE](https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english) | Medical VQA | HuggingFace | Open |
| [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | Radiology VQA | HuggingFace | Open |
| [OmniMedVQA](https://huggingface.co/datasets/OpenGVLab/OmniMedVQA) | Medical VQA | HuggingFace | Open |
| [MedXpertQA](https://huggingface.co/datasets/TsinghuaMedInfo/MedXpertQA) | Expert medical QA | HuggingFace | Open |
| [MedQA](https://huggingface.co/datasets/bigbio/med_qa) | USMLE-style QA | HuggingFace | Open |
| [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | Biomedical research QA | HuggingFace | Open |
| [NEJM / MedQA / MIMIC-IV for Simulation](https://huggingface.co/ItsMaxNorm/MedAgentSim-datasets) | Clinical simulation | Huggingface | Open |

### Training Data

| Data Type | Source | Access |
|-----------|--------|--------|
| SFT trajectories (25,018 samples) | [CYX1998/Meissa-SFT](https://huggingface.co/datasets/CYX1998/Meissa-SFT) | Open |
| SFT trajectories (MIMIC, 18,192 samples) | Reconstructible via scripts | Requires PhysioNet |

**Total training samples:** 43,210 (three-tier stratified: 8.2K direct + 9.8K enhanced + 23.9K agentic)

---

## Training

See [training/README.md](training/README.md) for full instructions.

The released model is trained with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) using LoRA on Qwen3-VL-4B-Instruct.

```bash
# Requires LLaMA-Factory installation and dataset_info.json registered datasets
llamafactory-cli train training/configs/train_qwen3vl_meissa_combined.yaml
```

---

## Medical Disclaimer

**Meissa is a research prototype and is NOT intended for clinical use.** The model may produce incorrect or hallucinated medical information. Do not use it for diagnosis, treatment, or any clinical decision-making.

---

## License

- **Code**: [Apache 2.0](LICENSE)
- **Model weights**: Subject to the [Qwen License](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/LICENSE) in addition to Apache 2.0
- **SFT data**: Subject to the original dataset licenses (PathVQA, PubMedQA, SLAKE, VQArad)

---

## Citation

If you find Meissa useful, please cite:

```bibtex
@inproceedings{meissa2026,
  title={Meissa: Multi-modal Medical Agentic Intelligence},
  author={Chen, Yixiong and Bai, Xinyi and Pan, Yue and Zhou, Zongwei and Yuille,  Alan},
  journal={arXiv preprint arXiv:2603.09018},
  year={2026}
}
```

Related works:
```bibtex
@inproceedings{fallahpour2025medrax,
  title={MedRAX: Medical Reasoning Agent for Chest X-ray},
  author={Fallahpour, Adibvafa and Ma, Jun and Munim, Alif and Lyu, Hongwei and Wang, Bo},
  booktitle={International Conference on Machine Learning},
  pages={15661--15676},
  year={2025},
  organization={PMLR}
}

@article{su2025openthinkimg,
  title={Openthinkimg: Learning to think with images via visual tool reinforcement learning},
  author={Su, Zhaochen and Li, Linjie and Song, Mingyang and Hao, Yunzhuo and Yang, Zhengyuan and Zhang, Jun and Chen, Guanjie and Gu, Jiawei and Li, Juntao and Qu, Xiaoye and others},
  journal={arXiv preprint arXiv:2505.08617},
  year={2025}
}

@article{kim2024mdagents,
  title={Mdagents: An adaptive collaboration of llms for medical decision-making},
  author={Kim, Yubin and Park, Chanwoo and Jeong, Hyewon and Chan, Yik S and Xu, Xuhai and McDuff, Daniel and Lee, Hyeonhoon and Ghassemi, Marzyeh and Breazeal, Cynthia and Park, Hae W},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={79410--79452},
  year={2024}
}

@inproceedings{almansoori2025medagentsim,
  title={MedAgentSim: Self-evolving Multi-agent Simulations for Realistic Clinical Interactions},
  author={Almansoori, Mohammad and Kumar, Komal and Cholakkal, Hisham},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={362--372},
  year={2025},
  organization={Springer}
}

@article{jiang2025incentivizing,
  title={Incentivizing Tool-augmented Thinking with Images for Medical Image Analysis},
  author={Jiang, Yankai and Zhang, Yujie and Zhang, Peng and Li, Yichen and Chen, Jintai and Shi, Xiaoming and Zhen, Shihui},
  journal={arXiv preprint arXiv:2512.14157},
  year={2025}
}
```

---

## Acknowledgements

We thank the teams behind [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [MedRAX](https://github.com/bowang-lab/MedRAX), [OpenThinkIMG](https://github.com/OpenThinkIMG/OpenThinkIMG) [MDAgents](https://github.com/mitmedialab/MDAgents), and [MedAgentSim](https://github.com/MohammadAlmansoori/MedAgentSim) for their open-source contributions.
