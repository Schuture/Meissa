# Agent Environments

Meissa unifies four medical agent paradigms. Each is implemented in `environments/` with the folder name matching the paper's framework naming.

---

## Framework I: Continuous Tool Calling

**Directory**: `environments/continuous_tool_calling/`
**Related work**: MedRAX (ICML 2025)
**Primary benchmark**: MIMIC-CXR-VQA

The agent sequentially invokes specialized radiology tools in a multi-step reasoning loop, deciding at each step which tool to call based on previous observations.

### Tools

| Tool | Model | Function |
|------|-------|----------|
| `ChestXRayClassifier` | DenseNet-121 | Classify findings (14 categories) |
| `ChestXRaySegmentation` | PSPNet | Segment anatomical regions |
| `CheXagent` | Remote VQA model | Open-ended chest X-ray QA |
| `LLaVA-Med` | LLaVA-Med-7B | Medical VQA |
| `XRayPhraseGrounding` | Maira-2 | Ground phrases to bounding boxes |
| `ChestXRayReportGenerator` | SwinV2 | Generate radiology reports |
| `DICOMProcessor` | Rule-based | Parse DICOM metadata |
| `ImageVisualizer` | PIL | Display and annotate images |

### Setup

```bash
cd environments/continuous_tool_calling
pip install -r requirements.txt

# Start remote tool servers (CheXagent and Maira-2 require separate envs)
export CHEXAGENT_SERVER_URL=http://127.0.0.1:19101
export MAIRA2_SERVER_URL=http://127.0.0.1:19102
```

### Interaction Format

```
Human: <image>\nWhat findings are present in this chest X-ray?

Model: <think>I'll start with classification.</think>
       {"name": "ChestXRayClassifier", "arguments": {"image_path": "..."}}

Tool:  {"findings": ["pleural effusion"], "confidence": 0.91}

Model: <think>Effusion confirmed. Check segmentation.</think>
       {"name": "ChestXRaySegmentation", "arguments": {"image_path": "...", "region": "pleural"}}

Tool:  {"mask": "...", "area_fraction": 0.23}

Model: <think>Large effusion. Final answer.</think>
       [FINAL] Large right pleural effusion occupying 23% of the right hemithorax.
```

---

## Framework II: Interleaved Thinking with Images

**Directory**: `environments/interleaved_thinking_images/`
**Related work**: Ophiuchus
**Primary benchmark**: PathVQA

The agent iteratively calls vision tools where each tool produces a modified or annotated image that is injected back into the conversation. This creates interleaved image-text reasoning chains.

### Tools

| Tool | Function |
|------|----------|
| `ZoomInSubfigure` | Zoom into a region of interest |
| `SegmentRegionAroundPoint` (SAM2) | Segment an object at a given point |
| `BioMedParseTextSeg` | Segment by text description |
| `OCR` | Extract text from image |
| `Point` | Highlight a point on the image |
| `Crop` | Crop to bounding box |
| `DrawHorizontalLineByY` | Draw horizontal reference line |
| `DrawVerticalLineByX` | Draw vertical reference line |
| `Terminate` | End tool use, provide final answer |

### Setup

```bash
cd environments/interleaved_thinking_images
pip install -r tool_server/requirements.txt

# Start the tool server
python tool_server/tf_eval/run_server.py --port 8080
```

### Interaction Format

```
Human: <image>\nWhat cell type is shown in the pathology slide?

Model: <think>USE_TOOL: need to zoom in on the cells.</think>
       {"name": "ZoomInSubfigure", "arguments": {"bbox": [100, 150, 300, 350]}}

Tool:  <image>  [zoomed image injected here]
       Zoomed to region [100, 150, 300, 350]

Model: <think>Now I can see the nuclear features clearly.</think>
       {"name": "Terminate", "arguments": {"answer": "plasma cells"}}
```

---

## Framework III: Multi-Agent Collaboration

**Directory**: `environments/multi_agent_collaboration/`
**Related work**: MDAgents (NeurIPS 2024 Oral)
**Primary benchmark**: PubMedQA

The system routes queries to one of three collaboration modes based on assessed complexity:
- **Basic** (low complexity): single-expert direct answer
- **Intermediate** (medium complexity): multi-agent debate among specialists
- **Advanced** (high complexity): multidisciplinary team consensus

As a single model, Meissa *emulates* multi-agent interaction by generating all agent roles sequentially within one forward pass.

### Virtual Tools (SFT Format)

| Tool | Function |
|------|----------|
| `AssessDifficulty` | Assess query complexity (basic/intermediate/advanced) |
| `RecruitExperts` | Identify relevant medical specialties |
| `ConsultExperts` | Generate expert opinions per specialty |
| `FacilitateDebate` | Synthesize conflicting expert views |
| `SynthesizeReports` | Produce final consensus answer |

### Interaction Format

```
Human: Does azithromycin reduce exacerbations in COPD patients?
       [abstract provided]

Model: <think>This requires evidence synthesis.</think>
       {"name": "AssessDifficulty", "arguments": {"query": "..."}}

Tool:  {"difficulty": "intermediate", "rationale": "Requires RCT interpretation"}

Model: {"name": "RecruitExperts", "arguments": {"specialties": ["pulmonology", "pharmacology"]}}
       ...
       [FINAL] yes
```

---

## Framework IV: Multi-turn Clinical Simulation

**Directory**: `environments/clinical_simulation/`
**Related work**: AgentClinic, MedAgentSim (MICCAI 2025)
**Primary benchmark**: MIMIC-IV

The agent plays a physician in an OSCE (Objective Structured Clinical Examination) simulation: reading a patient presentation, requesting physical exams and diagnostic tests, then issuing a diagnosis.

### Tools

| Tool | Function |
|------|----------|
| `RequestPhysicalExam` | Request a physical examination finding |
| `RequestTest` | Order a diagnostic test (lab, imaging) |
| `Terminate` | Issue final diagnosis |

### Setup

```bash
cd environments/clinical_simulation
pip install -r requirements.txt
```

### Interaction Format

```
Human: Patient: 65-year-old male presents with progressive dyspnea and bilateral leg edema.
       Vital signs: BP 145/90, HR 92, RR 18, SpO2 94% on room air.
       What is your diagnosis?

Model: <think>I need to assess cardiac function.</think>
       {"name": "RequestTest", "arguments": {"test": "echocardiogram"}}

Tool:  Echocardiogram: EF 35%, dilated LV, mild MR, elevated LVEDP.

Model: <think>Reduced EF with pulmonary congestion. Heart failure.</think>
       {"name": "RequestTest", "arguments": {"test": "BNP"}}

Tool:  BNP: 1250 pg/mL (markedly elevated)

Model: <think>Confirmed HFrEF.</think>
       {"name": "Terminate", "arguments": {}}
       [FINAL] Heart failure with reduced ejection fraction (HFrEF)
```

---

## Adding a Custom Environment

Meissa's unified training format (state–action–observation trajectories in ShareGPT) makes it extensible to new medical environments. To add a new framework:

1. Create your environment in `environments/your_framework/`
2. Define tools following the JSON schema in `environments/continuous_tool_calling/tools_manifest.json`
3. Generate SFT data in ShareGPT format (see [data/README.md](../data/README.md) for the format spec)
4. Register your dataset in `training/configs/dataset_info.json`
5. Add an entry to `training/configs/train_qwen3vl_meissa_combined.yaml`
