# Meissa Demo Samples

This directory contains demo data for the four Meissa agent frameworks.
Each sample was selected from actual Meissa inference results to demonstrate
the model's distinctive multi-step reasoning capabilities.

---

## Framework I: Continuous Tool Calling (`continuous_tool_calling/`)

**Source**: MIMIC-CXR-VQA (PhysioNet — requires credentialing)
**Sample**: idx 6285, patient p12706984, study s51503517
**Question**: "What are all the anatomical findings present in either the right apical zone or the left mid lung zone?"
**Ground truth**: lung opacity
**Tool call chain**: `ChestXRayReportGenerator` → `XRayVQA (CheXagent)` → `XRayPhraseGrounding (MAIRA-2)` (3 tools)

> **Note**: The image file is not distributed here because MIMIC-CXR requires a PhysioNet Data Use Agreement.
> See [PhysioNet MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) for access instructions.

---

## Framework II: Interleaved Thinking with Images (`interleaved_thinking_images/`)

**Source**: PathVQA dataset (open access)
**Sample**: idx pathvqa_10927
**Image**: `pathvqa_demo.jpg` (Anitschkow cell from rheumatic heart disease histology slide)
**Question**: "What shows owl-eye appearance of central chromatin mass and perinuclear halo?"
**Ground truth**: Anitschkow cell
**Tool call chain**: `ZoomInSubfigure` × 4 → `Terminate` (iterative region zoom to locate labeled cell)

---

## Framework III: Multi-Agent Collaboration (`multi_agent_collaboration/`)

**Source**: PubMedQA (open access)
**Sample**: PubMed ID 11481599
**Question**: "Acute respiratory distress syndrome in children with malignancy—can we predict outcome?"
**Ground truth**: yes
**Agent structure**: 9 agents (Recruiter, Medical Expert, Pediatrician, Pulmonologist,
  Pediatric Critical Care Specialist, Hematologist/Oncologist, Critical Care Nurse,
  2× Medical Assistant, Moderator), 53 messages

---

## Framework IV: Multi-turn Clinical Simulation (`clinical_simulation/`)

**Source**: MIMIC-IV (open access via HuggingFace: [ItsMaxNorm/MedAgentSim-datasets](https://huggingface.co/datasets/ItsMaxNorm/MedAgentSim-datasets))
**Sample**: scenario_10
**Patient**: 65-year-old male, persistent dry cough × 3 weeks, ex-smoker
**Ground truth**: Right Lower Lobe Pneumonia
**Simulation**: 9-turn Doctor/Patient/Measurement dialogue; doctor requests Chest X-Ray and CBC before diagnosing correctly
