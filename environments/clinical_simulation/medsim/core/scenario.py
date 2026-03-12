"https://github.com/samuelschmidgall/"

import json
import random
import os

from pathlib import Path

def resolve_dataset_path(input_path: str, dataset: str) -> Path:
    """Resolve a dataset file path.

    Looks for DATA_ROOT env var first, then falls back to a 'datasets/'
    directory relative to the clinical_simulation environment root.
    """
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        return Path(data_root) / "clinical_simulation" / dataset

    # Fallback: datasets/ dir relative to this file's environment root
    env_root = Path(__file__).resolve().parents[2]
    return env_root / "datasets" / dataset

class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        project_root = Path(os.getcwd())
        data_path = resolve_dataset_path(project_root, "_medqa.jsonl")
        with open(data_path, "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        


class ScenarioMedQAExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQAExtended:
    def __init__(self) -> None:
        project_root = Path(os.getcwd())
        data_path = resolve_dataset_path(project_root, "_medqa_extended.jsonl")
        with open(data_path, "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQAExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        


class ScenarioMIMICIVQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMIMICIV:
    def __init__(self) -> None:
        project_root = Path(os.getcwd())
        data_path = resolve_dataset_path(project_root, "datasets_mimiciv.jsonl")
        with open(data_path, "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMIMICIVQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJMExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict 
        self.question = scenario_dict["question"] 
        self.image_url = scenario_dict["image_url"] 
        self.diagnosis = [_sd["text"] 
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
    
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJMExtended:
    def __init__(self) -> None:
        project_root = Path(os.getcwd())
        data_path = resolve_dataset_path(project_root, "datasets_nejm_extended.jsonl")
        with open(data_path, "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJMExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJM:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict 
        self.question = scenario_dict["question"] 
        self.image_url = scenario_dict["image_url"] 
        self.diagnosis = [_sd["text"] 
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
    
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJM:
    def __init__(self) -> None:
        project_root = Path(os.getcwd())
        data_path = resolve_dataset_path(project_root, "_nejm.jsonl")
        with open(data_path, "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJM(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

# ============================================================
# MedQA Train (full 10K, Gemini-converted OSCE format)
# ============================================================

class ScenarioMedQATrain:
    """Scenario from Gemini-converted MedQA training data (same OSCE structure)."""
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        osce = scenario_dict["OSCE_Examination"]
        self.tests = osce.get("Test_Results", {})
        self.diagnosis = osce.get("Correct_Diagnosis", "")
        self.patient_info = osce.get("Patient_Actor", {})
        self.examiner_info = osce.get("Objective_for_Doctor", "")
        self.physical_exams = osce.get("Physical_Examination_Findings", {})

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> str:
        return self.examiner_info

    def exam_information(self) -> dict:
        import copy
        exams = copy.deepcopy(self.physical_exams)
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderMedQATrain:
    def __init__(self) -> None:
        project_root = Path(os.getcwd())
        data_path = resolve_dataset_path(project_root, "medqa_train_osce.jsonl")
        with open(data_path, "r") as f:
            self.scenario_strs = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # Skip failed conversions
                if record.get("OSCE_Examination") is None:
                    continue
                self.scenario_strs.append(record)
        self.scenarios = [ScenarioMedQATrain(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None:
            return self.sample_scenario()
        return self.scenarios[id]


# ============================================================
# MIMIC-CXR (multimodal: MIMIC-IV + CXR + Notes)
# ============================================================

class ScenarioMIMICCXR:
    """Multimodal scenario linking MIMIC-IV clinical data with CXR images."""
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        osce = scenario_dict["OSCE_Examination"]
        self.tests = osce.get("Test_Results", {})
        self.diagnosis = osce.get("Correct_Diagnosis", "")
        self.patient_info = osce.get("Patient_Actor", {})
        self.examiner_info = osce.get("Objective_for_Doctor", "")
        self.physical_exams = osce.get("Physical_Examination_Findings", {})
        self.chexpert_labels = osce.get("CheXpert_Labels", {})
        self.image_url = osce.get("image_url", None)

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> str:
        return self.examiner_info

    def exam_information(self) -> dict:
        import copy
        exams = copy.deepcopy(self.physical_exams)
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderMIMICCXR:
    def __init__(self) -> None:
        project_root = Path(os.getcwd())
        data_path = resolve_dataset_path(project_root, "mimic_cxr_train.jsonl")
        with open(data_path, "r") as f:
            self.scenario_strs = [json.loads(line) for line in f if line.strip()]
        self.scenarios = [ScenarioMIMICCXR(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None:
            return self.sample_scenario()
        return self.scenarios[id]


# ============================================================
# Legacy model aliases (from original codebase)
# ============================================================

import time
import datetime
MODEL_ALIASES = {
    "llama2": "meta/llama-2-70b-chat",
    "llama2_hf": "meta-llama/Llama-2-70b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-70B-Instruct",
    "mixtral": "mistralai/mixtral-8x7b-instruct-v0.1",
    "llama11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3_7b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "med": "AdaptLLM/medicine-LLM",
    "med_chat": "AdaptLLM/medicine-chat",
    "llama3_1_8": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_70_3p1": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "llama_med": "HPAI-BSC/Llama3.1-Aloe-Beta-8B",
    "qwue":"Qwen/Qwen2.5-0.5B-Instruct",
    "qwue7": "Qwen/Qwen2.5-7B-Instruct"
}

def resolve_model_name(model_name):
    """
    Resolve the model alias to its full name.
    """
    return MODEL_ALIASES.get(model_name, model_name)