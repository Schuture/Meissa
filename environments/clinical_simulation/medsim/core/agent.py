import os
import re
import json
import time
import random
import logging

from medsim.query_model import BAgent, extract_question, generate_possible_diagnoses, generate_answer_choices

logger = logging.getLogger(__name__)


class PatientAgent:
    def __init__(self, backend_str="Qwen/Qwen2.5-0.5B-Instruct") -> None:
        # language model backend for patient agent
        self.backend = backend_str
        self.pipe = BAgent(model_name=backend_str)

    def update_scenario(self, scenario, bias_present=None):
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # sample initial question from dataset
        self.scenario = scenario        
        self.reset()
        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]
    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            logger.warning("Unsupported bias type: %s, ignoring.", self.bias_present)
        return ""

    def inference_patient(self, question) -> str:
        answer = self.pipe.query_model("\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the doctor response: " + question + "Now please continue your dialogue\nPatient: ", self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer
    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(self.symptoms)
        return base + bias_prompt + symptoms
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

class DoctorAgent:
    def __init__(self, backend_str="gpt4", graph=False) -> None:
        self.backend = backend_str
        # if graph:
        #     self.pipe = LBAgent(model_name=backend_str)
        # else:
        self.pipe = BAgent(model_name=backend_str)
        self.num_doctors = 5
    def update_scenario(self, scenario, max_infs=20, bias_present=None, img_request=False):
        # number of inference calls to the doctor
        self.infs = 0
        # maximum number of inference calls to the doctor
        self.MAX_INFS = max_infs
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for doctor
        self.presentation = ""
        # language model backend for doctor agent

        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()      
        self.img_request = img_request
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            logger.warning("Unsupported bias type: %s, ignoring.", self.bias_present)
        return ""

    def inference_doctor(self, question, image_requested=False, thread_id = 1, scenario_id=None) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS-1:
            return self.internal_discussion(question, scenario_id=scenario_id)
        answer = self.pipe.query_model("\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: ", self.system_prompt(), image_requested=image_requested, scene=self.scenario, thread_id = thread_id)
        self.agent_hist += question + "\n\n" + answer + "\n\nq[[[]"
        self.infs += 1
        return answer

    def col_system_prompt(self):
        return (
            "You are a team of collaborative doctors engaged in a discussion about a patient's case. Each doctor is expected to provide their professional opinion "
            "based on the presented symptoms and test results. The objective is to work together to reach a consensus diagnosis. "
            "Each response should be formatted as 'Doctor X: [response]'. Once the discussion concludes, the group will collectively decide on the diagnosis, "
            "summarized as 'DIAGNOSIS READY: [final diagnosis based on majority vote]'."
        )

    def internal_discussion(self, patient_statement, image_requested = None, thread_id=1, scenario_id=None) -> str:
        """
        Simulates an internal multi-doctor discussion to refine diagnosis.

        Args:
            patient_statement (str): The patient's latest statement or response.
            context (str): Additional context (e.g., test results or history).

        Returns:
            str: The final diagnosis after internal discussion.
        """
        discussion_prompt = (
            f"Patient Statement: {patient_statement}\n"
            f"Additional Context: {self.agent_hist}\n"
            f"Doctors, please discuss this case and refine your opinions based on the symptoms and test results. "
        )
        responses = []

        for i in range(1, self.num_doctors + 1):
            prompt = f"{discussion_prompt} Doctor {i}, please share your opinion on the diagnosis and reasoning.\n"
            response = self.pipe.query_model(
                prompt, self.col_system_prompt(),
                image_requested=image_requested, scene=self.scenario, thread_id=thread_id,
            )
            responses.append(f"Doctor {i}: {response.strip()}")

        # Final consensus based on discussion
        consensus_prompt = (
            "The following discussion occurred among doctors:\n\n"
            + "\n".join(responses)
            + "\n\nBased on this discussion and findings, provide a Final Diagnosis."
        )

        final_response = self.pipe.query_model(consensus_prompt, self.col_system_prompt(), image_requested=None, scene=self.scenario, thread_id = thread_id)
        self.agent_hist += "\n".join(responses) + f"\nFinal Diagnosis: {final_response.strip()}\n"

        return final_response.strip()

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {} questions total before you must make a decision. You have asked {} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"".format(self.MAX_INFS, self.infs) + ("You may also request medical images related to the disease to be returned with \"REQUEST IMAGES\"." if self.img_request else "")
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(self.presentation)
        return base + bias_prompt + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()

class MeasurementAgent:
    def __init__(self, backend_str="gpt4") -> None:
        # language model backend for measurement agent
        self.backend = backend_str
        self.pipe = BAgent(model_name=backend_str)

    def update_scenario(self, scenario):
        self.agent_hist = ""
        # presentation information for measurement 
        self.presentation = ""
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()
    def inference_measurement(self, question) -> str:
        answer = str()
        answer = self.pipe.query_model("\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor measurement request: " + question, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(self.information)
        return base + presentation
    
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()

def compare_results(diagnosis, correct_diagnosis, mod_pipe, similarity_threshold=0.8, tries=3, timeout=5.0):
    """
    Compares the doctor's diagnosis with the correct diagnosis using a similarity-based approach.

    Args:
        diagnosis (str): The diagnosis provided by the doctor.
        correct_diagnosis (str): The correct diagnosis for the case.
        mod_pipe (BAgent): The initialized moderator instance.
        similarity_threshold (float): Threshold for similarity to decide "Yes". Defaults to 0.8.
        tries (int): Number of retry attempts. Defaults to 3.
        timeout (float): Time in seconds between retries. Defaults to 5.0.

    Returns:
        tuple: (decision (str), similarity (float))
    """
    prompt = (
        f"Here is the correct diagnosis: {correct_diagnosis}\n"
        f"Here was the doctor's diagnosis: {diagnosis}\n"
        f"Rate the similarity between the two diagnoses on a scale of 0 to 1, where 0 means completely dissimilar and 1 means identical. "
        f"Based on the similarity score, decide whether they match. Respond strictly in the format:\n"
        f"[0.XX]\n"
        f"Do not include any additional text or explanation."
    )
    system_prompt = (
        "You are a medical moderator responsible for assessing similarity between two diagnoses. "
        "Respond strictly in the format [0.XX] where 0.XX is similarity. Do not include any extra text."
    )
    for attempt in range(tries):
        try:
            response = mod_pipe.query_model(prompt=prompt, system_prompt=system_prompt, tries=1, timeout=timeout)
            logger.debug("Attempt %d response: %s", attempt + 1, response)
            if response.startswith("[") and response.endswith("]"):
                similarity_str = response[1:-1]  # Remove square brackets
                similarity = float(similarity_str)
                if 0 <= similarity <= 1:
                    return similarity>=similarity_threshold
                else:
                    logger.warning("Invalid similarity score: %s", similarity)

            else:
                logger.warning("Response not in expected format: %s", response)

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(timeout)

    raise Exception("Failed to compare results after multiple attempts.")
