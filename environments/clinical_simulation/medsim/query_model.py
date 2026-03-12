import os
import re
import time
import json
import random
import logging
import requests
from collections import Counter
from tqdm import tqdm

import openai
import anthropic

from transformers import pipeline, AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Build the default vLLM server URL from environment variables
_openai_base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
if _openai_base:
    DEFAULT_SERVER_URL = (
        f"{_openai_base}/chat/completions"
        if _openai_base.endswith("/v1")
        else f"{_openai_base}/v1/chat/completions"
    )
else:
    _vllm_port = os.getenv("VLLM_PORT", "8001")
    DEFAULT_SERVER_URL = f"http://127.0.0.1:{_vllm_port}/v1/chat/completions"


class BAgent:
    """Inference agent that uses a vLLM server if available, else loads the model locally."""

    def __init__(self, model_name="Qwen3-VL-4B-Instruct", server_url=None):
        if server_url is None:
            server_url = DEFAULT_SERVER_URL
        self.server_url = server_url
        self.model_name = model_name
        self.use_server = self._check_server()
        if not self.use_server:
            self._load_model()
        else:
            logger.info(f"Using vLLM server at {self.server_url}")

    def _check_server(self):
        try:
            response = requests.get(
                self.server_url.replace("/v1/chat/completions", "/health"), timeout=2
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _load_model(self):
        logger.info("Loading model locally...")
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto",
                trust_remote_code=True,
            )
        except ValueError as e:
            if "Unknown quantization type" in str(e):
                logger.warning("Quantization not supported; loading without quantization.")
                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                if hasattr(config, "quantization_config"):
                    delattr(config, "quantization_config")
                model = AutoModel.from_pretrained(
                    self.model_name, config=config, device_map="auto", trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.pipeline = pipeline(
                    "text-generation", model=model, tokenizer=tokenizer,
                    device_map="auto", trust_remote_code=True,
                )
            else:
                raise

    def query_model(
        self, prompt, system_prompt="You are a helpful assistant.",
        tries=5, timeout=120, image_requested=False, scene=None,
        max_prompt_len=2500, clip_prompt=False, thread_id=1,
    ):
        if self.use_server:
            return self._query_server(prompt, system_prompt, tries, timeout)
        return self._query_local(
            prompt, system_prompt, image_requested, scene,
            max_prompt_len, clip_prompt, tries, timeout,
        )

    def _query_server(self, user_prompt, system_prompt, tries=10, timeout=20.0) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 512,
        }
        headers = {"Content-Type": "application/json"}
        for attempt in range(tries):
            try:
                response = requests.post(
                    self.server_url, headers=headers, json=payload, timeout=timeout
                )
                response.raise_for_status()
                time.sleep(2.0)
                content = response.json()["choices"][0]["message"]["content"] or ""
                # Strip <think>...</think> block from output — keep only the actual response
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return content
            except requests.RequestException as e:
                logger.warning(f"Server query attempt {attempt + 1} failed: {e}")
                time.sleep(timeout)
        logger.error("Max retries exceeded.")
        return "Error: Failed to fetch response from server."

    def _query_local(
        self, prompt, system_prompt, image_requested=False, scene=None,
        max_prompt_len=2500, clip_prompt=False, tries=3, timeout=5.0,
    ):
        for attempt in range(tries):
            try:
                if clip_prompt:
                    prompt = prompt[:max_prompt_len]
                if image_requested and scene is not None and hasattr(scene, "image_url"):
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": scene.image_url}},
                        ]},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                outputs = self.pipeline(
                    messages, max_new_tokens=200, do_sample=True,
                    temperature=0.7, top_p=0.9,
                )
                return outputs[0]["generated_text"][-1]["content"]
            except Exception as e:
                logger.warning(f"Local query attempt {attempt + 1} failed: {e}")
                time.sleep(timeout)
        logger.error("Max retries exceeded.")
        return "Error: Failed to generate response from local model."

    def query_model_with_ensembling(
        self, prompt, system_prompt, tries=3, timeout=5.0,
        image_requested=False, scene=None, max_prompt_len=2 ** 14,
        clip_prompt=False, thread_id=1, shuffle_ensemble_count=3,
    ):
        for attempt in range(tries):
            if clip_prompt:
                prompt = prompt[:max_prompt_len]
            try:
                responses = [
                    self._query_server(
                        self.shuffle_choices_in_prompt(prompt), system_prompt, tries, timeout
                    )
                    for _ in range(shuffle_ensemble_count)
                ]
                return self.aggregate_responses(responses)
            except Exception as e:
                logger.warning(f"Ensemble attempt {attempt + 1} failed: {e}")
                time.sleep(timeout)
        raise RuntimeError("Max retries exceeded.")

    def shuffle_choices_in_prompt(self, prompt):
        choices_pattern = r"\((a|b|c|d)\)\s+[^\n]+"
        choices = re.findall(choices_pattern, prompt)
        if choices:
            random.shuffle(choices)
            return re.sub(choices_pattern, lambda m: choices.pop(0), prompt, count=len(choices))
        return prompt

    def build_messages(self, system_prompt, prompt, image_requested, scene):
        if image_requested and scene is not None and hasattr(scene, "image_url"):
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": scene.image_url}},
                ]},
            ]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    def aggregate_responses(self, responses):
        counts = {r: responses.count(r) for r in responses}
        return max(counts, key=counts.get)


_fallback_agent = None


def get_fallback_agent():
    global _fallback_agent
    if _fallback_agent is None:
        model_name = os.getenv("SERVED_MODEL_NAME", os.getenv("MODEL_PATH", "Qwen3-VL-4B-Instruct"))
        _fallback_agent = BAgent(model_name=model_name)
    return _fallback_agent


def query_model(
    model_str: str,
    prompt: str,
    system_prompt: str,
    tries: int = 1,
    timeout: float = 30.0,
    image_requested: bool = False,
    scene=None,
    max_prompt_len: int = 2 ** 14,
    clip_prompt: bool = False,
):
    """Query a language model backend. Supported backends: gpt4, gpt4o, gpt-4o-mini,
    gpt3.5, o1-preview, claude3.5sonnet, or any OpenAI-compatible vLLM endpoint."""
    for _ in tqdm(range(tries), desc="Querying model"):
        if clip_prompt:
            prompt = prompt[:max_prompt_len]
        try:
            if image_requested:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"{scene.image_url}"}},
                    ]},
                ]
                model_map = {
                    "gpt4v": "gpt-4-vision-preview",
                    "gpt-4o-mini": "gpt-4o-mini",
                    "gpt4": "gpt-4-turbo",
                    "gpt4o": "gpt-4o",
                }
                if model_str in model_map:
                    response = openai.ChatCompletion.create(
                        model=model_map[model_str], messages=messages,
                        temperature=0.05, max_tokens=200,
                    )
                    return response["choices"][0]["message"]["content"]

            elif model_str in ("gpt4", "gpt4v", "gpt-4o-mini", "gpt4o", "gpt3.5"):
                model_map = {
                    "gpt4": "gpt-4-turbo-preview",
                    "gpt4v": "gpt-4-vision-preview",
                    "gpt-4o-mini": "gpt-4o-mini",
                    "gpt4o": "gpt-4o",
                    "gpt3.5": "gpt-3.5-turbo",
                }
                response = openai.ChatCompletion.create(
                    model=model_map[model_str],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.05, max_tokens=200,
                )
                return re.sub(r"\s+", " ", response["choices"][0]["message"]["content"])

            elif model_str == "o1-preview":
                response = openai.ChatCompletion.create(
                    model="o1-preview-2024-09-12",
                    messages=[{"role": "user", "content": system_prompt + prompt}],
                )
                return re.sub(r"\s+", " ", response["choices"][0]["message"]["content"])

            elif model_str == "claude3.5sonnet":
                client_anthropic = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client_anthropic.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt, max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                return json.loads(message.to_json())["content"][0]["text"]

            else:
                agent = get_fallback_agent()
                return agent.query_model(prompt, system_prompt)

        except Exception:
            time.sleep(timeout)


def extract_bracket_content(s: str):
    if "[" in s or "]" in s:
        return "[" + s.split("[", 1)[-1].split("]")[0] + "]"
    return s


def clean_diagnosis(message):
    """Parse a list of diagnoses from a model-generated string."""
    try:
        cleaned = (
            message.replace("'", '"')
                   .replace("```python", "")
                   .replace("```", "")
                   .replace("\n", "")
                   .replace(", ", ",")
                   .strip()
        )
        cleaned = extract_bracket_content(cleaned)
        if not cleaned.startswith("["):
            cleaned = "[" + cleaned
        if not cleaned.endswith("]"):
            cleaned = cleaned + "]"
        diagnosis_list = json.loads(cleaned)
        if isinstance(diagnosis_list, list):
            return [str(d).strip() for d in diagnosis_list]
    except Exception:
        pass

    try:
        matches = re.findall(r'["\'](.*?)["\']', message)
        matches = [m for m in matches if len(m) > 3 and m not in ("response", "text", "success")]
        if len(matches) >= 3:
            return matches[:3]
    except Exception:
        pass

    return ["Other condition A", "Other condition B", "Other condition C"]


def generate_possible_diagnoses(question: str, answer: str, backend: str):
    """Generate three plausible alternative diagnoses for a clinical question."""
    prompt = (
        f"Given the following medical question, suggest three possible diagnoses in a format "
        f"similar to the correct diagnosis. Ensure the diagnoses are unique, medically plausible, "
        f"and formatted to be indistinguishable from the correct answer. "
        f"Do NOT suggest the correct diagnosis.\n\n"
        f"Question: {question}\n"
        f"Correct Diagnosis: {answer}\n\n"
        f"Provide the diagnoses in a Python list format."
    )
    system_prompt = (
        "You are a highly knowledgeable AI assistant specializing in medical reasoning. "
        "Generate accurate, evidence-based differential diagnoses."
    )
    response = query_model(backend, prompt, system_prompt)
    return clean_diagnosis(response)


def generate_answer_choices(correct_answer, answer_list):
    letter_answers = ["A", "B", "C", "D"]
    if not isinstance(answer_list, list):
        answer_list = []
    answer_list = [x for x in answer_list if x != correct_answer][:3]
    while len(answer_list) < 3:
        answer_list.append(f"Other Possibility {len(answer_list) + 1}")
    answer_list.append(correct_answer)
    random.shuffle(answer_list)
    result = dict(zip(letter_answers, answer_list))
    try:
        answer_letter = letter_answers[answer_list.index(correct_answer)]
    except (ValueError, IndexError):
        answer_letter = "A"
        result["A"] = correct_answer
    return result, answer_letter


def extract_question(patient_statement, agent_hist, parsed_responses, backend):
    prompt = (
        f"Given the following patient statement, additional context, and doctor discussion, "
        f"generate a structured diagnostic question that extracts key insights and relevant test results:\n\n"
        f"Patient Statement:\n{patient_statement}\n\n"
        f"Additional Context:\n{agent_hist}\n\n"
        f"Doctor Discussion:\n{parsed_responses}\n\n"
        f"Ensure the question follows this format: "
        f"Provide the most likely final diagnosis for the following patient. A ___ year old [man/woman] "
        f"presents with [duration] of [symptom description], associated with [other symptoms]..."
    )
    system_prompt = (
        "You are a highly knowledgeable AI assistant specializing in medical reasoning. "
        "Analyze clinical scenarios and provide accurate differential diagnoses."
    )
    return query_model(backend, prompt, system_prompt)
