import os
import json
import random
import base64
import io
from tqdm import tqdm
from prettytable import PrettyTable
from termcolor import cprint
from pptree import Node, print_tree
import google.generativeai as genai
from openai import OpenAI

from PIL import Image
import uuid
import time


# ─── Consecutive failure tracking ───
_consecutive_failures = 0
_max_consecutive_failures = 3

class APIResourceExhausted(Exception):
    """Raised when API quota is exhausted or resources are unavailable (HTTP 429)."""
    pass

def reset_failure_counter():
    """Reset the consecutive failure counter (call after a successful API call)."""
    global _consecutive_failures
    _consecutive_failures = 0

def increment_failure_counter():
    """Increment the consecutive failure counter; raise SystemExit when the threshold is reached."""
    global _consecutive_failures
    _consecutive_failures += 1
    print(f"[WARN] Consecutive API failures: {_consecutive_failures}/{_max_consecutive_failures}")
    if _consecutive_failures >= _max_consecutive_failures:
        raise SystemExit(f"[FATAL] Stopped after {_consecutive_failures} consecutive API failures. Check API keys, quotas, and service status.")


def encode_image(image_path, max_size=(512, 512)):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.thumbnail(max_size)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _supports_vision(model_name):
    """Check if a model supports vision/multimodal input based on its name."""
    lower = model_name.lower()
    # gpt-4o/gpt-4-vision, Qwen-VL/Qwen3-VL, gemini (all multimodal), llava, etc.
    return any(kw in lower for kw in ['gpt-4', 'vl', 'vision', 'gemini', 'llava'])


# ─── Trace system: collect all Agent conversations per sample ───
_active_agents = []

def reset_trace():
    global _active_agents
    _active_agents = []

def _clean_message(msg):
    """Strip base64 image data from a message for trace saving."""
    content = msg.get('content')
    if isinstance(content, list):
        cleaned_parts = []
        for part in content:
            if isinstance(part, dict) and part.get('type') == 'image_url':
                cleaned_parts.append({'type': 'image_url', 'note': 'base64 image data omitted'})
            else:
                cleaned_parts.append(part)
        return {'role': msg['role'], 'content': cleaned_parts}
    return msg

def get_trace():
    """Return trace data for all agents created since last reset."""
    trace = []
    for agent in _active_agents:
        trace.append({
            'agent_role': agent.role,
            'model': agent.model_info,
            'instruction': agent.instruction,
            'messages': [_clean_message(m) for m in agent.messages]
        })
    return trace


class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-4o-mini', img_path=None,
                 use_think_format=False):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        self._seen_images = set()

        # Always maintain messages list for trace
        self.messages = [{"role": "system", "content": instruction}]

        if 'gemini' in self.model_info:
            self.model = genai.GenerativeModel(
                model_name=self.model_info,
                system_instruction=instruction
            )
            history = []
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    if use_think_format:
                        # SFT model: reason already contains <think>...</think>\n(X) answer
                        assistant_content = exampler['reason']
                    else:
                        # Gemini: original Answer: prefix + reasoning format
                        assistant_content = exampler['answer'] + "\n\n" + exampler['reason']
                    self.messages.append({"role": "assistant", "content": assistant_content})
                    history.append({"role": "user", "parts": [exampler['question']]})
                    history.append({"role": "model", "parts": [assistant_content]})
            self._chat = self.model.start_chat(history=history)
        else:
            # OpenAI-compatible (GPT, vLLM/Qwen, etc.)
            self.client = OpenAI(api_key=os.environ['openai_api_key'], timeout=300, max_retries=2)
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    if use_think_format:
                        assistant_content = exampler['reason']
                    else:
                        assistant_content = exampler['answer'] + "\n\n" + exampler['reason']
                    self.messages.append({"role": "assistant", "content": assistant_content})

        _active_agents.append(self)

    def chat(self, message, img_path=None, chat_mode=True, max_tokens=2048):
        if 'gemini' in self.model_info:
            # Build content parts for Gemini
            content_parts = [message]
            if img_path and img_path not in self._seen_images:
                self._seen_images.add(img_path)
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.thumbnail((512, 512))
                content_parts.append(img)

            # Record in messages for trace (text only, no PIL objects)
            self.messages.append({"role": "user", "content": message, "img_path": img_path})

            for attempt in range(8):
                try:
                    response = self._chat.send_message(
                        content_parts,
                        generation_config={'temperature': 1.2, 'max_output_tokens': max_tokens}
                    )
                    content = response.text
                    # Log truncation warning
                    if hasattr(response, 'candidates') and response.candidates:
                        fr = response.candidates[0].finish_reason
                        if fr and str(fr) not in ('STOP', '1', 'FinishReason.STOP'):
                            print(f"[WARN] Gemini finish_reason={fr} (may be truncated), max_output_tokens={max_tokens}")
                    self.messages.append({"role": "assistant", "content": content})
                    reset_failure_counter()
                    return content
                except Exception as e:
                    error_str = str(e).lower()
                    if '429' in error_str or 'resource exhausted' in error_str or 'quota' in error_str:
                        print(f"[FATAL] Gemini API resource exhausted (429): {e}")
                        increment_failure_counter()
                        raise APIResourceExhausted(f"Gemini API quota exhausted: {e}")

                    if attempt < 7:
                        wait = min(2 ** attempt, 30)
                        print(f"[WARN] Gemini API error (attempt {attempt+1}/8): {e}. Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        import traceback
                        traceback.print_exc()
                        increment_failure_counter()
                        return f"Error in Gemini API after 8 retries: {str(e)}"

        else:
            # OpenAI-compatible path (GPT, vLLM/Qwen, etc.)
            user_content = [{"type": "text", "text": message}]

            attach_image = False
            if img_path and _supports_vision(self.model_info):
                if img_path not in self._seen_images:
                    attach_image = True
                    self._seen_images.add(img_path)

            if attach_image:
                base64_image = encode_image(img_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
                self.messages.append({"role": "user", "content": user_content})
            else:
                self.messages.append({"role": "user", "content": message})

            for attempt in range(8):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_info,
                        messages=self.messages,
                        temperature=1.2,
                        max_tokens=max_tokens,
                    )
                    content = response.choices[0].message.content
                    self.messages.append({"role": "assistant", "content": content})
                    reset_failure_counter()
                    return content
                except Exception as e:
                    error_str = str(e).lower()
                    if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                        print(f"[FATAL] OpenAI API resource exhausted (429): {e}")
                        increment_failure_counter()
                        raise APIResourceExhausted(f"OpenAI API quota exhausted: {e}")

                    if attempt < 7:
                        wait = min(2 ** attempt, 30)
                        print(f"[WARN] OpenAI API error (attempt {attempt+1}/8): {e}. Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        import traceback
                        traceback.print_exc()
                        increment_failure_counter()
                        return f"Error in LLM API after 8 retries: {str(e)}"

    def temp_responses(self, message, max_tokens=2048, img_path=None):
        if 'gemini' in self.model_info:
            content_parts = [message]
            if img_path and img_path not in self._seen_images:
                self._seen_images.add(img_path)
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.thumbnail((512, 512))
                content_parts.append(img)

            self.messages.append({"role": "user", "content": message, "img_path": img_path})

            for attempt in range(8):
                try:
                    response = self._chat.send_message(
                        content_parts,
                        generation_config={'temperature': 0.0, 'max_output_tokens': max_tokens}
                    )
                    content = response.text
                    # Log truncation warning
                    if hasattr(response, 'candidates') and response.candidates:
                        fr = response.candidates[0].finish_reason
                        if fr and str(fr) not in ('STOP', '1', 'FinishReason.STOP'):
                            print(f"[WARN] Gemini finish_reason={fr} (may be truncated), max_output_tokens={max_tokens}")
                    self.messages.append({"role": "assistant", "content": content})
                    reset_failure_counter()
                    return {0.0: content}
                except Exception as e:
                    error_str = str(e).lower()
                    if '429' in error_str or 'resource exhausted' in error_str or 'quota' in error_str:
                        print(f"[FATAL] Gemini API resource exhausted (429): {e}")
                        increment_failure_counter()
                        raise APIResourceExhausted(f"Gemini API quota exhausted: {e}")

                    if attempt < 7:
                        wait = min(2 ** attempt, 30)
                        print(f"[WARN] Gemini API error (attempt {attempt+1}/8): {e}. Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        import traceback
                        traceback.print_exc()
                        increment_failure_counter()
                        return {0.0: f"Error in Gemini API after 8 retries: {str(e)}"}

        else:
            # OpenAI-compatible path
            user_content = [{"type": "text", "text": message}]

            attach_image = False
            if img_path and _supports_vision(self.model_info):
                if img_path not in self._seen_images:
                    attach_image = True
                    self._seen_images.add(img_path)

            if attach_image:
                base64_image = encode_image(img_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
                self.messages.append({"role": "user", "content": user_content})
            else:
                self.messages.append({"role": "user", "content": message})

            temperatures = [0.0]
            responses = {}
            for temperature in temperatures:
                for attempt in range(8):
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_info,
                            messages=self.messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            frequency_penalty=0.5,
                        )
                        content = response.choices[0].message.content
                        self.messages.append({"role": "assistant", "content": content})
                        responses[temperature] = content
                        reset_failure_counter()
                        break
                    except Exception as e:
                        error_str = str(e).lower()
                        if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                            print(f"[FATAL] OpenAI API resource exhausted (429): {e}")
                            increment_failure_counter()
                            raise APIResourceExhausted(f"OpenAI API quota exhausted: {e}")

                        if attempt < 7:
                            wait = min(2 ** attempt, 30)
                            print(f"[WARN] OpenAI API error (attempt {attempt+1}/8): {e}. Retrying in {wait}s...")
                            time.sleep(wait)
                        else:
                            import traceback
                            traceback.print_exc()
                            increment_failure_counter()
                            responses[temperature] = f"Error in LLM API after 8 retries: {str(e)}"

            return responses

class Group:
    def __init__(self, goal, members, question, examplers=None, model_info='gpt-4o-mini', dataset='medqa'):
        self.goal = goal
        self.members = []
        self.dataset = dataset
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info=model_info)
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        if not self.members:
            return "Error: empty group members"

        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                if not assist_members:
                    lead_member = self.members[0]
                else:
                    lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {}".format(self.question)
            delivery = lead_member.chat(delivery_prompt, img_path=img_path)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery), img_path=img_path)
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])
            
            if self.dataset in ['pathvqa', 'mimic-cxr-vqa']:
                suffix_prompt = "return your concise answer to the medical query based on the visual evidence."
            elif self.dataset == 'medqa':
                suffix_prompt = "return your answer to the medical query among the option provided."
            elif self.dataset == 'pubmedqa':
                suffix_prompt = "return your concise answer to the medical query based on the context provided."

            if self.examplers is not None:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, after reviewing the following example cases, {suffix_prompt}\n\n{self.examplers}\nQuestion: {self.question}"""
            else:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, {suffix_prompt}\n\nQuestion: {self.question}"""

            response = lead_member.chat(investigation_prompt, img_path=img_path)

            return response

        elif comm_type == 'external':
            return

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        if count >= len(emojis):
            break
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()

        if hierarchy is None:
            hierarchy = 'Independent'

        if 'independent' not in hierarchy.lower() and '>' in hierarchy:
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)

        count += 1

    return agents

def parse_group_info(group_info):
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    goal_parts = lines[0].split('-', 1)
    parsed_info['group_goal'] = goal_parts[1].strip() if len(goal_parts) > 1 else lines[0].strip()

    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':', 1)
            member_role_description = member_info[1].split('-', 1)

            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info

def setup_model(model_name):
    if 'gemini' in model_name:
        genai.configure(api_key=os.environ['genai_api_key'])
        return genai, None
    else:
        # OpenAI-compatible (GPT, vLLM/Qwen, etc.)
        client = OpenAI(api_key=os.environ['openai_api_key'], timeout=300, max_retries=2)
        return None, client

def _load_mimic_items(raw_data, base_img_dir, limit=None):
    """Load valid samples from MIMIC-CXR-VQA raw data, filtering empty answers and missing images."""
    items = []
    for item in raw_data:
        ans = item['answer']
        if isinstance(ans, list) and len(ans) > 0:
            ans = ans[0]
        ans_str = str(ans).lower().strip()
        full_img_path = os.path.join(base_img_dir, item['image_path'])
        if ans_str not in ["[]", ""] and os.path.exists(full_img_path):
            items.append({
                'id': str(item['idx']),
                'question': item['question'],
                'answer': ans_str,
                'img_path': full_img_path
            })
        if limit and len(items) >= limit:
            break
    return items


def load_data(dataset, split='test'):
    main_data = []
    examplers = []

    if dataset == 'medqa':
        base_dir = '${MEDQA_ROOT}/data_clean/questions/US'
        test_path = os.path.join(base_dir, 'test.jsonl')
        train_path = os.path.join(base_dir, 'train.jsonl')

        if split == 'test':
            with open(test_path, 'r') as file:
                for line in file:
                    main_data.append(json.loads(line))
            with open(train_path, 'r') as file:
                for line in file:
                    examplers.append(json.loads(line))
        else:
            with open(train_path, 'r') as file:
                for idx, line in enumerate(file):
                    sample = json.loads(line)
                    sample['id'] = f"medqa_train_{idx}"
                    main_data.append(sample)
            with open(test_path, 'r') as file:
                for line in file:
                    examplers.append(json.loads(line))

    elif dataset == 'pubmedqa':
        test_path = '${PUBMEDQA_ROOT}/data/test_set.json'
        train_path = '${PUBMEDQA_ROOT}/data/pqal_fold0/train_set.json'

        if split == 'test':
            with open(test_path, 'r') as f:
                data = json.load(f)
                for pid, item in data.items():
                    item['id'] = pid
                    main_data.append(item)
            with open(train_path, 'r') as f:
                data = json.load(f)
                for pid, item in data.items():
                    item['id'] = pid
                    examplers.append(item)
        else:
            with open(train_path, 'r') as f:
                data = json.load(f)
                for pid, item in data.items():
                    item['id'] = pid
                    main_data.append(item)
            with open(test_path, 'r') as f:
                data = json.load(f)
                for pid, item in data.items():
                    item['id'] = pid
                    examplers.append(item)

    elif dataset == 'pathvqa':
        base_dir = '${PATHVQA_ROOT}/data'
        test_path = os.path.join(base_dir, 'pathvqa_test_yesno.json')
        train_path = os.path.join(base_dir, 'pathvqa_train.json')

        if split == 'test':
            with open(test_path, 'r') as f:
                raw_test = json.load(f)
            for idx, item in enumerate(raw_test):
                main_data.append({
                    'id': f"pathvqa_test_{idx}",
                    'question': item['question'],
                    'answer': item['label'],
                    'img_path': item['image_path']
                })
            with open(train_path, 'r') as f:
                raw_train = json.load(f)
            for idx, item in enumerate(raw_train):
                examplers.append({
                    'id': f"pathvqa_train_{idx}",
                    'question': item['question'],
                    'answer': item['answer'],
                    'img_path': item['image_path']
                })
        else:
            with open(train_path, 'r') as f:
                raw_train = json.load(f)
            for idx, item in enumerate(raw_train):
                main_data.append({
                    'id': f"pathvqa_train_{idx}",
                    'question': item['question'],
                    'answer': item['answer'],
                    'img_path': item['image_path']
                })
            with open(test_path, 'r') as f:
                raw_test = json.load(f)
            for idx, item in enumerate(raw_test):
                examplers.append({
                    'id': f"pathvqa_test_{idx}",
                    'question': item['question'],
                    'answer': item['label'],
                    'img_path': item['image_path']
                })

        print(f"[INFO] Loaded {len(main_data)} PathVQA {split} samples, {len(examplers)} examplers")

    elif dataset == 'mimic-cxr-vqa':
        base_img_dir = '${DATA_ROOT}/mimic-cxr-jpg/2.1.0/files'
        base_json_dir = '${DATA_ROOT}/mimic-ext-mimic-cxr-vqa/1.0.0/MIMIC-Ext-MIMIC-CXR-VQA/dataset'
        test_path = os.path.join(base_json_dir, 'test.json')
        train_path = os.path.join(base_json_dir, 'train.json')

        if split == 'test':
            print(f"[INFO] Loading MIMIC-CXR-VQA test set from {test_path}...")
            with open(test_path, 'r') as f:
                raw_test = json.load(f)
            main_data = _load_mimic_items(raw_test, base_img_dir)
            print(f"[INFO] Loaded {len(main_data)} valid samples for testing.")

            print(f"[INFO] Loading MIMIC-CXR-VQA train set for exemplars...")
            try:
                with open(train_path, 'r') as f:
                    raw_train = json.load(f)
                examplers = _load_mimic_items(raw_train, base_img_dir, limit=5)
                for ex in examplers:
                    ex['reason'] = 'Based on the chest X-ray findings.'
            except Exception as e:
                print(f"[WARN] Failed to load train set for exemplars: {e}")
        else:
            print(f"[INFO] Loading MIMIC-CXR-VQA train set from {train_path}...")
            with open(train_path, 'r') as f:
                raw_train = json.load(f)
            main_data = _load_mimic_items(raw_train, base_img_dir)
            print(f"[INFO] Loaded {len(main_data)} valid train samples.")

            print(f"[INFO] Loading MIMIC-CXR-VQA test set for exemplars...")
            try:
                with open(test_path, 'r') as f:
                    raw_test = json.load(f)
                examplers = _load_mimic_items(raw_test, base_img_dir, limit=5)
                for ex in examplers:
                    ex['reason'] = 'Based on the chest X-ray findings.'
            except Exception as e:
                print(f"[WARN] Failed to load test set for exemplars: {e}")

    print(f"[INFO] Dataset={dataset}, Split={split}: {len(main_data)} samples, {len(examplers)} examplers")
    return main_data, examplers


# ─── Advanced evaluation helpers for VQA datasets ───
# Ported from OpenThinkIMG evaluation tasks for better handling of
# synonyms, medical abbreviations, and open-ended answers.

import re
import unicodedata

try:
    from thefuzz import fuzz as _fuzz
except ImportError:
    _fuzz = None


def _norm_text(s):
    """Unicode-normalize, lowercase, collapse whitespace."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_yn(s):
    """Normalize to 'yes'/'no' or return ''."""
    s = _norm_text(s)
    if re.search(r"\byes\b", s):
        return "yes"
    if re.search(r"\bno\b", s):
        return "no"
    if re.search(r"\btrue\b", s):
        return "yes"
    if re.search(r"\bfalse\b", s):
        return "no"
    if s in ("y",):
        return "yes"
    if s in ("n",):
        return "no"
    return ""


def _strip_explanations(s):
    """Strip verbose model output down to the core answer token."""
    s = _norm_text(s)
    if not s:
        return ""
    s = re.sub(r"\([^)]*\)", "", s).strip()
    s = re.split(r"[.;:\n\r]|\\n", s, maxsplit=1)[0].strip()
    s = re.split(r"\bis\b|\bare\b", s, maxsplit=1)[0].strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _simple_tokenize(s):
    """Tokenize + crude singularization for token-level matching."""
    s = _norm_text(s)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    toks = s.split(" ")
    norm_toks = []
    for t in toks:
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            norm_toks.append(t[:-1])
        else:
            norm_toks.append(t)
    return [t for t in norm_toks if t]


def _token_subset_match(pred, gold):
    """True if >=60% of pred tokens appear in gold tokens (or vice versa)."""
    pt = set(_simple_tokenize(pred))
    gt = set(_simple_tokenize(gold))
    if not pt or not gt:
        return False
    inter = pt & gt
    if not inter:
        return False
    return len(inter) / max(len(pt), 1) >= 0.6


def _check_correct_pathvqa(response_text, label):
    """
    PathVQA evaluation (ported from OpenThinkIMG pathvqa/task.py):
      - yes/no subset: exact match after normalization
      - open-ended: exact → containment → token-subset → fuzzy (threshold=70)
    """
    gold_raw = str(label)
    pred_raw = str(response_text)

    # yes/no branch
    gold_yn = _norm_yn(gold_raw)
    if gold_yn in ("yes", "no"):
        pred_yn = _norm_yn(pred_raw)
        return pred_yn == gold_yn

    # open-ended branch
    gold_n = _norm_text(gold_raw)
    pred_n = _norm_text(pred_raw)
    pred_c = _strip_explanations(pred_raw)
    gold_c = _strip_explanations(gold_raw)

    # 1) normalized exact match
    if gold_n and pred_n == gold_n:
        return True
    if gold_c and pred_c and pred_c == gold_c:
        return True

    # 2) containment (either direction, both raw-normalized and cleaned)
    if gold_n and pred_n:
        if gold_n in pred_n or pred_n in gold_n:
            return True
    if gold_c and pred_c:
        if gold_c in pred_c or pred_c in gold_c:
            return True

    # 3) token subset match
    if _token_subset_match(pred_n, gold_n) or _token_subset_match(pred_c, gold_n):
        return True

    # 4) fuzzy match (requires thefuzz)
    if _fuzz is not None and gold_n and pred_n:
        r1 = _fuzz.partial_ratio(pred_n, gold_n)
        r2 = _fuzz.token_set_ratio(pred_n, gold_n)
        r3 = _fuzz.ratio(pred_n, gold_n)
        best = max(r1, r2, r3)
        if pred_c and pred_c != pred_n:
            r1c = _fuzz.partial_ratio(pred_c, gold_n)
            r2c = _fuzz.token_set_ratio(pred_c, gold_n)
            r3c = _fuzz.ratio(pred_c, gold_n)
            best = max(best, r1c, r2c, r3c)
        if best >= 70:
            return True

    return False


# ─── MIMIC-CXR-VQA evaluation helpers ───

_MIMIC_PUNCT_RE = re.compile(r"[^a-z0-9\s]")

_MIMIC_SYNONYMS = {
    "picc line": "picc",
    "peripherally inserted central catheter": "picc",
    "endotracheal tube": "ett",
    "enteric tube": "ng tube",
    "nasogastric tube": "ng tube",
    "orogastric tube": "og tube",
    "cardiac pacemaker": "pacemaker",
    "cardiac pacer": "pacemaker",
    "pacer": "pacemaker",
    "wires": "wire",
    "leads": "wire",
    "elevation": "elevated",
    "opacities": "opacity",
    "effusions": "effusion",
    "abnormalities": "abnormality",
    "structures": "structure",
}

_MIMIC_FLUFF = [
    "present", "seen", "visible", "identified", "noted", "evidence of",
    "demonstrated", "observed", "cannot determine", "suspicious for",
    "consistent with", "suggestive of", "findings of",
]


def _mimic_normalize(s, remove_fluff=False):
    """Normalize text with medical synonym replacement (MIMIC-CXR-VQA)."""
    s = (s or "").lower().strip()
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    for k, v in _MIMIC_SYNONYMS.items():
        if k in s:
            s = s.replace(k, v)
    s = _MIMIC_PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if remove_fluff:
        for w in _MIMIC_FLUFF:
            s = s.replace(w, " ")
        s = re.sub(r"\s+", " ", s).strip()
    return s


def _mimic_normalize_yesno(text):
    """Normalize to 'yes'/'no' or None (MIMIC-CXR-VQA)."""
    s = _mimic_normalize(text, remove_fluff=True)
    if s in ("yes", "no"):
        return s
    if re.search(r"\byes\b", s):
        return "yes"
    if re.search(r"\bno\b", s):
        return "no"
    return None


def _mimic_is_negative_finding(text):
    """Check if text represents a negative/normal finding."""
    text = (text or "").lower()
    if text.strip() in ("no", "none", "normal", "negative"):
        return True
    clean = re.sub(r"[^a-z]", "", text)
    if clean in ("none", "non", "nothing", "normal"):
        return True
    for sn in ("no acute", "no active", "no significant", "lungs are clear", "heart is normal"):
        if sn in text:
            return True
    return False


def _mimic_parse_terms(s):
    """Split answer into individual terms for multi-term matching."""
    s = _mimic_normalize(s, remove_fluff=True)
    if not s:
        return set()
    parts = re.split(r"\s*(?:,|;|\band\b|\bor\b|\.|\n)\s*", s)
    return {p.strip() for p in parts if p.strip()}


def _mimic_soft_match(gt_term, pred_term):
    """Soft matching with token overlap for medical terms."""
    gt_term = (gt_term or "").strip()
    pred_term = (pred_term or "").strip()
    if not gt_term or not pred_term:
        return False
    gt_words = set(gt_term.split())
    pred_words = set(pred_term.split())
    if gt_term in pred_term:
        return True
    if gt_words.issubset(pred_words):
        return True
    intersection = gt_words.intersection(pred_words)
    valid_overlap = [w for w in intersection if len(w) > 2 and w not in ("and", "with", "the")]
    if len(valid_overlap) >= 2:
        return True
    if gt_words and len(valid_overlap) / len(gt_words) >= 0.5:
        return True
    if len(gt_words) == 1 and list(gt_words)[0] in pred_words:
        return True
    return False


def _mimic_extract_final_answer(text):
    """Extract the final answer from potentially verbose model output."""
    if not text:
        return ""
    t = str(text).strip()
    if "Final Answer:" in t:
        try:
            t = t.split("Final Answer:", 1)[1].strip()
        except Exception:
            pass
    for pat in (
        r"\[FINAL\]\s*(.+?)\s*$",
        r"\*\*Final Answer:?\*\*\s*(.+?)\s*$",
        r"Final Answer:\s*(.+?)\s*$",
        r"\[DECISION\]\s*(.+?)\s*$",
    ):
        m = re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if m:
            return m.group(1).strip()
    if t.endswith("."):
        t = t[:-1]
    return t


def _check_correct_mimic(response_text, label):
    """
    MIMIC-CXR-VQA evaluation (ported from OpenThinkIMG mimic_cxr_vqa/task.py):
      - Medical synonym normalization (PICC, ETT, ng tube, etc.)
      - Negative finding detection (none/normal/no acute/...)
      - yes/no normalization
      - Open-ended: containment → soft token matching
    """
    pred_raw = str(response_text)
    pred_final = _mimic_extract_final_answer(pred_raw)
    pred_norm = _mimic_normalize(pred_final, remove_fluff=True)
    pred_terms = _mimic_parse_terms(pred_final)

    # Build gt_list from label (could be list or string)
    gt_list = []
    if isinstance(label, list):
        for x in label:
            s = _mimic_normalize(str(x) if x is not None else "")
            if s:
                gt_list.append(s)
    else:
        s = _mimic_normalize(str(label) if label is not None else "")
        if s:
            gt_list.append(s)

    if not gt_list:
        gt_list = ["none"]

    gt_set = set(gt_list)

    # none / negative finding
    if gt_set == {"none"}:
        if _mimic_is_negative_finding(pred_norm):
            return True
        if _mimic_is_negative_finding(pred_final):
            return True
        if not pred_norm:
            return True
        return False

    # yes/no branch
    if gt_set.issubset({"yes", "no"}):
        p_yn = _mimic_normalize_yesno(pred_final)
        if p_yn:
            return p_yn in gt_set
        return False

    # If prediction is negative but gt is not "none", it's wrong
    if _mimic_is_negative_finding(pred_norm) and gt_set != {"none"}:
        return False

    # Open-ended: check if any gt term matches
    for g in gt_set:
        # direct containment
        if g in pred_norm:
            return True
        # soft match against parsed prediction terms
        for p_term in pred_terms:
            if _mimic_soft_match(g, p_term):
                return True

    return False


def _format_instruction(dataset, context='basic', use_think_format=False):
    """Return format instruction string based on dataset, context, and format mode.

    Args:
        dataset: 'medqa', 'pubmedqa', 'pathvqa', 'mimic-cxr-vqa'
        context: 'basic', 'expert', 'final', 'synthesis', 'moderator_medqa',
                 'moderator_pubmedqa', 'moderator_vqa'
        use_think_format: True for SFT-trained models, False for Gemini
    """
    if dataset in ['pathvqa', 'mimic-cxr-vqa']:
        if use_think_format:
            return ("Provide your response in the following format:\n"
                    "<think>your step-by-step analysis of the image and question</think>\n"
                    "your brief answer in 1-5 words only")
        return ("Provide your response in the following format:\n"
                "Thought: <your step-by-step analysis of the image and question>\n"
                "Answer: <your brief answer in 1-5 words only>")
    # medqa, pubmedqa
    if context == 'expert':
        if use_think_format:
            return ("Provide your response in the following format:\n"
                    "<think>your expert reasoning in 2-3 key points</think>\n"
                    "your concise answer")
        return ("Provide your response in the following format:\n"
                "Thought: <your expert reasoning in 2-3 key points>\n"
                "Answer: <your concise answer>")
    if context == 'final':
        if use_think_format:
            return ("Provide your response in the following format:\n"
                    "<think>brief reasoning in 2-3 points</think>\n"
                    "your answer")
        return ("Provide your response in the following format:\n"
                "Thought: <brief reasoning in 2-3 points>\n"
                "Answer: <your answer>")
    # 'basic' default
    if use_think_format:
        return ("Provide your response in the following format:\n"
                "<think>your reasoning in 3-5 sentences. Be concise — do not discuss every option.</think>\n"
                "the correct option, e.g. (C) Primidone")
    return ("Provide your response in the following format:\n"
            "Thought: <your reasoning in 3-5 sentences. Be concise — do not discuss every option.>\n"
            "Answer: <the correct option, e.g. (C) Primidone>")


def _extract_answer_from_response(text):
    """Extract the answer portion from model output, supporting both <think>...</think> and Thought:/Answer: formats."""
    m = re.search(r'</think>\s*(.+)', text, re.DOTALL)
    if m:
        answer = m.group(1).strip().split('\n')[0].strip()
        if answer:
            return answer
    m = re.search(r'\bAnswer:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    if m:
        answer = m.group(1).strip()
        answer = answer.split('\n')[0].strip()
        return answer
    return text


def _check_correct(result, dataset):
    """Check if the model's answer is correct using dataset-specific evaluation logic."""
    response = result.get('response', '')
    if isinstance(response, dict):
        response = str(list(response.values())[0])

    response = _extract_answer_from_response(response)
    label = result.get('label', '')

    if not label and not isinstance(label, list):
        return True

    if dataset == 'medqa':
        return f"({str(label).lower().strip()})" in response.lower().strip()
    elif dataset == 'pubmedqa':
        return str(label).lower().strip() in response.lower().strip()
    elif dataset == 'pathvqa':
        return _check_correct_pathvqa(response, label)
    elif dataset == 'mimic-cxr-vqa':
        return _check_correct_mimic(response, label)
    return True


def filter_wrong_samples(data, results_dir, dataset):
    """Load all chunk results from the results directory, find incorrect sample IDs, and filter the main dataset."""
    import glob

    all_results = []
    json_files = sorted(glob.glob(os.path.join(results_dir, '*.json')))
    for jf in json_files:
        if '_traces' in jf:
            continue
        try:
            with open(jf, 'r') as f:
                results = json.load(f)
            all_results.extend(results)
        except Exception as e:
            print(f"[WARN] Failed to load {jf}: {e}")

    print(f"[INFO] Loaded {len(all_results)} total results from {results_dir}")

    wrong_ids = set()
    correct_count = 0
    for result in all_results:
        if _check_correct(result, dataset):
            correct_count += 1
        else:
            wrong_ids.add(str(result['id']))

    print(f"[INFO] Correctness: {correct_count}/{len(all_results)} ({100*correct_count/max(len(all_results),1):.1f}%), {len(wrong_ids)} wrong")

    filtered = [s for s in data if str(s.get('id', '')) in wrong_ids]
    print(f"[INFO] Filtered to {len(filtered)} wrong samples for re-inference")
    return filtered

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question, None
    
    elif dataset == 'pubmedqa':
        context_str = " ".join(sample['CONTEXTS'])
        question = f"Context: {context_str}\n\nQuestion: {sample['QUESTION']}\n\nAnswer (yes/no/maybe):"
        return question, None
    
    elif dataset == 'mimic-cxr-vqa':
        question = f"Question: {sample['question']}\n\nPlease provide a short answer based on the chest X-ray image."
        return question, sample['img_path']
    
    elif dataset == 'pathvqa':
        question = (f"Question: {sample['question']}\n\n"
                    f"Please provide a very brief answer (1-5 words). "
                    f"If the question asks about an organ, body system, or tissue type, "
                    f"answer with just the organ/system/tissue name. "
                    f"Do not explain or give a full diagnosis.")
        return question, sample['img_path']

    return sample['question'], None

def determine_difficulty(question, difficulty, model_name, img_path=None):
    if difficulty != 'adaptive':
        return difficulty
    
    difficulty_prompt = f"""Now, given the medical query (and potentially an image), you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer based on the visual and text info.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
    
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query based on text and image.', role='medical expert', model_info=model_name)

    response = medical_agent.chat(difficulty_prompt, img_path=img_path)

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic'
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate'
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced'
    else:
        return 'basic'


def build_exampler_cache(examplers, model, dataset, cache_dir, num_examplers=3,
                         use_think_format=False):
    """
    Pre-generate reasoning for few-shot exemplars and cache to disk.
    Avoids re-generating exemplar reasoning for every test sample.

    When use_think_format=True, answer field uses bare answer (no 'Answer:' prefix),
    and cache file name includes '_think' tag to avoid mixing with Gemini caches.

    Returns: list of dicts {question, answer, reason}, or None for vision datasets.
    """
    if dataset not in ['pubmedqa', 'medqa']:
        return None  # Vision datasets don't use text exemplars

    model_safe = model.replace('/', '_').replace('\\', '_')
    fmt_tag = "_think" if use_think_format else ""
    cache_path = os.path.join(cache_dir, f"{dataset}_{model_safe}{fmt_tag}_exampler_cache.json")

    # Try loading existing cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if len(cached) >= num_examplers:
                print(f"[INFO] Loaded {len(cached)} cached exampler reasonings from {cache_path}")
                return cached[:num_examplers]
            else:
                print(f"[INFO] Cache has {len(cached)} exemplars but need {num_examplers}, regenerating...")
        except (json.JSONDecodeError, Exception) as e:
            print(f"[WARN] Failed to load exampler cache: {e}, regenerating...")

    # Generate reasoning for exemplars (one-time cost)
    print(f"[INFO] Generating reasoning for {num_examplers} {dataset} exemplars (one-time, will be cached)...")
    cached = []

    for ie, exampler in enumerate(examplers[:num_examplers]):
        medical_agent = Agent(
            instruction='You are a helpful medical agent.',
            role='medical expert',
            model_info=model
        )

        if dataset == 'pubmedqa':
            ex_ctx = " ".join(exampler['CONTEXTS'])
            fmt_q = (f"[Example {ie+1}]\nContext: {ex_ctx}\n"
                     f"Question: {exampler['QUESTION']}\nAnswer (yes/no/maybe):")
            if use_think_format:
                fmt_a = f"{exampler['final_decision']}"
            else:
                fmt_a = f"Answer: {exampler['final_decision']}\n"
            reason = medical_agent.chat(
                f"You are a helpful medical agent. Below is an example of a medical research "
                f"question based on a context. Can you provide 1-2 sentences of reasoning to "
                f"support the answer?\n\n{fmt_q}\n\nAnswer: {exampler['final_decision']}"
            )

        elif dataset == 'medqa':
            fmt_q = exampler['question']
            # Use sorted key order for consistent caching
            choices = [f"({k}) {v}" for k, v in sorted(exampler['options'].items())]
            fmt_q += " " + ' '.join(choices)
            if use_think_format:
                fmt_a = f"({exampler['answer_idx']}) {exampler['answer']}"
            else:
                fmt_a = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}\n\n"
            reason = medical_agent.chat(
                f"You are a helpful medical agent. Below is an example of medical knowledge "
                f"question and answer. After reviewing the below medical question and answering, "
                f"can you provide 1-2 sentences of reason that support the answer as you didn't "
                f"know the answer ahead?\n\nQuestion: {fmt_q}\n\n"
                f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
            )

        cached.append({
            'question': fmt_q,
            'answer': fmt_a,
            'reason': reason,
        })
        print(f"  [INFO] Exampler {ie+1}/{num_examplers} reasoning generated.")

    # Save cache
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved exampler cache to {cache_path}")

    return cached


# ─── Hindsight Recap: post-hoc reasoning for correct samples ───

def _build_basic_recap_prompt(question, answer):
    """Build the hindsight recap prompt for a basic-difficulty sample (difficulty selection + evidence reasoning)."""
    expert_response = ""
    for agent in _active_agents:
        if agent.role == 'medical expert':
            for msg in reversed(agent.messages):
                if msg.get('role') == 'assistant':
                    expert_response = msg.get('content', '')
                    break
            break

    return (
        f"A medical question was correctly answered. Review the record and generate "
        f"hindsight analysis for each section below.\n\n"
        f"Question: {question}\n"
        f"Expert's analysis: {expert_response}\n"
        f"Final answer: {answer}\n\n"
        f"Generate your analysis using EXACTLY these delimiters:\n\n"
        f"[DIFFICULTY_RECAP]\n"
        f"Explain why this question is appropriate for single-expert (basic) analysis. "
        f"What makes it straightforward enough that multi-expert debate is unnecessary? (2-3 sentences)\n"
        f"[/DIFFICULTY_RECAP]\n\n"
        f"[SYNTHESIS_RECAP]\n"
        f"Identify the key evidence from the context that supports the answer. "
        f"Trace the clear reasoning chain from evidence to conclusion. (3-5 sentences)\n"
        f"[/SYNTHESIS_RECAP]"
    )


def _build_intermediate_recap_prompt(question, answer):
    """Build the hindsight recap prompt for an intermediate-difficulty sample, covering each pipeline step."""
    recruiter_output = ""
    expert_opinions = {}
    synthesis_report = ""
    moderator_decision = ""

    for agent in _active_agents:
        role = agent.role.lower()
        if role == 'recruiter':
            for msg in agent.messages:
                if msg.get('role') == 'assistant':
                    recruiter_output = msg.get('content', '')
        elif role == 'medical assistant':
            for msg in agent.messages:
                if msg.get('role') == 'assistant':
                    synthesis_report = msg.get('content', '')
        elif role == 'moderator':
            for msg in reversed(agent.messages):
                if msg.get('role') == 'assistant':
                    moderator_decision = msg.get('content', '')
                    break
        elif role not in ('medical expert', 'recap'):
            for msg in agent.messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    if len(content) > 20:
                        expert_opinions[agent.role] = content
                        break

    expert_summary = "\n".join(f"- {r}: {o[:300]}" for r, o in expert_opinions.items())

    return (
        f"A multi-expert medical consultation correctly answered a question. "
        f"Review the complete record and generate hindsight analysis.\n\n"
        f"Question: {question}\n"
        f"Recruited experts:\n{recruiter_output[:500]}\n"
        f"Expert opinions:\n{expert_summary}\n"
        f"Synthesis:\n{synthesis_report[:500]}\n"
        f"Final answer: {answer}\n\n"
        f"Generate analysis using EXACTLY these delimiters:\n\n"
        f"[DIFFICULTY_RECAP]\n"
        f"Explain why this question requires intermediate-level multi-expert analysis "
        f"rather than a single expert. Be specific to this question. (2-3 sentences)\n"
        f"[/DIFFICULTY_RECAP]\n\n"
        f"[RECRUITMENT_RECAP]\n"
        f"Explain why each recruited specialist was the right choice for this specific "
        f"question. (1 sentence per expert)\n"
        f"[/RECRUITMENT_RECAP]\n\n"
        f"[SYNTHESIS_RECAP]\n"
        f"Trace the evidence chain: what each expert contributed, how their insights "
        f"converged, and how this leads to the final answer. (3-5 sentences)\n"
        f"[/SYNTHESIS_RECAP]"
    )


def generate_hindsight_recap(question, difficulty, result_entry, model, dataset, img_path=None):
    """Generate a hindsight recap for a correctly answered sample using a single API call."""
    cprint("\n[INFO] Generating hindsight recap (correct sample)", 'yellow')

    response = result_entry.get('response', '')
    if isinstance(response, dict):
        response = str(list(response.values())[0])
    answer = _extract_answer_from_response(response)

    if difficulty == 'basic':
        recap_prompt = _build_basic_recap_prompt(question, answer)
    elif difficulty == 'intermediate':
        recap_prompt = _build_intermediate_recap_prompt(question, answer)
    else:
        return

    recap_agent = Agent(
        instruction=(
            "You are a medical reasoning analyst. Generate structured hindsight "
            "analysis using the exact section delimiters provided."
        ),
        role="recap",
        model_info=model
    )
    recap_response = recap_agent.chat(recap_prompt, max_tokens=2048)
    print(f"Recap: {recap_response[:200]}...")


def process_basic_query(question, examplers, model, args, img_path=None, cached_examplers=None):
    use_think = getattr(args, 'use_think_format', False)
    new_examplers = []

    if args.dataset in ['pathvqa', 'mimic-cxr-vqa']:
        instruction = 'You are a helpful medical assistant that answers questions based on medical images. First think step by step, then provide a very brief answer (1-5 words).'
        fmt = _format_instruction(args.dataset, 'basic', use_think)

        single_agent = Agent(instruction=instruction, role='medical expert', model_info=model)
        final_decision = single_agent.temp_responses(
            f'''{question}\n\n{fmt}''',
            max_tokens=2048, img_path=img_path)
        return final_decision

    if cached_examplers:
        # Use pre-generated exemplar reasonings from cache (no API calls needed)
        new_examplers = cached_examplers[:3]
    elif args.dataset == 'pubmedqa':
        for ie, exampler in enumerate(examplers[:3]):
            medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)

            ex_ctx = " ".join(exampler['CONTEXTS'])
            exampler_question = f"[Example {ie+1}]\nContext: {ex_ctx}\nQuestion: {exampler['QUESTION']}\nAnswer (yes/no/maybe):"
            exampler_answer = f"Answer: {exampler['final_decision']}\n"
            exampler_reason = medical_agent.chat(f"You are a helpful medical agent. Below is an example of a medical research question based on a context. Can you provide 1-2 sentences of reasoning to support the answer?\n\n{exampler_question}\n\n{exampler_answer}")

            new_examplers.append({'question': exampler_question, 'reason': exampler_reason, 'answer': exampler_answer})

    elif args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:3]):
            medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
            exampler_question = exampler['question']
            choices = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(choices)
            exampler_question += " " + ' '.join(choices)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}\n\n"
            exampler_reason = medical_agent.chat(f"You are a helpful medical agent. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\n{exampler_answer}")

            new_examplers.append({'question': exampler_question, 'reason': exampler_reason, 'answer': exampler_answer})

    instruction = 'You are a helpful assistant that answers medical questions based on provided contexts. Answer with yes, no, or maybe.' if args.dataset == 'pubmedqa' else 'You are a helpful assistant that answers multiple choice questions about medical knowledge.'
    fmt = _format_instruction(args.dataset, 'basic', use_think)

    single_agent = Agent(instruction=instruction, role='medical expert', examplers=new_examplers,
                         model_info=model, use_think_format=use_think)
    final_decision = single_agent.temp_responses(
        f'''{question}\n\n{fmt}''',
        max_tokens=4096, img_path=img_path)

    return final_decision

def _truncate_text(text, max_chars=400):
    """Truncate text to max_chars characters, breaking at sentence boundaries."""
    if not text or len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.5:
        truncated = truncated[:last_period + 1]
    return truncated + " ..."


def process_intermediate_query(question, examplers, model, args, img_path=None, cached_examplers=None):
    use_think = getattr(args, 'use_think_format', False)
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])

    if args.dataset in ['pathvqa', 'mimic-cxr-vqa']:
        recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts to solve a visual medical query (e.g., Pathology or Radiology)."""
        task_instruction = "Considering the medical question and the image, what kind of experts (e.g., Radiologist, Cardiologist, etc.) will you recruit?"
    else:
        recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."""
        task_instruction = "Considering the medical question and the options for the answer, what kind of experts will you recruit?"

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model)
    num_agents = 3  # You can adjust this number as needed
    recruited = tmp_agent.chat(f"Question: {question}\nYou can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nIMPORTANT: Do NOT answer the medical question itself. Only output the numbered list of experts in the format above. Do not include any reasoning or diagnosis.", img_path=img_path)

    agents_info = [agent_info.split(" - Hierarchy: ") for agent_info in recruited.split('\n') if agent_info]
    agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    agent_list = ""
    for i, agent in enumerate(agents_data):
        try:
            raw = agent[0].strip()

            if '.' in raw:
                _, raw = raw.split('.', 1)
            raw = raw.strip()

            parts = [p.strip() for p in raw.split('-', 1)]
            agent_role = parts[0].lower()

            description = parts[1].lower() if len(parts) == 2 and parts[1] else "provides medical expertise"
            agent_list += f"Agent {i+1}: {agent_role} - {description}\n"
        except Exception:
            agent_list += f"Agent {i+1}: {agent[0]}\n"

    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        try:
            agent_role = agent[0].split('-', 1)[0].split('.')[1].strip().lower()
            description = agent[0].split('-', 1)[1].strip().lower()
        except:
            continue
        
        inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."""
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    for idx, agent in enumerate(agents_data):
        try:
            print(f"Agent {idx+1} ({agent_emoji[idx]} {agent[0].split('-', 1)[0].strip()}): {agent[0].split('-', 1)[1].strip()}")
        except:
            print(f"Agent {idx+1} ({agent_emoji[idx]}): {agent[0]}")

    fewshot_examplers = ""

    if cached_examplers and args.dataset in ['pubmedqa', 'medqa']:
        # Use first cached exemplar (no API call needed)
        ex = cached_examplers[0]
        if use_think:
            # SFT model: reason already contains <think>...</think>\n(X) answer
            fewshot_examplers = f"{ex['question']}\n{ex['reason']}\n\n"
        else:
            fewshot_examplers = f"{ex['question']}\n{ex['answer'].rstrip()}\n{ex['reason']}\n\n"
    elif args.dataset == 'pubmedqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:1]):
            medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)

            ex_ctx = " ".join(exampler['CONTEXTS'])
            exampler_question = f"[Example {ie+1}]\nContext: {ex_ctx}\nQuestion: {exampler['QUESTION']}\nAnswer (yes/no/maybe):"
            exampler_answer = f"Answer: {exampler['final_decision']}"

            exampler_reason = medical_agent.chat(f"Below is a medical question based on a context. Provide a short reason for the answer.\n\n{exampler_question}\n\n{exampler_answer}")

            exampler_question += f"\n{exampler_answer}\n{exampler_reason}\n\n"
            fewshot_examplers += exampler_question

    elif args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:1]):
            medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
            exampler_question = f"[Example {ie+1}]\n" + exampler['question']
            options = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(options)
            exampler_question += " " + " ".join(options)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"

            exampler_reason = medical_agent.chat(f"Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\n{exampler_answer}")

            exampler_question += f"\n{exampler_answer}\n{exampler_reason}\n\n"
            fewshot_examplers += exampler_question

    print()
    cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
    print_tree(hierarchy_agents[0], horizontal=False)
    print()

    num_rounds = 2
    num_turns = 3
    num_agents = len(medical_agents)

    interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}

    cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

    round_opinions = {n: {} for n in range(1, num_rounds+1)}
    round_answers = {n: None for n in range(1, num_rounds+1)}
    initial_report = ""
    expert_fmt = _format_instruction(args.dataset, 'expert', use_think)
    for k, v in agent_dict.items():
        if args.dataset in ['pathvqa', 'mimic-cxr-vqa']:
            opinion = v.chat(
                f'''Please view the image and question, then provide your professional analysis.\nQuestion: {question}\n\n{expert_fmt}''',
                img_path=img_path, max_tokens=1024)
        else:
            opinion = v.chat(
                f'''Given the examplers, please return your answer to the medical query among the option provided.\n\n{fewshot_examplers}\n\nQuestion: {question}\n\n{expert_fmt}''',
                img_path=img_path, max_tokens=1024)
        initial_report += f"({k.lower()}): {opinion}\n"
        round_opinions[1][k.lower()] = opinion

    final_answer = None
    for n in range(1, num_rounds+1):
        print(f"== Round {n} ==")
        round_name = f"Round {n}"
        agent_rs = Agent(instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.", role="medical assistant", model_info=model)
        
        assessment = "".join(f"({k.lower()}): {_truncate_text(v, 400)}\n" for k, v in round_opinions[n].items())

        if use_think:
            synthesis_fmt = ("Provide your analysis in the following format:\n"
                             "<think>Key Knowledge: [3-5 bullet points]\n"
                             "Total Analysis: [2-3 sentences]</think>\n"
                             "your final answer")
        else:
            synthesis_fmt = ("You should output in exactly the same format as: Key Knowledge:; Total Analysis:\n"
                             "Be concise — limit Key Knowledge to 3-5 bullet points and Total Analysis to 2-3 sentences.")
        report = agent_rs.chat(f'''Here are some reports from different medical domain experts.\n\n{assessment}\n\nYou need to complete the following steps\n1. Take careful and comprehensive consideration of the following reports.\n2. Extract key knowledge from the following reports.\n3. Derive the comprehensive and summarized analysis based on the knowledge\n4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n\n{synthesis_fmt}''')
        
        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            print(f"|_{turn_name}")

            num_yes = 0
            for idx, v in enumerate(medical_agents):
                all_comments = "".join(f"{_k} -> Agent {idx+1}: {_truncate_text(str(_v.get(f'Agent {idx+1}', '') or ''), 300)}\n" for _k, _v in interaction_log[round_name][turn_name].items())
                
                participate = v.chat("Given the opinions from other medical experts in your team, please indicate whether you want to talk to any expert. Reply with yes or no.\n\nOpinions:\n{}".format(assessment if n == 1 else all_comments))

                # Strip <think>...</think> tags before checking — the fine-tuned model
                # wraps its reasoning in thinking tags which can interfere with parsing.
                _participate_text = re.sub(r'<think>.*?</think>', '', participate, flags=re.DOTALL).lower().strip()
                if 'yes' in _participate_text or 'want to' in _participate_text or 'discuss' in _participate_text:                
                    chosen_expert = v.chat(f"Enter the number of the expert you want to talk to:\n{agent_list}\nFor example, if you want to talk with Agent 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 and don't return the reasons.")

                    # Strip <think> tags from expert number response
                    _chosen_text = re.sub(r'<think>.*?</think>', '', chosen_expert, flags=re.DOTALL).strip()
                    chosen_experts = [int(ce) for ce in _chosen_text.replace('.', ',').split(',') if ce.strip().isdigit()]

                    # Fallback: if no valid expert number parsed, default to talking to all other agents
                    if not chosen_experts:
                        chosen_experts = [j+1 for j in range(len(medical_agents)) if j != idx]
                        print(f" Agent {idx+1} ({agent_emoji[idx]} {v.role}): wants to discuss (defaulting to all other experts)")

                    for ce in chosen_experts:
                        # Bounds check: agent numbers are 1-indexed, must be in [1, len(medical_agents)]
                        if ce < 1 or ce > len(medical_agents):
                            print(f" [WARN] Agent {idx+1} chose invalid expert number {ce} (valid: 1-{len(medical_agents)}), skipping.")
                            continue
                        specific_question = v.chat(f"Please remind your medical expertise and then leave your opinion to an expert you chose (Agent {ce}. {medical_agents[ce-1].role}). Deliver your opinion in 2-3 concise sentences to convince the other expert.", img_path=img_path, max_tokens=512)

                        print(f" Agent {idx+1} ({agent_emoji[idx]} {medical_agents[idx].role}) -> Agent {ce} ({agent_emoji[ce-1]} {medical_agents[ce-1].role}) : {specific_question}")
                        interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
                
                    num_yes += 1
                else:
                    print(f" Agent {idx+1} ({agent_emoji[idx]} {v.role}): \U0001f910")

            if num_yes == 0:
                break
        
        if num_yes == 0:
            break

        final_fmt = _format_instruction(args.dataset, 'final', use_think)
        tmp_final_answer = {}
        for i, agent in enumerate(medical_agents):
            if args.dataset in ['pathvqa', 'mimic-cxr-vqa']:
                response = agent.chat(f"Now that you've interacted with other medical experts, make your final answer.\n{question}\n\n{final_fmt}", img_path=img_path, max_tokens=1024)
            else:
                response = agent.chat(f"Now that you've interacted with other medical experts, make your final answer concisely.\n{question}\n\n{final_fmt}", img_path=img_path, max_tokens=1024)
            tmp_final_answer[agent.role] = response

        round_answers[round_name] = tmp_final_answer
        final_answer = tmp_final_answer

    print('\nInteraction Log')        
    myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents)+1):
        row = [f"Agent {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents)+1):
            if i == j:
                row.append('')
            else:
                i2j = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                j2i = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                
                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u270B ({i}<->{j})')

        myTable.add_row(row)
        if i != len(medical_agents):
            myTable.add_row(['' for _ in range(len(medical_agents)+1)])
    
    print(myTable)

    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
    
    moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model)

    if args.dataset in ['pathvqa', 'mimic-cxr-vqa']:
        vote_prompt = f"Given each agent's final answer and the medical image, please make the final answer.\n{final_answer}\n\nQuestion: {question}\n\nRespond with ONLY the answer in 1-5 words. No explanation, no description. For example: \"yes\", \"no\", \"liver\", \"Ziehl-Neelsen stain\"."
    elif args.dataset == 'pubmedqa':
        if use_think:
            vote_prompt = f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote.\n<think>brief reasoning</think>\nyour answer: yes/no/maybe\n{final_answer}\n\nQuestion: {question}"
        else:
            vote_prompt = f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote. like below format:\nAnswer: yes/no/maybe\n{final_answer}\n\nQuestion: {question}"
    elif args.dataset == 'medqa':
        if use_think:
            vote_prompt = f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote. Your answer should be in format like:\n<think>brief reasoning</think>\n(C) 2nd pharyngeal arch\n{final_answer}\n\nQuestion: {question}"
        else:
            vote_prompt = f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote. Your answer should be like below format:\nAnswer: C) 2th pharyngeal arch\n{final_answer}\n\nQuestion: {question}"

    _decision = moderator.temp_responses(vote_prompt, img_path=img_path)
    final_decision = {'majority': _decision}

    print("moderator's final decision (by majority vote):", _decision)
    print()

    return final_decision

def process_advanced_query(question, model, args, img_path=None):
    print("[STEP 1] Recruitment")
    group_instances = []

    if args.dataset == 'pubmedqa':
        task_specific_instruction = "Considering the medical question and the provided context, please return your recruitment plan to better make an accurate answer (yes, no, or maybe)."
        final_output_instruction = "Answer (yes/no/maybe): "
    elif args.dataset in ['pathvqa', 'mimic-cxr-vqa']:
        task_specific_instruction = "Considering the medical question and the provided medical image, please return your recruitment plan to better make an accurate diagnosis."
        final_output_instruction = "Answer: "
    else:
        task_specific_instruction = "Considering the medical question and the options, please return your recruitment plan to better make an accurate answer."
        final_output_instruction = "Answer: "

    recruit_prompt = f"""You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."""

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model)
    num_teams = 3
    num_agents = 3 

    recruited = tmp_agent.chat(f"Question: {question}\n\nYou should organize {num_teams} MDTs with different specialties or purposes and each MDT should have {num_agents} clinicians. {task_specific_instruction}\n\nFor example, the following can an example answer:\nGroup 1 - Initial Assessment Team (IAT)\nMember 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.\nMember 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.\nMember 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.\n\nGroup 2 - Diagnostic Evidence Team (DET)\nMember 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.\nMember 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.\nMember 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.\n\nGroup 3 - Patient History Team (PHT)\nMember 1: Psychiatrist or Psychologist (Lead) - Addresses any psychological impacts of the chronic disease and its treatments, including issues related to voice changes, self-esteem, and coping strategies.\nMember 2: Physical Therapist - Offers exercises and strategies to maintain physical health and potentially support vocal function recovery indirectly through overall well-being.\nMember 3: Vocational Therapist - Assists the patient in adapting to changes in voice, especially if their profession relies heavily on vocal communication, helping them find strategies to maintain their occupational roles.\n\nGroup 4 - Final Review and Decision Team (FRDT)\nMember 1: Senior Consultant from each specialty (Lead) - Provides overarching expertise and guidance in decision\nMember 2: Clinical Decision Specialist - Coordinates the different recommendations from the various teams and formulates a comprehensive treatment plan.\nMember 3: Advanced Diagnostic Support - Utilizes advanced diagnostic tools and techniques to confirm the exact extent and cause of nerve damage, aiding in the final decision.\n\nAbove is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you return your answer, please strictly refer to the above format.", img_path=img_path)

    groups = [group.strip() for group in recruited.split("Group") if group.strip()]
    group_strings = ["Group " + group for group in groups]
    
    for i1, gs in enumerate(group_strings):
        res_gs = parse_group_info(gs)
        print(f"Group {i1+1} - {res_gs['group_goal']}")
        for i2, member in enumerate(res_gs['members']):
            print(f" Member {i2+1} ({member['role']}): {member['expertise_description']}")
        print()

        group_instance = Group(res_gs['group_goal'], res_gs['members'], question, model_info=model, dataset=args.dataset)
        group_instances.append(group_instance)

    # STEP 2. initial assessment from each group
    # STEP 2.1. IAP Process
    initial_assessments = []
    for group_instance in group_instances:
        if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower():
            init_assessment = group_instance.interact(comm_type='internal', img_path=img_path)
            initial_assessments.append([group_instance.goal, init_assessment])

    initial_assessment_report = ""
    for idx, init_assess in enumerate(initial_assessments):
        initial_assessment_report += f"Group {idx+1} - {init_assess[0]}\n{init_assess[1]}\n\n"

    # STEP 2.2. other MDTs Process
    assessments = []
    for group_instance in group_instances:
        if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
            assessment = group_instance.interact(comm_type='internal', img_path=img_path)
            assessments.append([group_instance.goal, assessment])
    
    assessment_report = ""
    for idx, assess in enumerate(assessments):
        assessment_report += f"Group {idx+1} - {assess[0]}\n{assess[1]}\n\n"
    
    # STEP 2.3. FRDT Process
    final_decisions = []
    for group_instance in group_instances:
        if 'review' in group_instance.goal.lower() or 'decision' in group_instance.goal.lower() or 'frdt' in group_instance.goal.lower():
            decision = group_instance.interact(comm_type='internal', img_path=img_path)
            final_decisions.append([group_instance.goal, decision])
    
    compiled_report = ""
    for idx, decision in enumerate(final_decisions):
        compiled_report += f"Group {idx+1} - {decision[0]}\n{decision[1]}\n\n"

    # STEP 3. Final Decision
    decision_prompt = f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query."""
    
    tmp_agent = Agent(instruction=decision_prompt, role='decision maker', model_info=model)
    tmp_agent.chat(decision_prompt)

    final_decision = tmp_agent.temp_responses(f"""Investigation:\n{initial_assessment_report}\n\nQuestion: {question}\n\n{final_output_instruction}""", img_path=img_path)

    return final_decision