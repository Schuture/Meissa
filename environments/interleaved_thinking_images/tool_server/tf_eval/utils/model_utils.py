import os,sys,torch

from .utils import *

import json, re
from copy import deepcopy


def answer_sequence_to_str(answer_sequence):
    res = []
    for idx,step in enumerate(answer_sequence):
        res.append(f"Step {idx+1}. {step['text']}\n\n")
    res_str = "".join(res)
    return res_str

def answer_sequence_to_shepherd_str(answer_sequence,step_tag = 'ки'):
    res = []
    for idx,step in enumerate(answer_sequence):
        res.append(f"Step {idx+1}: {step['text']} {step_tag}\n")
    res_str = "".join(res)
    return res_str

def answer_sequence_to_reasoneval_list(answer_sequence):
    res = []
    for idx,step in enumerate(answer_sequence):
        res.append(f"{idx+1}. {step['text']}")
    return res
    


def score_list_to_str(score_list):
    valid2_list = [str(round(i,2)) for i in score_list]
    res =  ", ".join(valid2_list)
    return res


def clean_str(input_str):
    res_str = deepcopy(input_str)
    res_str = re.sub(r'\\+([^\\\s])', r'\\\\\1', res_str)
    res_str = re.sub(r'\\+([\s])', r'\\\\\\\\\1', res_str)
    return res_str

def remove_comments_from_json(json_string):
    """
     JSON 
    """

    return re.sub(r'//.*?$|#.*?$', '', json_string, flags=re.MULTILINE)

def extract_nested_json(text):
    """
    Extract the first valid JSON object or array from a text string.
    Args:
        text (str): Text that may contain embedded JSON.
    Returns:
        dict or list or None: Parsed JSON data, or None if parsing fails.
    """
    stack = []
    start = -1
    for i, char in enumerate(text):
        if char == "{":
            if not stack:
                start = i
            stack.append("{")
        elif char == "}":
            stack.pop()
            if not stack:
                try:
                    json_str = text[start:i+1]
                    json_cleaned = remove_comments_from_json(json_str)
                    return json.loads(json_cleaned)
                except json.JSONDecodeError as e:
                    continue
    return None

def process_policy_lm_evaluation_response(response):
    """ process the response STRING from the language model"""
    try:
        json_object = extract_nested_json(response)
        assert json_object is not None
        assert "validity" in json_object and "redundancy" in json_object
        return json_object
    except :
        print(f"Invalid JSON Str, response: {response}")
        return None


def remove_step_prefix(text):
    """Remove step prefixes like 'Step x. ', 'step x. ', or 'x. ' from text."""
    text = text.strip()
    return re.sub(r"^(Step\s*\d+\.\s*|\d+\.\s*)", "", text, flags=re.IGNORECASE)

def find_subsequence(tensor, subsequence):
    """
    Find all starting positions of a subsequence within a tensor.

    Args:
        tensor (torch.Tensor): The main tensor to search in.
        subsequence (torch.Tensor): The subsequence tensor to find.

    Returns:
        List[int]: List of start indices where the subsequence is found.
    """
    main_len = tensor.size(0)
    sub_len = subsequence.size(0)

    positions = []
    for i in range(main_len - sub_len + 1):
        if torch.equal(tensor[i:i+sub_len], subsequence):
            positions.append(i)
    return positions

