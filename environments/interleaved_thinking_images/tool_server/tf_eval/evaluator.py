import os
import itertools
import json
import base64
import shutil
from pathlib import Path
import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch, gc
import yaml
import argparse
from torch.utils.data import DataLoader
import torch.distributed as dist

from .models import get_model
from .tasks import get_task_object, get_task_functions
from .tasks.base_dataset.base_evaluation_dataset import BaseEvalDataset, DataCollatorForSupervisedDataset

from .utils.utils import *
from .utils.arguments import *

from .utils.log_utils import get_logger, set_verbosity
from .tool_inferencer import BaseToolInferencer
import pdb
import re
try:
    from math_verify import parse, verify
except ImportError:
    print("math_verify package not found. Please install it to use math verification features.")

logger = get_logger(__name__)

def _write_jsonl_lines(path: str, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def _strip_edited_image_for_logging(obj):
    """
    Remove huge base64 strings from tool responses before writing raw_results.jsonl.
    Keep a small placeholder so debugging still works.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "edited_image" and isinstance(v, str):
                new_obj[k] = "<B64_STRIPPED>"
            else:
                new_obj[k] = _strip_edited_image_for_logging(v)
        return new_obj
    if isinstance(obj, list):
        return [_strip_edited_image_for_logging(x) for x in obj]
    return obj

def save_b64_image(b64_str, root_dir, sample_idx, round_idx):
    """Save a Base64-encoded intermediate image to the specified directory."""
    try:
        sample_dir = os.path.join(root_dir, str(sample_idx))
        os.makedirs(sample_dir, exist_ok=True)
        
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
            
        img_data = base64.b64decode(b64_str)
        file_name = f"round_{round_idx}.png"
        file_path = os.path.join(sample_dir, file_name)
        
        with open(file_path, "wb") as f:
            f.write(img_data)
        return os.path.abspath(file_path)
    except Exception as e:
        logger.error(f"Error saving b64 image for {sample_idx} round {round_idx}: {e}")
        return None

def copy_original_image(source_path, root_dir, sample_idx):
    """
     trajectory 
    """
    try:
        if not source_path or not os.path.exists(source_path):
            return None
            
        sample_dir = os.path.join(root_dir, str(sample_idx))
        os.makedirs(sample_dir, exist_ok=True)
        
        ext = os.path.splitext(source_path)[1]
        if not ext:
            ext = ".jpg"
            
        file_name = f"original{ext}"
        dest_path = os.path.join(sample_dir, file_name)
        
        shutil.copy2(source_path, dest_path)
        
        return os.path.abspath(dest_path)
    except Exception as e:
        logger.error(f"Error copying original image for {sample_idx}: {e}")
        return source_path

def _safe_json_loads_maybe_fenced(s: str):
    """Parse JSON that may be wrapped in ```json fences."""
    if s is None:
        return None
    clean = s.strip()
    if clean.startswith("```json"):
        clean = clean[7:].strip()
    if clean.startswith("```"):
        clean = clean[3:].strip()
    if clean.endswith("```"):
        clean = clean[:-3].strip()
    try:
        return json.loads(clean)
    except Exception:
        return None


def convert_to_sft_format(item, full_meta, tools_def_str, image_root_dir):
    """
    Convert one inference result into a unified SFT trajectory format.

    Supported formats (auto-branch):
    - Direct mode (no tools_def_str):
      * "Thought: ... Final Answer: ..." text
      * already formatted "[FINAL] ..."
      * anything else -> treat as final answer

    - Tool mode (tools_def_str exists):
      * Old JSON per-round: {"thought": "...", "actions": [...]}
      * New JSON per-round: {"actions": [...]} and final Terminate JSON contains {"recap": [...]}
        - recap includes tool steps + final Terminate step (Terminate has only "why")
      * Mixed / partial -> best-effort, otherwise drop the sample
    """
    import os, json

    def _strip_json_fence(s: str) -> str:
        if s is None:
            return ""
        t = s.strip()
        if t.startswith("```json"):
            t = t[7:].strip()
        if t.startswith("```"):
            t = t[3:].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
        return t.strip()

    def _safe_json_loads(s: str):
        try:
            return json.loads(_strip_json_fence(s))
        except Exception:
            return None

    def _think_block(text: str) -> str:
        text = "" if text is None else str(text)
        return f"<think>\n{text}\n</think>\n"

    def _post_summary_from_recap(recap_item: dict) -> str:
        # For tool steps only. Terminate recap is handled separately.
        if not isinstance(recap_item, dict):
            return ""
        if str(recap_item.get("tool", "")).lower() == "terminate":
            return ""

        got = (recap_item.get("got") or "").strip()
        evidence = (recap_item.get("evidence") or "").strip()
        inference = (recap_item.get("inference") or "").strip()

        parts = []
        if got:
            parts.append(got)
        if evidence:
            parts.append(evidence)
        if inference:
            parts.append(inference)
        return "\n".join(parts).strip()

    def _normalize_recap_sequence(recap: list, action_names: list) -> list:
        # Best-effort alignment by index (do not overfit strict matching).
        if not isinstance(recap, list):
            return [{} for _ in action_names]
        out = []
        for i in range(len(action_names)):
            if i < len(recap) and isinstance(recap[i], dict):
                out.append(recap[i])
            else:
                out.append({})
        return out

    meta = item.get("meta_data", {}) or {}
    idx = meta.get("idx", "unknown")
    model_responses = item.get("model_response", []) or []
    tool_responses = item.get("tool_response", []) or []

    # 1) Find original image path
    original_source_path = (
        meta.get("image_path")
        or meta.get("image_file")
        or full_meta.get(idx, {}).get("image_path")
        or full_meta.get(idx, {}).get("image_file")
        or ""
    )

    # Tool mode if tools_def_str non-empty
    is_tool_mode = bool(tools_def_str is not None and len(tools_def_str) > 0)

    final_original_path = ""
    if original_source_path:
        if is_tool_mode:
            copied_path = copy_original_image(original_source_path, image_root_dir, idx)
            final_original_path = copied_path if copied_path else original_source_path
        else:
            final_original_path = original_source_path

    # images list starts with original (if exists)
    images_list = [final_original_path] if final_original_path else []

    # 2) Build human message
    conversations = []
    question = meta.get("text", "") or full_meta.get(idx, {}).get("text", "")
    if final_original_path or len(images_list) == 0:
        human_val = f"<image>\n{question}"
    else:
        human_val = f"Question: {question}"
    conversations.append({"from": "human", "value": human_val})

    # =========================
    # Direct mode (no tools)
    # =========================
    if not tools_def_str:
        if not model_responses:
            return None

        raw_resp = str(model_responses[-1]).strip()
        formatted_value = ""

        # Case: "Thought: ... Final Answer: ..."
        if "Final Answer:" in raw_resp:
            parts = raw_resp.split("Final Answer:", 1)
            thought_part = parts[0].replace("Thought:", "").strip()
            answer_part = parts[1].strip()
            formatted_value = f"{_think_block(thought_part)}[FINAL] {answer_part}"

        # Case: already has [FINAL]
        elif "[FINAL]" in raw_resp:
            # Ensure there is a think block; if none, add empty
            if "<think>" in raw_resp and "</think>" in raw_resp:
                formatted_value = raw_resp
            else:
                # put everything before [FINAL] into think if possible
                pre, post = raw_resp.split("[FINAL]", 1)
                pre = pre.strip()
                post = post.strip()
                formatted_value = f"{_think_block(pre)}[FINAL] {post}" if pre else f"{_think_block('')}[FINAL] {post}"

        # Case: "Thought:" only
        elif "Thought:" in raw_resp:
            thought_part = raw_resp.replace("Thought:", "").strip()
            formatted_value = f"{_think_block(thought_part)}[FINAL] (No answer generated)"

        # Fallback: treat as final answer
        else:
            formatted_value = f"{_think_block('')}[FINAL] {raw_resp}"

        target_system_prompt = (
            "You are a helpful visual assistant. "
            "Given an image and a question, answer the question directly.\n"
            "You possess an internal chain of thought.\n"
            "1. First, reason about the image and question inside <think> and </think> tags.\n"
            "2. Then, output the concise final answer starting with [FINAL].\n"
            "Do not use any tools."
        )

        conversations.append({"from": "gpt", "value": formatted_value})

        # If original image missing, drop (keeps your previous safety filter)
        if not final_original_path:
            return None

        return {
            "conversations": conversations,
            "images": images_list,
            "tools": "",
            "system": target_system_prompt,
            "meta": {"dataset": meta.get("dataset", "pathvqa"), "idx": idx},
        }

    # =========================
    # Tool mode (JSON actions)
    # =========================
    # Pass 1: collect action sequence + observations + (old thoughts) + (final recap)
    actions_seq = []         # [{"name":..., "arguments":..., "model_i": i}]
    obs_seq = []             # one per non-Terminate tool action
    action_names = []        # action name per step (including Terminate)
    thought_seq = []         # old format per step (may be "")
    has_any_thought = False

    recap_seq = None
    final_ans = None

    for i, raw_resp in enumerate(model_responses):
        resp_json = _safe_json_loads(str(raw_resp))
        if not isinstance(resp_json, dict):
            continue
        actions = resp_json.get("actions", [])
        if not isinstance(actions, list) or len(actions) == 0:
            continue

        action = actions[0] if isinstance(actions[0], dict) else None
        if not isinstance(action, dict):
            continue

        action_name = action.get("name")
        if not action_name:
            continue
        act_args = action.get("arguments", {}) or {}

        # record this step
        actions_seq.append({"name": action_name, "arguments": act_args, "model_i": i})
        action_names.append(action_name)

        # old-format thought (may be absent)
        th = resp_json.get("thought", "")
        if isinstance(th, str) and th.strip():
            has_any_thought = True
            thought_seq.append(th.strip())
        else:
            thought_seq.append("")

        # terminate?
        if str(action_name).lower() == "terminate":
            final_ans = (
                act_args.get("ans")
                or act_args.get("answer")
                or act_args.get("param")
                or ""
            )
            final_ans = str(final_ans).strip()
            recap_seq = resp_json.get("recap", None)
            break

        # build observation for this tool call from tool_responses[i]
        obs_content = ""
        if i < len(tool_responses):
            t_res = tool_responses[i]
            if t_res:
                if "edited_image" in t_res:
                    saved_path = save_b64_image(t_res["edited_image"], image_root_dir, idx, i)
                    if saved_path:
                        images_list.append(saved_path)
                        obs_content += "<image>\n"
                if "text" in t_res:
                    obs_content += t_res["text"]
                elif "error" in t_res:
                    obs_content += f"Error: {t_res['error']}"
            else:
                obs_content = "Tool execution failed."
        else:
            obs_content = "Tool execution failed."

        obs_seq.append(obs_content)

    # Must have final answer
    if not final_ans:
        return None

    # If original image missing, drop
    if not final_original_path:
        return None

    # Decide branch:
    # - New format: final recap exists -> backfill thinks from recap
    # - Old format: no recap but has thoughts -> use thoughts directly
    use_recap = isinstance(recap_seq, list)
    use_thought = (not use_recap) and has_any_thought

    if not use_recap and not use_thought:
        return None

    # Pass 2: build conversations with unified "<think>...</think>yyy"
    tool_obs_idx = 0

    if use_recap:
        recap_seq = _normalize_recap_sequence(recap_seq, action_names)

        prev_post = ""  # got/evidence/inference from previous tool step
        for step_idx, act in enumerate(actions_seq):
            act_name = act["name"]
            act_args = act["arguments"] or {}
            recap_item = recap_seq[step_idx] if step_idx < len(recap_seq) else {}

            cur_why = (recap_item.get("why") or "").strip()

            # think = prev_post + cur_why
            think_parts = []
            if prev_post:
                think_parts.append(prev_post)
            if cur_why:
                think_parts.append(cur_why)
            think_text = "\n".join([p for p in think_parts if p]).strip()

            if str(act_name).lower() == "terminate":
                conversations.append({
                    "from": "gpt",
                    "value": f"{_think_block(think_text)}[FINAL] {final_ans}"
                })
                break

            # tool call
            tool_call_payload = json.dumps({"name": act_name, "arguments": act_args}, ensure_ascii=False)
            conversations.append({
                "from": "function_call",
                "value": f"{_think_block(think_text)}{tool_call_payload}"
            })

            # observation
            if tool_obs_idx < len(obs_seq):
                conversations.append({"from": "observation", "value": obs_seq[tool_obs_idx]})
            else:
                conversations.append({"from": "observation", "value": "Tool execution failed."})
            tool_obs_idx += 1

            # update prev_post from THIS recap tool step (not Terminate)
            prev_post = _post_summary_from_recap(recap_item)

    else:
        # Old format fallback: per-step thought is already the tool-call thinking
        for step_idx, act in enumerate(actions_seq):
            act_name = act["name"]
            act_args = act["arguments"] or {}
            th = thought_seq[step_idx] if step_idx < len(thought_seq) else ""

            if str(act_name).lower() == "terminate":
                conversations.append({
                    "from": "gpt",
                    "value": f"{_think_block(th)}[FINAL] {final_ans}"
                })
                break

            tool_call_payload = json.dumps({"name": act_name, "arguments": act_args}, ensure_ascii=False)
            conversations.append({
                "from": "function_call",
                "value": f"{_think_block(th)}{tool_call_payload}"
            })

            if tool_obs_idx < len(obs_seq):
                conversations.append({"from": "observation", "value": obs_seq[tool_obs_idx]})
            else:
                conversations.append({"from": "observation", "value": "Tool execution failed."})
            tool_obs_idx += 1

    # Must end with [FINAL]
    if not conversations or conversations[-1].get("from") != "gpt" or "[FINAL]" not in conversations[-1].get("value", ""):
        return None

    # System prompt (kept identical to your existing tool-mode target)
    system_prompt = (
        "You are a visual assistant for medical images. "
        "Your goal is to answer correctly by deciding whether to use tools.\n"
        "--------------------------------------------------\n"
        "IMAGE REFERENCE PROTOCOL\n"
        "--------------------------------------------------\n"
        "- \"img_original\": The initial full-resolution input image.\n"
        "- \"img_last\": The output image from the immediate previous step (default).\n"
        "- \"img_round_N\": The output image from a specific past step N (e.g. \"img_round_0\").\n"
        "The system will explicitly tell you the ID of the generated image in the Observation (e.g. \"[Output Image ID: img_round_0]\").\n"
        "--------------------------------------------------\n"
        "THINKING PROCESS\n"
        "--------------------------------------------------\n"
        "You possess an internal chain of thought. Before taking any action (calling a tool) or giving a final answer, "
        "you MUST enclose your reasoning, planning, and reflection process within <think> and </think> tags.\n"
        "--------------------------------------------------\n"
        "TOOL USE\n"
        "--------------------------------------------------\n"
        "If you need external information to answer the question, generate a function call.\n"
        "1. Analyze the current state in your <think> block.\n"
        "2. Call the appropriate tool with precise arguments.\n"
        "3. Review the observation in your next <think> block.\n"
        "--------------------------------------------------\n"
        "FINAL ANSWER\n"
        "--------------------------------------------------\n"
        "When you have sufficient information, output [FINAL] followed by the answer."
    )

    return {
        "conversations": conversations,
        "images": images_list,
        "tools": tools_def_str,
        "system": system_prompt,
        "meta": {"dataset": meta.get("dataset", "pathvqa"), "idx": idx},
    }



class TFEvaluator():
    def __init__(self, model_args, task_args, script_args):
        self.config = script_args.config
        self.model_args = model_args
        self.task_args = task_args
        self.script_args = script_args
        self.tasks = self.task_args.task_name
        self.model = get_model(self.model_args.model)(**self.model_args.model_args)
        
        max_rounds = self.model_args.max_rounds
        stop_token = self.model_args.stop_token

        set_verbosity(self.script_args.verbosity)

        self.inferencer = BaseToolInferencer(
            tp_model=self.model,
            batch_size=self.model_args.batch_size,
            model_mode=self.model_args.model_mode,
            max_rounds=max_rounds, 
            stop_token=stop_token,
            controller_addr=self.script_args.controller_addr,
        )

    def evaluate(self):
        for task_name in self.tasks:
            logger.info(f"evaluating {task_name}")
            task_dict = get_task_functions(task_name)
            load_data_function = task_dict["load_data_function"]
            evaluate_function = task_dict["evaluate_function"] 
            task_config = task_dict["task_config"]

            self.model.set_generation_config(task_config.generation_config)

            dataset = BaseEvalDataset(
                load_data_function=load_data_function,
                getitem_function=self.model.getitem_fn,
                evaluate_function=evaluate_function,
                task_config=task_config,
                task_args=self.task_args,
                model_args=self.model_args,
            )

            # ================= [Setup Paths] =================
            raw_path = self.script_args.output_path + ".raw_results.jsonl"
            per_sample_path = self.script_args.output_path + ".per_sample.jsonl"
            sft_path = self.script_args.output_path + ".sft_trajectory.jsonl"
            
            log_file_path = Path(self.script_args.output_path)
            image_root_dir = log_file_path.parent / (log_file_path.stem + "_images")
            if not image_root_dir.exists():
                image_root_dir.mkdir(parents=True, exist_ok=True)

            # ================= [Resume Logic: Load Finished] =================
            finished_idxs = set()
            if os.path.exists(raw_path):
                logger.info(f"Checking existing results in {raw_path} for resume...")
                try:
                    with open(raw_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    item = json.loads(line)
                                    if "idx" in item:
                                        finished_idxs.add(item["idx"])
                                except:
                                    pass
                except Exception as e:
                    logger.warning(f"Failed to read existing log for resume: {e}")
            
            logger.info(f"Found {len(finished_idxs)} finished samples. They will be skipped.")

            # ================= [Filter Logic: Load Error Samples] =================
            target_idxs = None
            filter_file = os.environ.get("ONLY_RUN_ERRORS_FROM", None)
            
            if filter_file and os.path.exists(filter_file):
                logger.info(f"Filtering dataset based on errors in: {filter_file}")
                target_idxs = set()
                try:
                    with open(filter_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip(): continue
                            item = json.loads(line)
                            if item.get("score", 0) == 0:
                                target_idxs.add(item["idx"])
                    
                    logger.info(f"Found {len(target_idxs)} error samples to re-run.")
                    
                    if len(target_idxs) == 0:
                        logger.warning("No error samples found! The inference will run nothing.")
                        
                except Exception as e:
                    logger.error(f"Failed to load filter file: {e}")
                    target_idxs = None # Fallback to run all

            # ================= [Prepare Tools Definition] =================
            if self.model_args.max_rounds == 0:
                tools_def_str = ""
                logger.info("Evaluation Mode: DIRECT (No Tools)")
            else:
                tools_def_str = json.dumps([
                    {
                        "name": "ZoomInSubfigure",
                        "description": "Crops the image to a specific region to see visual details clearly. Useful when the question refers to a local region but the image is too large or cluttered.",
                        "parameters": {"type": "object", "properties": {"image": {"type": "string", "description": 'The image identifier to operate on. Use "img_last" for the result of the previous step (default), "img_original" for the initial full image, or "img_round_N" (e.g., "img_round_0") for a specific past result.'}, "param": {"type": "string", "description": "the bounding box coordinates as a list [x1, y1, x2, y2]. current image size is 1000x1000"}}, "required": ["image", "param"]}
                    },
                    {
                        "name": "SegmentRegionAroundPoint",
                        "description": "Segments a specific object or region around given point coordinates. This tool should be used ONLY when the location of interest is known or can be precisely specified by coordinates.",
                        "parameters": {"type": "object", "properties": {"image": {"type": "string", "description": 'The image identifier. Use "img_last" (default), "img_original", or "img_round_N".'}, "param": {"type": "string", "description": "the coordinates x=\"value\" y=\"value\" (0-1000 scale, where 500 is the center)."}}, "required": ["image", "param"]}
                    },
                    {
                        "name": "BioMedParseTextSeg",
                        "description": "Performs text-guided semantic segmentation on medical images. This tool is especially useful for identifying and localizing semantic medical entities such as neoplastic cells, inflammatory cells, tumor tissue, normal tissue, pathological structures. Use this tool when the question asks about the presence, location, extent, or appearance of a medically meaningful region or cell type, and precise coordinates are NOT given.",
                        "parameters": {"type": "object", "properties": {"image": {"type": "string", "description": 'The image identifier. Use "img_last" (default), "img_original", or "img_round_N".'}, "param": {"type": "string", "description": "a text description of the target(s) to segment, e.g. \"neoplastic cells; inflammatory cells\"."}}, "required": ["image", "param"]}
                    },
                    {
                        "name": "Terminate",
                        "description": "Concludes the task and provides the final answer. This tool must be used to finalize the response.",
                        "parameters": {"type": "object", "properties": {"ans": {"type": "string", "description": "the final answer to the question being addressed."}}, "required": ["ans"]}
                    }
                ])
                logger.info("Evaluation Mode: TOOL USE")

            full_meta_dict = {}
            full_meta_data_list = getattr(dataset, 'meta_data', None)
            if full_meta_data_list is None:
                 if hasattr(dataset, 'data'): full_meta_data_list = dataset.data
                 elif hasattr(dataset, 'dataset'): 
                     full_meta_data_list = dataset.dataset if isinstance(dataset.dataset, list) else dataset.dataset.get('data')
            
            if full_meta_data_list:
                full_meta_dict = {item['idx']: item for item in full_meta_data_list}
            else:
                logger.error("Could not find meta_data in dataset!")

            # ================= [Universal Streaming Callback] =================
            def streaming_callback(res_item):
                if not is_main_process():
                    return
                
                idx = res_item.get("idx")
                if not idx: return
                
                safe_res_item = _strip_edited_image_for_logging(res_item)
                with open(raw_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(safe_res_item, ensure_ascii=False) + "\n")
                
                current_meta = full_meta_dict.get(idx)
                if current_meta:
                    try:
                        import copy
                        clean_item = copy.deepcopy(res_item)
                        raw_output = clean_item["results"].get("final_answer", "")
                        
                        if "Final Answer:" in raw_output:
                            pure_answer = raw_output.split("Final Answer:", 1)[1].strip()
                            if pure_answer.endswith("."):
                                pure_answer = pure_answer[:-1]
                            clean_item["results"]["final_answer"] = pure_answer
                        
                        elif raw_output.strip().startswith("{") and "actions" in raw_output:
                            try:
                                data = json.loads(raw_output)
                                extracted_ans = None
                                if "actions" in data and isinstance(data["actions"], list):
                                    for act in data["actions"]:
                                        if act.get("name", "").lower() == "terminate":
                                            args = act.get("arguments", {})
                                            extracted_ans = args.get("ans") or args.get("answer") or args.get("param")
                                            break
                                
                                if extracted_ans:
                                    clean_item["results"]["final_answer"] = str(extracted_ans).strip()
                            except Exception as parse_e:
                                pass
                        # ---------------------------

                        eval_out = evaluate_function([clean_item], [current_meta])
                        
                        compare_logs = eval_out.get("compare_logs", [])
                        if compare_logs:
                            log_item = compare_logs[0]
                            
                            out_sample = {
                                "task_name": task_name,
                                "model_name": self.model_args.model,
                                "idx": idx,
                                "score": log_item.get("score"),
                                "gold": log_item.get("gold"),
                                "pred": log_item.get("pred"),
                                "question": current_meta.get("text"),
                                "image_path": current_meta.get("image_path") or current_meta.get("image_file")
                            }
                            
                            with open(per_sample_path, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(out_sample, ensure_ascii=False) + "\n")
                    except Exception as e:
                        logger.error(f"Error evaluating per-sample for {idx}: {e}")

                inner_result = res_item.get("results", {})
                if inner_result.get("status") == "finished":
                    try:
                        sft_item = convert_to_sft_format(inner_result, full_meta_dict, tools_def_str, str(image_root_dir))
                        if sft_item:
                            with open(sft_path, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(sft_item, ensure_ascii=False) + "\n")
                    except Exception as e:
                        logger.error(f"Error generating SFT data for {idx}: {e}")

            # ================= [Run Inference] =================
            self.inferencer.batch_inference(
                dataset, 
                result_callback=streaming_callback,
                finished_idxs=finished_idxs,
                target_idxs=target_idxs
            )

            # ================= [Final Global Evaluation] =================
            if is_main_process():
                logger.info(f"Inference finished. Calculating global metrics from {raw_path}...")
                
                all_results = []
                if os.path.exists(raw_path):
                    with open(raw_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    all_results.append(json.loads(line))
                                except:
                                    pass
                else:
                    logger.warning(f"Raw results file not found at {raw_path}")

                if full_meta_data_list and all_results:
                    try:
                        result_idxs = set(r['idx'] for r in all_results)
                        
                        filtered_meta_data = [m for m in full_meta_data_list if m['idx'] in result_idxs]
                        
                        logger.info(f"Computing metrics on {len(filtered_meta_data)} samples.")
                        
                        clean_results = []
                        import copy
                        
                        for res in all_results:
                            clean_item = copy.deepcopy(res)
                            raw_output = clean_item["results"].get("final_answer", "")
                            
                            
                            if "Final Answer:" in raw_output:
                                try:
                                    pure_answer = raw_output.split("Final Answer:", 1)[1].strip()
                                    if pure_answer.endswith("."):
                                        pure_answer = pure_answer[:-1]
                                    clean_item["results"]["final_answer"] = pure_answer
                                except:
                                    pass
                            
                            elif raw_output.strip().startswith("{") and "actions" in raw_output:
                                try:
                                    data = json.loads(raw_output)
                                    extracted_ans = None
                                    if "actions" in data and isinstance(data["actions"], list):
                                        for act in data["actions"]:
                                            if act.get("name", "").lower() == "terminate":
                                                args = act.get("arguments", {})
                                                extracted_ans = args.get("ans") or args.get("answer") or args.get("param")
                                                break
                                    if extracted_ans:
                                        clean_item["results"]["final_answer"] = str(extracted_ans).strip()
                                except:
                                    pass
                            # -------------------------------------------------------
                            
                            clean_results.append(clean_item)
                            
                        final_metrics = evaluate_function(clean_results, filtered_meta_data)
                        # ===============================================================

                        global_acc = final_metrics.get("Acc", 0.0)
                        logger.info(f"Task: {task_name} | Global Acc: {global_acc}")

                        summary_file = self.script_args.output_path + '.overall.jsonl'
                        summary_data = {
                            "task_name": task_name,
                            "model_name": self.model_args.model,
                            "Acc": global_acc,
                            "num_samples": len(all_results)
                        }
                        
                        with open(summary_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(summary_data, ensure_ascii=False) + "\n")
                            
                        logger.info(f"Global summary saved to {summary_file}")
                        
                    except Exception as e:
                        logger.error(f"Failed to calculate global metrics: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    logger.warning("Skipping global evaluation: results or metadata missing.")

            logger.info(f"evaluation of {task_name} completed")