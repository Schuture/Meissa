
import torch
from torch.utils.data import DataLoader,Dataset
from accelerate import Accelerator
import requests
import re
import copy
import time
import random

from ..models.abstract_model import tp_model
from .dynamic_batch_manager import DynamicBatchManager
from ..utils.utils import *
from ..utils.log_utils import get_logger
from ...tool_workers.tool_manager.base_manager import ToolManager
import torch.distributed as dist

logger = get_logger(__name__)

def is_rate_limit_error(e):
    e_str = str(e).lower()
    return "429" in e_str or "quota" in e_str or "resource exhausted" in e_str

def backoff_sleep(attempt, base=2.0, cap=60.0):
    sleep_time = min(cap, base * (2 ** attempt))
    jitter = random.uniform(0, 0.1 * sleep_time)
    time.sleep(sleep_time + jitter)

class BaseToolInferencer(object):
    def __init__(
        self,
        tp_model: tp_model = None,
        # dataset: Dataset = None,
        batch_size: int = 1,
        model_mode: str = "general",
        max_rounds: int = 1,
        stop_token: str = "<stop>",
        controller_addr: str = "http://SH-IDCA1404-10-140-54-5:20001",
    ):
        self.accelerator = Accelerator()
        self.tp_model = tp_model
        self.model_mode = model_mode 
        self.generate_conversation_fn = self.tp_model.generate_conversation_fn
        self.append_conversation_fn = self.tp_model.append_conversation_fn

        if dist.is_initialized() and self.accelerator.device.type == "cuda" and not 'vllm_models' in str(type(self.tp_model)):
            self.tp_model = self.tp_model.to(self.accelerator.device)
            self.tp_model = self.tp_model.to(torch.bfloat16)

        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.stop_token = stop_token
        self.controller_addr = controller_addr
        self.manager = DynamicBatchManager(
            batch_size=self.batch_size, 
            max_rounds=self.max_rounds, 
            stop_token=self.stop_token,
            generate_conversation_fn = self.tp_model.generate_conversation_fn,
        )
        self.tool_manager = ToolManager(controller_url_location=self.controller_addr if self.controller_addr else None)
        self.available_models = self.tool_manager.available_tools
        

    def batch_tool_response_to_next_round_input(self):
        current_batch = self.manager.get_current_batch()
        
        for idx,item in enumerate(current_batch):
            if item.model_response is None or item.status != "processing":
                continue
            
            tool_cfg = item.tool_cfg[item.current_round-1]
            tool_response = item.tool_response[item.current_round-1]
            # ... (asserts skipped)
            original_prompt = item.meta_data.get("text", "")
            
            if tool_response is not None:
                try:
                    if "edited_image" in tool_response:
                        edited_image_raw = tool_response.get("edited_image")
                        
                        # -------------------------------------------------
                        # -------------------------------------------------
                        if isinstance(edited_image_raw, str):
                            edited_image = base64_to_pil(edited_image_raw)
                        else:
                            edited_image = edited_image_raw
                            
                        item.current_image = edited_image

                        if self.model_mode == "llava_plus": 
                            pass 
                        elif self.model_mode in ["general", "opensource"]: 
                            pass
                        
                    else:
                        edited_image = None
                    
                    if "text" in tool_response:
                        tool_response_text = tool_response["text"]
                    else:
                        tool_response_text = None
                    
                    api_name = tool_cfg[0].get("API_name", tool_cfg[0].get("api_name", ""))

                    if self.model_mode == "llava_plus": 
                        new_response = f"{api_name} model outputs: {tool_response_text}\n\n"
                        new_round_prompt = f"{new_response} Please summarize the model outputs and answer my first question."
                    
                    elif self.model_mode in ["general", "opensource"]:
                        new_response = f"OBSERVATION:\n{api_name} model outputs: {tool_response_text}\n"
                        new_round_prompt = f"{new_response}Please summarize the model outputs and answer my first question."
                    
                    else:
                        new_round_prompt = original_prompt

                except Exception as e:
                    print(f"Error processing tool response: {e}")
                    import traceback
                    traceback.print_exc()
                    edited_image = None
                    new_round_prompt = original_prompt
            else:
                edited_image = None
                new_round_prompt = original_prompt
            
            new_round_input = dict(text=new_round_prompt,image=edited_image)
            item.new_round_input.append(new_round_input)
            
            item.conversation = self.append_conversation_fn(
                conversation=item.conversation, text=new_round_prompt, image=edited_image, role="user"
            )

    def _get_image_by_id(self, image_id, item):
        """
        Helper function: Resolve image_id string to PIL Image object
        """
        if image_id == "img_original":
            raw = item.meta_data.get("image", None)
            return load_image(raw) if raw else None

        if image_id == "img_last" or image_id == "img_1":
            if item.current_image is not None:
                return item.current_image
            else:
                raw = item.meta_data.get("image", None)
                return load_image(raw) if raw else None

        if image_id.startswith("img_round_"):
            try:
                round_idx = int(image_id.split("_")[-1])
                if round_idx < len(item.tool_response):
                    past_response = item.tool_response[round_idx]
                    if past_response and "edited_image" in past_response:
                        b64_str = past_response["edited_image"]
                        return base64_to_pil(b64_str)
            except Exception as e:
                print(f"Error retrieving {image_id}: {e}")
                pass
        
        if item.current_image: return item.current_image
        return load_image(item.meta_data.get("image"))
    
    def batch_get_tool_response(self):
        current_batch = self.manager.get_current_batch()
        for item in current_batch:
            if item.model_response is None or item.status != "processing":
                continue
            
            tool_cfg = item.tool_cfg[-1] if len(item.tool_cfg) > 0 else None
            assert len(item.tool_cfg) == item.current_round

            if item.current_image is not None:
                image = item.current_image
            else:
                image = item.meta_data.get("image", None)

            if tool_cfg is not None and len(tool_cfg) > 0:
                assert item.status == "processing"
                try:
                    assert len(tool_cfg) == 1, "Only one tool is supported for now, but got: {}".format(tool_cfg)

                    api_name = tool_cfg[0].get("API_name", tool_cfg[0].get("api_name", ""))

                    api_params = tool_cfg[0].get("api_params", tool_cfg[0].get("API_params", {}))
                    target_image_id = api_params.get("image", "img_last")
                    
                    image = self._get_image_by_id(target_image_id, item)
                    
                    
                    api_params.pop('image', None)
                    
                    api_paras = {
                        "box_threshold": 0.3,
                        "text_threshold": 0.25,
                        **api_params,
                    }

                    if image:
                        api_paras['image'] = pil_to_base64(image)
                    else:
                        raise ValueError(f"Could not load image for ID: {target_image_id}")
                    
                    
                    tool_response = self.tool_manager.call_tool(api_name, api_paras)
                    tool_response_clone = copy.deepcopy(tool_response)

                    if tool_response['error_code'] == 0:
                        logger.info(f"The {api_name} calls successfully!")
                        
                        if "edited_image" in tool_response:
                            current_img_id = f"img_round_{len(item.tool_response)}"
                            
                            original_text = tool_response.get("text", "")
                            tool_response["text"] = f"{original_text}\n[Output Image ID: {current_img_id}]"
                            
                            if "text" in tool_response_clone:
                                tool_response_clone["text"] = tool_response["text"]
                        # =================================
                        
                    else:
                        logger.info(f"The {api_name} calls failed!")
                    
                    item.tool_response.append(tool_response_clone)
                    continue
                    # return tool_response_clone
                except:
                    logger.info(f"Tool {api_name} failed to answer the question, tool_cfg is {tool_cfg}")
                    item.tool_response.append(dict(text=f"Tool {api_name} failed to answer the question.",error_code=1))
                    continue
                    # return dict(text=f"Tool {api_name} failed to answer the question.")
            else:
                item.tool_response.append(None)
                
                continue
            
    def extract_actions(self, text: str):
        import json
        import re
        import ast

        if not text:
            return None

        def robust_parse(s):
            try: return json.loads(s)
            except: pass
            try: return ast.literal_eval(s)
            except: pass
            return None

        def clean_markdown(s):
            """Strip markdown code block markers from a string."""
            s = s.strip()
            if s.startswith("```json"): s = s[7:]
            elif s.startswith("```"): s = s[3:]
            if s.endswith("```"): s = s[:-3]
            return s.strip()

        actions = []

        # ======================================================
        # ======================================================
        xml_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        xml_matches = re.findall(xml_pattern, text, re.DOTALL)
        
        if xml_matches:
            print(f"[DEBUG-EXTRACT] Found {len(xml_matches)} <tool_call> tags.")
            for match_content in xml_matches:
                clean_content = clean_markdown(match_content)
                obj = robust_parse(clean_content)
                if isinstance(obj, dict) and "name" in obj:
                    actions.append(obj)
                elif isinstance(obj, list):
                    actions.extend(obj)
            
            if actions:
                print(f"[DEBUG-EXTRACT] Successfully extracted actions from XML: {actions}")
                return actions

        # ======================================================
        # ======================================================
        clean_text = clean_markdown(text)
        
        start_pattern = r'[\"\']actions[\"\']\s*:\s*\['
        match = re.search(start_pattern, clean_text)
        
        if match:
            start_idx = match.end() - 1 
            stack = 0
            end_idx = -1
            
            for i in range(start_idx, len(clean_text)):
                char = clean_text[i]
                if char == '[': stack += 1
                elif char == ']': 
                    stack -= 1
                    if stack == 0:
                        end_idx = i
                        break
            
            if end_idx != -1:
                candidate = clean_text[start_idx : end_idx + 1]
                res = robust_parse(candidate)
                if res and isinstance(res, list):
                    print(f"[DEBUG-EXTRACT] Success via 'actions':[] list extraction.")
                    return res

        # ======================================================
        # ======================================================
        full_obj = robust_parse(clean_text)
        if isinstance(full_obj, dict):
            if "actions" in full_obj and isinstance(full_obj["actions"], list):
                return full_obj["actions"]
            if "name" in full_obj:
                return [full_obj]
        elif isinstance(full_obj, list):
            return full_obj

        # ======================================================
        # ======================================================
        try:
            fallback_pattern = r'\{\s*[\"\']name[\"\']\s*:\s*[\"\'](\w+)[\"\']\s*,\s*[\"\']arguments[\"\']\s*:\s*\{(.*?)\}\s*\}'
            matches = re.findall(fallback_pattern, clean_text, re.DOTALL)
            if matches:
                print("[DEBUG-EXTRACT] Attempting regex fallback...")
                pass
        except:
            pass

        print("[DEBUG-EXTRACT] All extraction methods failed.")
        return None
       
    def batch_parse_tool_config(self):
        print("\n" + "="*20 + " ENTERING BATCH PARSE " + "="*20)
        current_batch = self.manager.get_current_batch()
        
        for item in current_batch:
            idx = item.meta_data.get('idx', 'unknown')
            # print(f"[DEBUG] Processing Item: {idx}")
            
            tool_cfg = None 

            if len(item.model_response) == 0:
                item.tool_cfg.append(None)
                continue

            model_response = item.model_response[-1]
            
            if item.status != "processing":
                item.tool_cfg.append(None)
                continue

            try:
                # -------------------------------------------------
                # -------------------------------------------------
                actions = self.extract_actions(model_response)
                print(f"[DEBUG] Extracted Actions for {idx}: {actions}")

                if actions is not None and len(actions) > 0:
                    action = actions[0]
                    
                    if not isinstance(action, dict):
                        print(f"[DEBUG] ERROR: Action is not a dict: {type(action)}")
                        tool_cfg = None
                    else:
                        # -------------------------------------------------
                        # -------------------------------------------------
                        action_name = action.get('name')
                        if not action_name: action_name = action.get('API_name')
                        
                        arguments = action.get('arguments', {})
                        if not arguments and 'API_params' in action:
                            arguments = action['API_params']

                        print(f"[DEBUG] Action: {action_name}, Args: {arguments}")

                        target_image = arguments.get('image', 'img_last')
                        
                        core_param = (
                            arguments.get('param') or 
                            arguments.get('ans') or 
                            arguments.get('answer') or 
                            arguments.get('text')
                        )

                        if action_name == "OCR":
                            tool_cfg = [{'API_name': 'OCR',
                                        'API_params': {
                                            'image': target_image,
                                            'param': core_param
                                        }}]
                        
                        elif action_name == "Terminate":
                            tool_cfg = [{
                                'API_name': 'Terminate',
                                'API_params': {
                                    'ans': core_param,
                                    'param': core_param
                                }
                            }]
                        
                        else:
                            tool_cfg = [{
                                'API_name': action_name,
                                'API_params': {
                                    'image': target_image,
                                    'param': core_param
                                }
                            }]
                        
                        print(f"[DEBUG] Constructed tool_cfg: {tool_cfg}")
                else:
                    print(f"[DEBUG] Actions list is empty.")
                    tool_cfg = None

            except Exception as e:
                print(f"[DEBUG] !!! EXCEPTION DURING PARSING !!! {str(e)}")
                import traceback
                traceback.print_exc()
                tool_cfg = None

            item.tool_cfg.append(tool_cfg)
        
        print("="*20 + " EXITING BATCH PARSE " + "="*20 + "\n")

    ## Batch Inference
    def batch_inference(self, dataset, result_callback=None, finished_idxs=None, target_idxs=None):
        if finished_idxs is None: finished_idxs = set()
        
        self.dataset = dataset
        g = torch.Generator()
        g.manual_seed(0)

        self.dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            num_workers=2, 
            shuffle=True,
            generator=g,
            collate_fn=lambda x: x[0]
        )

        if dist.is_initialized() and not 'vllm_models' in str(type(self.tp_model)):
            self.dataloader = self.accelerator.prepare(self.dataloader)
        
        self.tp_model.eval()

        if target_idxs is not None:
             effective_target = target_idxs - finished_idxs
             remaining_len = len(effective_target)
             desc_str = f"Agent Inference (Targeting {len(target_idxs)} errors)"
        else:
             total_len = len(self.dataloader)
             remaining_len = max(0, total_len - len(finished_idxs))
             desc_str = "Agent Inference"

        progress_bar = tqdm_rank0(remaining_len, desc=desc_str)

        if len(self.dataloader) == 0:
            return

        # =================================================================
        # =================================================================
        
        for item_data in self.dataloader:
            idx = item_data.get("idx")
            
            if idx in finished_idxs:
                continue
            if target_idxs is not None and idx not in target_idxs:
                continue
            # --------------------------------
            
            max_attempts = 8
            success = False
            
            for attempt in range(max_attempts):
                try:
                    self.manager.drop_all_items()
                    
                    self.manager.append_item(item_data)
                    
                    while len(self.manager.get_current_batch()) > 0:
                        
                        current_batch = self.manager.get_current_batch()
                        
                        self.tp_model.generate(current_batch)
                        
                        for item in current_batch:
                            if not item.model_response or not item.model_response[-1].strip():
                                raise RuntimeError(f"Empty model response for {item.meta_data['idx']}")

                        self.manager.update_item_status()
                        
                        finished_items = self.manager.pop_qualified_items()
                        
                        if finished_items:
                            for res in finished_items:
                                self.dataset.store_results(dict(idx=idx, results=res))
                                if result_callback:
                                    result_callback(dict(idx=idx, results=res))
                                
                                success = True
                        
                        if success:
                            break

                        self.batch_parse_tool_config()
                        self.batch_get_tool_response()
                        self.batch_tool_response_to_next_round_input()

                    # End While
                    
                    if success:
                        progress_bar.update(1)
                        break # Break Retry Loop (Attempt)

                except Exception as e:
                    logger.warning(f"[Retry {attempt+1}/{max_attempts}] Error processing {idx}: {str(e)}")
                    
                    if is_rate_limit_error(e):
                        logger.error("Rate limit hit! Sleeping 60s...")
                        time.sleep(60)
                    
                    if attempt == max_attempts - 1:
                        logger.error(f"Failed to process {idx} after {max_attempts} attempts.")
                        error_res = {
                            "meta_data": item_data,
                            "final_answer": "Error: Max retries exceeded",
                            "status": "error",
                            "error": str(e)
                        }
                        self.dataset.store_results(dict(idx=idx, results=error_res))
                        if result_callback:
                            result_callback(dict(idx=idx, results=error_res))
                        progress_bar.update(1)
                    else:
                        backoff_sleep(attempt)
                        # Continue to next attempt...

        # End For Dataset
        
        if not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()