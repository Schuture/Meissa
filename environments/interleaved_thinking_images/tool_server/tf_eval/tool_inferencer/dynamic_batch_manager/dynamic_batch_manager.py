from dataclasses import dataclass, field, asdict
from typing import Dict, Sequence, Optional,List
from tool_server.tf_eval.utils.log_utils import get_logger
from ...utils.utils import *
from PIL import Image

logger = get_logger(__name__)


direct_system_prompt = """You are a helpful visual assistant.
Given an image and a question, you must answer following this strict format:

Thought: first, analyze the image and reason about the question step by step.
Final Answer: then, provide the concise final answer. The answer must be EXTREMELY concise (a single word or a short phrase). Do not write full sentences.

Examples:
Thought: ... reasoning ...
Final Answer: lung

Thought: ... reasoning ...
Final Answer: yes

Do not use any tools. Ensure you strictly follow the "Thought:" and "Final Answer:" format.
"""

online_system_prompt_old = """
[BEGIN OF GOAL]
You are a visual assistant for medical images. Given an image and a question, decide whether to use tools to help you answer.
You must output a JSON object with fields "thought" and "actions".
You may call tools when they are helpful for visual understanding, localization, or segmentation.
If tools are not helpful, leave "actions" empty.

IMAGE REFERENCE PROTOCOL:
- "img_original": The initial full-resolution input image.
- "img_last": The output image from the immediate previous step (default).
- "img_round_N": The output image from a specific past step N (e.g., "img_round_0").
The system will explicitly tell you the ID of the generated image in the Observation (e.g., "[Output Image ID: img_round_0]").
[END OF GOAL]

[BEGIN OF ACTIONS]

Name: ZoomInSubfigure
Description: Crops the image to a specific region to see visual details clearly. Useful when the question refers to a local region but the image is too large or cluttered.
Arguments: {
  'image': 'The image identifier to operate on. Use "img_last" for the result of the previous step (default), "img_original" for the initial full image, or "img_round_N" (e.g., "img_round_0") for a specific past result.',
  'param': 'the bounding box coordinates as a list [x1, y1, x2, y2].'
}
Returns: {
  'image': 'the cropped subfigure image.'
}
Examples:
{"name": "ZoomInSubfigure", "arguments": {"image": "img_original", "param": "[100, 200, 500, 600]"}}


Name: SegmentRegionAroundPoint
Description: Segments a specific object or region around given point coordinates. 
This tool should be used ONLY when the location of interest is known or can be precisely specified by coordinates.
Arguments: {
  'image': 'The image identifier. Use "img_last" (default), "img_original", or "img_round_N".',
  'param': 'the coordinates x="value" y="value" (0-1000 scale, where 500 is the center).'
}
Returns: {
  'image': 'the image with the segmented region mask overlay.'
}
Examples:
{"name": "SegmentRegionAroundPoint", "arguments": {"image": "img_last", "param": "x=\"215\" y=\"285\""}}


Name: BioMedParseTextSeg
Description: Performs text-guided semantic segmentation on medical images.
This tool is especially useful for identifying and localizing semantic medical entities such as:
- neoplastic cells
- inflammatory cells
- tumor tissue
- normal tissue
- pathological structures

Use this tool when the question asks about the presence, location, extent, or appearance of a medically meaningful region or cell type, and precise coordinates are NOT given.

Arguments: {
  'image': 'The image identifier. Use "img_last" (default), "img_original", or "img_round_N".',
  'param': 'a text description of the target(s) to segment, e.g. "neoplastic cells; inflammatory cells". It must be a semicolon-separated list of short noun phrases (each <= 6 words). Do not pass long full-sentence descriptions.'
}
Returns: {
  'image': 'the image with segmentation mask overlays for the queried targets.'
}
Examples:
{"name": "BioMedParseTextSeg", "arguments": {"image": "img_last", "param": "neoplastic cells"}}
{"name": "BioMedParseTextSeg", "arguments": {"image": "img_round_0", "param": "tumor tissue; inflammatory cells"}}


Name: Terminate
Description: Concludes the task and provides the final answer. This tool must be used to finalize the response.
Arguments: {
  'ans': 'the final answer to the question being addressed.'
}
Returns: {
  'ans': 'the finalized short-form answer.'
}
Examples:
{"name": "Terminate", "arguments": {"ans": "Yes."}}

[END OF ACTIONS]


[BEGIN OF TASK INSTRUCTIONS]
1. Only select actions from ACTIONS.
2. Call at most one action at a time.
3. Prefer BioMedParseTextSeg for semantic medical targets (cells, tissues, lesions).
4. Use SegmentRegionAroundPoint only when a specific point or coordinate is clearly known.
5. If no action is needed, output "actions": [].
6. Always finish by calling Terminate with the final answer.
7. YOUR OUTPUT MUST BE VALID JSON. Do NOT wrap it in markdown like ```json ... ```.
[END OF TASK INSTRUCTIONS]


[BEGIN OF FORMAT INSTRUCTIONS]
Your output must be in strict JSON format:
{
  "thought": "detailed reasoning, planning, and reflection process",
  "actions": [
    {
      "name": "action_name",
      "arguments": {
        "argument1": "value1"
      }
    }
  ]
}
[END OF FORMAT INSTRUCTIONS]
"""


online_system_prompt = """
[BEGIN OF GOAL]
You are a visual assistant for medical images. Given an image and a question, decide whether to use tools to help you answer.
You possess an internal chain of thought. Before taking any action (calling a tool) or giving a final answer, you MUST enclose your reasoning, planning, and reflection process within <think> and </think> tags.

If you decide to use a tool, output the tool call strictly enclosed within <tool_call> and </tool_call> tags. The content inside must be a valid JSON object representing the action.

IMAGE REFERENCE PROTOCOL:
- "img_original": The initial full-resolution input image.
- "img_last": The output image from the immediate previous step (default).
- "img_round_N": The output image from a specific past step N (e.g., "img_round_0").
[END OF GOAL]

[BEGIN OF ACTIONS]

Name: ZoomInSubfigure
Description: Crops the image to a specific region to see visual details clearly. Useful when the question refers to a local region but the image is too large or cluttered.
Arguments: {
  'image': 'The image identifier to operate on. Use "img_last" for the result of the previous step (default), "img_original" for the initial full image, or "img_round_N" (e.g., "img_round_0") for a specific past result.',
  'param': 'the bounding box coordinates as a list [x1, y1, x2, y2], using a 0–1000 normalized coordinate system where (0,0) is the top-left and (1000,1000) is the bottom-right of the image.'
}
Returns: {
  'image': 'the cropped subfigure image.'
}
Examples:
{"name": "ZoomInSubfigure", "arguments": {"image": "img_original", "param": "[100, 200, 500, 600]"}}


Name: SegmentRegionAroundPoint
Description: Segments a specific object or region around given point coordinates. 
This tool should be used ONLY when the location of interest is known or can be precisely specified by coordinates.
Arguments: {
  'image': 'The image identifier. Use "img_last" (default), "img_original", or "img_round_N".',
  'param': 'the coordinates x="value" y="value" (0-1000 scale, where 500 is the center).'
}
Returns: {
  'image': 'the image with the segmented region mask overlay.'
}
Examples:
{"name": "SegmentRegionAroundPoint", "arguments": {"image": "img_last", "param": "x=\"215\" y=\"285\""}}


Name: BioMedParseTextSeg
Description: Performs text-guided semantic segmentation on medical images.
This tool is especially useful for identifying and localizing semantic medical entities such as:
- neoplastic cells
- inflammatory cells
- tumor tissue
- normal tissue
- pathological structures

Use this tool when the question asks about the presence, location, extent, or appearance of a medically meaningful region or cell type, and precise coordinates are NOT given.

Arguments: {
  'image': 'The image identifier. Use "img_last" (default), "img_original", or "img_round_N".',
  'param': 'a text description of the target(s) to segment, e.g. "neoplastic cells; inflammatory cells". It must be a semicolon-separated list of short noun phrases (each <= 6 words). Do not pass long full-sentence descriptions.'
}
Returns: {
  'image': 'the image with segmentation mask overlays for the queried targets.'
}
Examples:
{"name": "BioMedParseTextSeg", "arguments": {"image": "img_last", "param": "neoplastic cells"}}
{"name": "BioMedParseTextSeg", "arguments": {"image": "img_round_0", "param": "tumor tissue; inflammatory cells"}}


Name: Terminate
Description: Concludes the task and provides the final answer. This tool must be used to finalize the response.

Output constraints (very important):
- The value of "ans" must be the final answer ONLY.
- Keep it short: usually 1–6 words for open-ended questions, or exactly "Yes" / "No" for yes-no questions.
- Do NOT add any explanation, justification, or extra sentence.
- Do NOT add parentheses, synonyms, or multiple alternatives.
- Do NOT add punctuation at the end (no ".", ",", ";", ":").
- If the expected answer is a list, use a short comma-separated list with no extra words.

Arguments: {
  'ans': 'A short final answer string that follows the constraints above.'
}

Returns: {
  'ans': 'The finalized short-form answer.'
}

Examples:
{"name": "Terminate", "arguments": {"ans": "Yes"}}
{"name": "Terminate", "arguments": {"ans": "fat necrosis"}}
{"name": "Terminate", "arguments": {"ans": "canals of Hering"}}
{"name": "Terminate", "arguments": {"ans": "bile duct cells, canals of Hering"}}

[END OF ACTIONS]

[BEGIN OF FORMAT INSTRUCTIONS]
1. Reasoning first:
<think>
Your detailed reasoning here...
</think>

2. Then, if you need to call a tool (e.g., BioMedParseTextSeg, Terminate):
<tool_call>
{"name": "BioMedParseTextSeg", "arguments": {"image": "img_last", "param": "tumor"}}
</tool_call>

3. You MUST output exactly one of USE_TOOL or NO_TOOL at the very first tool-call thinking process.
[END OF FORMAT INSTRUCTIONS]
"""


@dataclass
class DynamicBatchItem:
    max_rounds: int
    current_round : int
    status: str = "pending" # pending, processing, finished
    meta_data: Dict = field(default = None)
    conversation: object = field(default = None)
    model_response: List[str] = field(default_factory=list)
    tool_cfg :  List[str] = field(default_factory=list)
    tool_response :  List[str] = field(default_factory=list)
    new_round_input :  List[str] = field(default_factory=list)
    current_image : Image = field(default=None)


class DynamicBatchManager():
    def __init__(
        self,
        batch_size: int,
        stop_token: str = "<stop>",
        max_rounds: int = 3,
        generate_conversation_fn = None,
    ):
        self.dynamic_batch = []
        self.batch_size = batch_size
        self.stop_token = stop_token
        self.max_rounds = max_rounds
        self.generate_conversation_fn = generate_conversation_fn
    
    def extract_final_answer(self, final_response: str):
        """
        If the model returns the framework's Terminate JSON, parse ans from it.
        Otherwise, fall back to returning the raw text.
        """
        if final_response is None:
            return ""
        final_response = str(final_response)

        if self.max_rounds == 0:
            return final_response.strip()

        res_prefix = "{\"name\": \"Terminate\", \"arguments\": {\"ans\":"
        res_postfix = "}"

        if res_prefix not in final_response:
            # Early-stop case (no tool call, no Terminate): just return raw response
            return final_response.strip()

        temp = final_response.split(res_prefix, 1)[-1].strip()
        res = temp.split(res_postfix, 1)[0].strip()
        return res

        
    def pop_qualified_items(self):
        res = []
        new_batch = []
        for idx,item in enumerate(self.dynamic_batch):
            if item.status == "finished":
                item = asdict(item)
                item = remove_pil_objects(item)
                
                final_model_output = item["model_response"][-1]
                final_answer = self.extract_final_answer(final_model_output)
                item["final_answer"] = final_answer
                
                res.append(item)
            else:
                new_batch.append(item)
        self.dynamic_batch = new_batch
        return res
    
    def append_item(self, meta_data: Dict):
        if len(self.dynamic_batch) < self.batch_size:
            candidate_item = DynamicBatchItem(
                max_rounds=self.max_rounds,
                current_round=0,
                meta_data=meta_data,
                status="pending"
            )
            conv = self.generate_conversation_fn(
                text=meta_data["text"],
                image=meta_data["image"],
                role="user",
            )

            if self.max_rounds == 0:
                system_msg = {"role": "system", "content": direct_system_prompt}
            else:
                system_msg = {"role": "system", "content": online_system_prompt}

            if isinstance(conv, list):
                if len(conv) == 0 or conv[0].get("role") != "system":
                    conv = [system_msg] + conv
            else:
                # fallback: wrap into list
                conv = [system_msg, {"role": "user", "content": conv}]

            candidate_item.conversation = conv
            
            self.dynamic_batch.append(candidate_item)
        else:
            raise ValueError("Batch is full")
    

    def append_item_to_full(self, dataloader, progress_bar=None, finished_idxs=None, target_idxs=None):
        while len(self.dynamic_batch) < self.batch_size:
            try:
                item_data = next(dataloader)
                idx = item_data.get("idx")
                
                if finished_idxs is not None and idx in finished_idxs:
                    continue
                
                if target_idxs is not None:
                    if idx not in target_idxs:
                        continue

                self.append_item(item_data)
                if progress_bar:
                    progress_bar.update(1)
            except StopIteration:
                break
            except Exception as e:
                logger.exception(f"append_item failed, skipping one sample: {e}")
                continue

        
    def drop_item_by_idx(self, bad_idx: str):
        """Remove items with meta_data['idx'] == bad_idx from dynamic_batch."""
        new_batch = []
        for it in self.dynamic_batch:
            if it.meta_data and it.meta_data.get("idx") == bad_idx:
                continue
            new_batch.append(it)
        self.dynamic_batch = new_batch


    def drop_all_items(self):
        """Clear current dynamic batch."""
        self.dynamic_batch = []


    def get_current_batch(self):
        return self.dynamic_batch
    
    
    # Caution: Only model.generate can call this function
    def update_item_status(self):
        for item in self.dynamic_batch:
            if item.status == "pending":
                if item.current_round == item.max_rounds or "Terminate" in item.model_response[-1]:
                    item.status = "finished"
                else:
                    item.current_round += 1
                    item.status = "processing"
            elif item.status == "processing":
                if item.current_round == item.max_rounds or "Terminate" in item.model_response[-1]:
                    item.status = "finished"
                else:
                    item.current_round += 1
            elif item.status == "finished":
                pass
            else:
                raise ValueError(f"Invalid status {item.status}")
