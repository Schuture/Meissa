from .abstract_model import tp_model
import uuid
import requests
import time
import json
import base64
import os
import random
from io import BytesIO
from PIL import Image
from typing import List

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem
from ..utils.log_utils import get_logger

logger = get_logger(__name__)


GEMINI_SYSTEM_PROMPT_FORWARD = """
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

[BEGIN OF TASK INSTRUCTIONS]
1. Only select actions from ACTIONS.
2. Call at most one action at a time.
3. Prefer BioMedParseTextSeg for semantic medical targets (cells, tissues, lesions).
4. Use SegmentRegionAroundPoint only when a specific point or coordinate is clearly known.
5. Always finish by calling Terminate with the final answer.
6. YOUR OUTPUT MUST BE VALID JSON. Do NOT wrap it in markdown like ```json ... ```.
7. If an action (other than Terminate) is called, the next round must first evaluate the Observation before deciding the next step.
8. The final answer will be evaluated mainly by string match; extra words can make a correct answer be judged incorrect.
9. When you call Terminate, output only the minimal answer string in "ans".
[END OF TASK INSTRUCTIONS]


[BEGIN OF FORMAT INSTRUCTIONS]
Your output must be in strict JSON format:
{
  "thought": "brief recap and decision rationale for the next step; keep concise",
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

GEMINI_SYSTEM_PROMPT_BACKWARD = """
[BEGIN OF GOAL]
You are a visual assistant for medical images. Given an image and a question, decide whether to use tools to help you answer.

You must output a JSON object with field "actions".
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

[BEGIN OF TASK INSTRUCTIONS]
1. Only select actions from ACTIONS.
2. Call at most one action at a time.
3. Prefer BioMedParseTextSeg for semantic medical targets (cells, tissues, lesions).
4. Use SegmentRegionAroundPoint only when a specific point or coordinate is clearly known.
5. Always finish by calling Terminate with the final answer.
6. YOUR OUTPUT MUST BE VALID JSON. Do NOT wrap it in markdown.
7. If an action (other than Terminate) is called, the next round must first evaluate the Observation before deciding the next step.
8. The final answer will be evaluated mainly by string match; extra words can make a correct answer be judged incorrect.
9. When you call Terminate, output only the minimal answer string in "ans".
[END OF TASK INSTRUCTIONS]

[BEGIN OF RECAP INSTRUCTIONS]
You MUST include a "recap" field ONLY in the same JSON object where you call Terminate.

The recap is a hindsight evidence and decision summary.
It is NOT a planning trace.

The "recap" field must be a list of objects in chronological order.
Each action taken (including the final Terminate) must have exactly one recap entry.

Schema:

For tool actions:
{
  "step": <integer>,
  "tool": "<tool_name>",
  "why": "<short reason for calling the tool>",
  "got": "<short finding from the tool output>",
  "update": "increase" | "decrease" | "no_change",
  "evidence": "<what in the observation supports the finding>",
  "inference": "<short inference if needed, otherwise empty string>",
  "confidence": <integer 0-100>
}

For the final Terminate action:
{
  "step": <integer>,
  "tool": "Terminate",
  "why": "<short reason why the model can now answer and stop>"
}

Rules:
- Include one recap entry for EVERY action, including Terminate.
- Recap entries must follow the exact chronological order of actions.
- For Terminate, include ONLY "step", "tool", and "why".
  Do NOT include got / update / evidence / inference / confidence.
- Keep all fields concise.
- "evidence" must be grounded in tool observations, not speculation.
- Do NOT include planning, hypotheses, or alternative branches.
- If no tools were called, recap must contain exactly ONE entry for Terminate.
[END OF RECAP INSTRUCTIONS]

[BEGIN OF FORMAT INSTRUCTIONS]
Your output must be in strict JSON format.

In non-final rounds:
{
  "actions": [
    {
      "name": "action_name",
      "arguments": {
        "argument1": "value1"
      }
    }
  ]
}

In the final round (calling Terminate):
{
  "actions": [
    {
      "name": "Terminate",
      "arguments": {
        "ans": "final answer"
      }
    }
  ],
  "recap": [
    {
      "step": 1,
      "tool": "BioMedParseTextSeg",
      "why": "...",
      "got": "...",
      "update": "increase",
      "evidence": "...",
      "inference": "",
      "confidence": 78
    },
    {
      "step": 2,
      "tool": "Terminate",
      "why": "tool evidence is sufficient to answer confidently"
    }
  ]
}
[END OF FORMAT INSTRUCTIONS]
"""




class GeminiModels(tp_model):
    def __init__(
      self,  
      model_name: str = "gemini-1.5-flash",
      max_retry: int = 5,
      temperature: float = 0.0,
      api_key: str = None
    ):
        self.model_name = model_name
        self.max_retry = int(max_retry) if max_retry else 5
        self.temperature = float(temperature) if temperature is not None else 0.0
        
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in env or args.")
            
        genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel(self.model_name)
        
        self.generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=4096,
            temperature=self.temperature,
            response_mime_type="application/json" 
        )
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def to(self, *args, **kwargs):
        pass

    def eval(self):
        pass
    
    def set_generation_config(self, config):
        pass

    def getitem_fn(self, meta_data, idx):
        item = meta_data[idx]
        image_path = item.get("image_path")
        if not image_path:
             image_path = item.get("image_file") or item.get("file_name")
        image = Image.open(image_path).convert("RGB") if image_path else None
        text = item["text"]
        item_idx = item["idx"]
        return dict(image=image, text=text, idx=item_idx)

    def generate_conversation_fn(self, text, image, role="user"):  
        messages = [
            {
                "role": "system",
                # "content": GEMINI_SYSTEM_PROMPT_FORWARD,
                "content": GEMINI_SYSTEM_PROMPT_BACKWARD
            }
        ]
        content_list = [{"type": "text", "text": text}]
        if image:
            content_list.append({"type": "image", "image": image})
            
        messages.append({
            "role": role,
            "content": content_list
        })
        return messages
    
    def append_conversation_fn(self, conversation, text, image, role):
        new_content = [{"type": "text", "text": text}]
        if image:
            if isinstance(image, Image.Image):
                new_content.append({"type": "image", "image": image})
            elif isinstance(image, str):
                try:
                    if "," in image: image = image.split(",")[1]
                    pil_img = Image.open(BytesIO(base64.b64decode(image)))
                    new_content.append({"type": "image", "image": pil_img})
                except:
                    pass
        conversation.append({
            "role": role,
            "content": new_content,
        })
        return conversation

    def _convert_messages_to_gemini_format(self, messages):
        gemini_contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg["role"]
            content_data = msg["content"]
            parts = []
            
            if isinstance(content_data, str):
                parts.append(content_data)
            elif isinstance(content_data, list):
                for item in content_data:
                    if item.get("type") == "text":
                        parts.append(item["text"])
                    elif item.get("type") == "image":
                        pil_img = item.get("image")
                        if pil_img:
                            parts.append(pil_img)
            
            if role == "system":
                if parts: system_instruction = parts[0]
            elif role == "user":
                gemini_contents.append({"role": "user", "parts": parts})
            elif role == "assistant" or role == "function_call":
                 gemini_contents.append({"role": "model", "parts": parts})
            elif role == "observation":
                 gemini_contents.append({"role": "user", "parts": parts})
                 
        return gemini_contents, system_instruction

    def generate(self, batch: List[DynamicBatchItem]):
        if not batch: return

        item = batch[0]
        messages = item.conversation
        
        gemini_contents, system_inst = self._convert_messages_to_gemini_format(messages)
        
        if system_inst and gemini_contents:
            if gemini_contents[0]["role"] == "user":
                 gemini_contents[0]["parts"].insert(0, system_inst + "\n\n")

        fail_times = 0
        base_sleeptime = 5
        
        while fail_times < self.max_retry:
            try:
                response = self.model.generate_content(
                    gemini_contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                final_text = response.text.strip()
                
                if final_text.startswith("```json"): final_text = final_text[7:]
                if final_text.endswith("```"): final_text = final_text[:-3]
                
                item.model_response.append(final_text)
                self.append_conversation_fn(item.conversation, final_text, None, "assistant")
                break

            except Exception as e:
                logger.error(f"Gemini API Error: {e}. Retrying {fail_times+1}/{self.max_retry}...")
                fail_times += 1
                time.sleep(fail_times * base_sleeptime)

        if fail_times >= self.max_retry:
            logger.error("Gemini failed after max retries.")
            item.model_response.append("")