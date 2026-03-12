# MedRAX/remote_tools.py
import os
from typing import Any, Dict, List, Optional, Tuple, Type

import requests
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


# -------------------------
# CheXagent 远程 VQA 工具
# -------------------------

class XRayVQAToolInput(BaseModel):
    """Input schema for the CheXagent Tool (remote)."""

    image_paths: List[str] = Field(
        ...,
        description="List of paths to chest X-ray images to analyze",
    )
    prompt: str = Field(
        ...,
        description="Question or instruction about the chest X-ray images",
    )
    max_new_tokens: int = Field(
        512,
        description="Maximum number of tokens to generate in the response",
    )


class XRayVQATool(BaseTool):
    """Remote tool that calls a CheXagent server for chest X-ray VQA."""

    # 必须保持和原来一致
    name: str = "chest_xray_expert"
    description: str = (
        "A versatile tool for analyzing chest X-rays. "
        "Can perform visual question answering, report-style descriptions, "
        "abnormality detection, comparison of studies, and clinical explanation. "
        "Input should be paths to X-ray images and a natural language prompt."
    )
    args_schema: Type[BaseModel] = XRayVQAToolInput
    return_direct: bool = True

    # 远程 server 地址从环境变量读取
    server_url: str = ""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # 不在这个环境里加载模型，只负责调 HTTP
        base_url = os.environ.get("CHEXAGENT_SERVER_URL")
        if not base_url:
            raise ValueError(
                "CHEXAGENT_SERVER_URL is not set. "
                "Please export CHEXAGENT_SERVER_URL='http://HOST:PORT'."
            )
        # 去掉末尾的斜杠，方便拼接
        self.server_url = base_url.rstrip("/")

    def _call_remote(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        payload = {
            "image_paths": image_paths,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
        resp = requests.post(
            f"{self.server_url}/vqa",
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()

    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Call remote CheXagent server and wrap response like原实现."""
        try:
            data = self._call_remote(image_paths, prompt, max_new_tokens)
            # 约定远端返回 {"response": "...", "raw": {...}} 之类结构
            output = {
                "response": data.get("response", ""),
                "raw": data,
            }
            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "completed",
                "remote_server": self.server_url,
            }
            return output, metadata
        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "failed",
                "remote_server": self.server_url,
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        return self._run(image_paths, prompt, max_new_tokens)


# -------------------------
# MAIRA-2 远程 Grounding 工具
# -------------------------

class XRayPhraseGroundingInput(BaseModel):
    """Input schema for the XRay Phrase Grounding Tool (remote)."""

    image_path: str = Field(
        ...,
        description=(
            "Path to the frontal chest X-ray image file, "
            "only supports JPG or PNG images"
        ),
    )
    phrase: str = Field(
        ...,
        description=(
            "Medical finding or condition to locate in the image "
            "(e.g., 'Pleural effusion')"
        ),
    )
    max_new_tokens: int = Field(
        default=300,
        description="Maximum number of new tokens to generate",
    )


class XRayPhraseGroundingTool(BaseTool):
    """Remote tool for phrase grounding using a MAIRA-2 server."""

    name: str = "xray_phrase_grounding"
    description: str = (
        "Locates and visualizes specific medical findings in chest X-ray images. "
        "Takes a chest X-ray image and a medical phrase to locate "
        "(e.g., 'Pleural effusion', 'Cardiomegaly'). "
        "Returns normalized bounding boxes and an optional visualization path."
    )
    args_schema: Type[BaseModel] = XRayPhraseGroundingInput
    return_direct: bool = True

    server_url: str = ""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = True,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        base_url = os.environ.get("MAIRA2_SERVER_URL")
        if not base_url:
            raise ValueError(
                "MAIRA2_SERVER_URL is not set. "
                "Please export MAIRA2_SERVER_URL='http://HOST:PORT'."
            )
        self.server_url = base_url.rstrip("/")

    def _call_remote(
        self,
        image_path: str,
        phrase: str,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        payload = {
            "image_path": image_path,
            "phrase": phrase,
            "max_new_tokens": max_new_tokens,
        }
        resp = requests.post(
            f"{self.server_url}/ground",
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()

    def _run(
        self,
        image_path: str,
        phrase: str,
        max_new_tokens: int = 300,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Call remote MAIRA-2 server and return prediction dict + metadata."""
        try:
            data = self._call_remote(image_path, phrase, max_new_tokens)
            output = {
                "predictions": data.get("predictions", []),
                "visualization_path": data.get("visualization_path"),
                "raw": data,
            }
            metadata = {
                "image_path": image_path,
                "phrase": phrase,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "completed",
                "remote_server": self.server_url,
            }
            return output, metadata
        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "image_path": image_path,
                "phrase": phrase,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "failed",
                "remote_server": self.server_url,
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(
        self,
        image_path: str,
        phrase: str,
        max_new_tokens: int = 300,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        return self._run(image_path, phrase, max_new_tokens)
