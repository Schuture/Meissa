import os
import warnings
import argparse
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from agent import Agent
from tools import (
    ChestXRayClassifierTool,
    ChestXRaySegmentationTool,
    ChestXRayReportGeneratorTool,
    ImageVisualizerTool,
    DicomProcessorTool,
    LlavaMedTool,
    XRayVQATool,
    XRayPhraseGroundingTool,
)
from tools.utils import load_prompts_from_file

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/model-weights",
    temp_dir="temp",
    device="cuda",
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    temperature=0.2,
    top_p=0.95,
):
    """Initialize the Continuous Tool Calling agent with specified tools.

    Args:
        prompt_file: Path to system prompt file.
        tools_to_use: List of tool class names to enable. None enables all tools.
        model_dir: Directory containing specialist model weights.
        temp_dir: Directory for temporary files.
        device: Torch device string ('cuda' or 'cpu').
        model_name: Model name or path served via OpenAI-compatible API.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.

    Returns:
        Tuple[Agent, dict]: Initialized agent and tool instance dictionary.
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT_DISTILLATION_SFT"]

    # XRayVQATool and XRayPhraseGroundingTool require large models that cannot coexist
    # with vLLM on the same GPU. When CHEXAGENT_SERVER_URL / MAIRA2_SERVER_URL are set,
    # use MedRAX remote tool variants (HTTP calls to standalone servers) instead.
    _chexagent_url = os.getenv("CHEXAGENT_SERVER_URL", "")
    _maira2_url = os.getenv("MAIRA2_SERVER_URL", "")

    if _chexagent_url or _maira2_url:
        import sys as _sys
        # remote_tools.py is in the same directory as this file
        _script_dir_rt = os.path.dirname(os.path.abspath(__file__))
        if _script_dir_rt not in _sys.path:
            _sys.path.insert(0, _script_dir_rt)
        from remote_tools import (
            XRayVQATool as _RemoteXRayVQATool,
            XRayPhraseGroundingTool as _RemoteXRayPhraseGroundingTool,
        )
        _xray_vqa_factory = lambda: _RemoteXRayVQATool()
        _xray_grounding_factory = lambda: _RemoteXRayPhraseGroundingTool()
    else:
        _xray_vqa_factory = lambda: XRayVQATool(cache_dir=model_dir, device=device)
        _xray_grounding_factory = lambda: XRayPhraseGroundingTool(
            cache_dir=model_dir, temp_dir=temp_dir, device=device
        )

    all_tools = {
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device),
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=True),
        "XRayVQATool": _xray_vqa_factory,
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(
            cache_dir=model_dir, device=device
        ),
        "XRayPhraseGroundingTool": _xray_grounding_factory,
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),
    }

    tools_dict = {}
    tools_to_use = tools_to_use or list(all_tools.keys())
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=MemorySaver(),
    )
    return agent, tools_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meissa: Continuous Tool Calling demo")
    parser.add_argument("--model", default=os.getenv("MEISSA_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
                        help="Model name served via OpenAI-compatible API")
    parser.add_argument("--model_dir", default="/model-weights",
                        help="Directory containing specialist model weights")
    parser.add_argument("--temp_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8585)
    parser.add_argument("--demo", action="store_true", help="Run a quick CLI demo instead of Gradio UI")
    parser.add_argument(
        "--tools",
        default=None,
        help=(
            "Comma-separated list of tools to load. "
            "Defaults to all tools in normal mode; "
            "defaults to 'ImageVisualizerTool,DicomProcessorTool' in --demo mode "
            "to avoid loading specialist models that require external weights."
        ),
    )
    args = parser.parse_args()

    # In demo mode, use the full MedRAX pipeline tools (same as training inference).
    # XRayVQATool requires CHEXAGENT_SERVER_URL; XRayPhraseGroundingTool requires MAIRA2_SERVER_URL.
    # ChestXRayClassifierTool and ChestXRayReportGeneratorTool work without external servers.
    tools_to_use = None  # None = all tools
    if args.tools:
        tools_to_use = [t.strip() for t in args.tools.split(",")]
    elif args.demo:
        tools_to_use = [
            "ChestXRayClassifierTool",
            "ChestXRayReportGeneratorTool",
            "XRayVQATool",
            "XRayPhraseGroundingTool",
        ]

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    agent, tools_dict = initialize_agent(
        prompt_file=os.path.join(_script_dir, "system_prompts.txt"),
        tools_to_use=tools_to_use,
        model_dir=args.model_dir,
        temp_dir=args.temp_dir,
        device=args.device,
        model_name=args.model,
    )

    if args.demo:
        from langchain_core.messages import HumanMessage
        import glob, json as _json
        print("=== Meissa Demo: Continuous Tool Calling (Framework I) ===")
        print(f"Model: {args.model}")
        print(f"Active tools: {list(tools_dict.keys())}\n")

        # Demo uses a public pneumonia chest X-ray (from MedRAX demo/chest/).
        # Set DEMO_IMAGE_PATH env var or place the image at the default path below.
        _demo_image = os.getenv(
            "DEMO_IMAGE_PATH",
            os.path.join(_script_dir, "..", "..", "data", "demo_samples",
                         "continuous_tool_calling", "demo_cxr.jpg"),
        )
        _question_text = (
            "What abnormalities are present in this chest X-ray? "
            "Please analyze the image and provide a detailed report."
        )
        print(f"Question: {_question_text}")
        print(f"Image: {_demo_image}\n")
        config = {"configurable": {"thread_id": "demo"}}

        # Save resized image to a temp file so tools can access it by path.
        # Also embed as base64 so the model can see the image inline.
        import base64 as _base64
        import io as _io
        from PIL import Image as _PILImage
        _msg_content: list = []
        _temp_img_path = os.path.join(args.temp_dir, "demo_input.jpg")
        os.makedirs(args.temp_dir, exist_ok=True)
        if os.path.exists(_demo_image):
            _img = _PILImage.open(_demo_image).convert("RGB")
            _img.thumbnail((512, 512), _PILImage.LANCZOS)
            # Save to temp file for tool access
            _img.save(_temp_img_path, format="JPEG", quality=85)
            _buf = _io.BytesIO()
            _img.save(_buf, format="JPEG", quality=85)
            _b64 = _base64.b64encode(_buf.getvalue()).decode()
            _msg_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64}"}}
            )
            print(f"Image resized to {_img.size} and saved to {_temp_img_path}\n")
        else:
            print(f"[WARN] Image not found at {_demo_image}; sending path as text only.\n")
            _temp_img_path = _demo_image  # fallback
        # Include image path in text so the model knows the correct path for tool calls
        _msg_content.append(
            {"type": "text", "text": f"Image path: {_temp_img_path}\n{_question_text}"}
        )

        # Record existing log files before running so we can find new ones
        _log_dir = os.path.join(_script_dir, "logs")
        _existing_logs = set(glob.glob(os.path.join(_log_dir, "tool_calls_*.json")))

        # Use .stream() to display intermediate tool calls in real-time
        print("--- Agent Execution ---")
        _step = 0
        for event in agent.workflow.stream(
            {"messages": [HumanMessage(content=_msg_content)]}, config
        ):
            if "execute" in event:
                for msg in event["execute"]["messages"]:
                    _step += 1
                    _tool_name = getattr(msg, "name", "?")
                    _tool_args = getattr(msg, "args", {})
                    _tool_content = getattr(msg, "content", "")
                    # Truncate content for display
                    _display_content = _tool_content[:200] + "..." if len(str(_tool_content)) > 200 else _tool_content
                    print(f"  [Step {_step}] Tool: {_tool_name}")
                    print(f"           Args: {list(_tool_args.keys()) if isinstance(_tool_args, dict) else _tool_args}")
                    print(f"           Result: {_display_content}")
            if "process" in event:
                _last_msg = event["process"]["messages"][-1]
                if not getattr(_last_msg, "tool_calls", []):
                    # Final response (no more tool calls)
                    print(f"\nAnswer: {_last_msg.content}")
        print("--- End ---")
    else:
        from interface import create_demo
        demo = create_demo(agent, tools_dict)
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)
