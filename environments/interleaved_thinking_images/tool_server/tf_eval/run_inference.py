"""
run_inference.py — Framework II: Interleaved Thinking with Images

Lightweight inference entry point for the Meissa model on medical VQA tasks.
Uses the original OpenThinkIMG JSON-based tool call format and supports an
offline ZoomInSubfigure fallback (no tool server required).

Usage (demo — bundled PathVQA sample, no external server needed):
    python tool_server/tf_eval/run_inference.py \\
        --model /path/to/Meissa-4B \\
        --demo

Usage (demo — with remote tool server):
    python tool_server/tf_eval/run_inference.py \\
        --model /path/to/Meissa-4B \\
        --tool_server_url http://localhost:8080 \\
        --demo

Usage (eval):
    python tool_server/tf_eval/run_inference.py \\
        --model /path/to/Meissa-4B \\
        --data_path /path/to/pathvqa_test.jsonl \\
        --image_dir /path/to/images \\
        --tool_server_url http://localhost:8080 \\
        --output results/iti_pathvqa.json

Environment variables:
    MEISSA_MODEL        Model path / HuggingFace ID (overrides --model)
    OPENAI_BASE_URL     Base URL of vLLM server (e.g. http://localhost:8001/v1)
    TOOL_SERVER_URL     Tool server URL (overrides --tool_server_url)
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# environments/interleaved_thinking_images/
_ENV_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
# Meissa/
_PROJ_ROOT = os.path.normpath(os.path.join(_ENV_ROOT, "..", ".."))

# ---------------------------------------------------------------------------
# Demo question (SLAKE sample — open access)
# Source: SLAKE dataset, idx slake_211
# Correct answer: "liver"
# ---------------------------------------------------------------------------
_DEMO_IMAGE = os.path.join(
    _PROJ_ROOT, "data", "demo_samples", "interleaved_thinking_images", "slake_demo.jpg"
)
_DEMO_QUESTION = "Where is/are the abnormality located?"


# ---------------------------------------------------------------------------
# System prompt (original OpenThinkIMG JSON-based tool call format)
# ---------------------------------------------------------------------------

def _load_system_prompt() -> str:
    """Load the original OpenThinkIMG system prompt for JSON-based tool calling."""
    prompt_path = os.path.join(_ENV_ROOT, "system_prompt.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, encoding="utf-8") as f:
            return f.read()
    logger.warning("system_prompt.txt not found at %s; using fallback prompt", prompt_path)
    return (
        "You are a visual assistant for medical images. Given an image and a question, "
        "decide whether to use tools to help you answer. Output strict JSON: "
        "{\"thought\": \"...\", \"actions\": [{\"name\": \"action_name\", \"arguments\": {...}}]}. "
        "Use ZoomInSubfigure to examine specific image regions. "
        "Always finish by calling Terminate with the final answer."
    )


# ---------------------------------------------------------------------------
# Minimal OpenAI-compatible client
# ---------------------------------------------------------------------------

def _get_base_url() -> str:
    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    if not base:
        port = os.environ.get("VLLM_PORT", "8001")
        base = f"http://127.0.0.1:{port}/v1"
    return base


def _call_model(
    model: str,
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Call model via OpenAI-compatible API.

    With --tool-call-parser hermes + --enable-auto-tool-choice, vLLM may parse
    <tool_call> blocks out of `content` and put them into `tool_calls`. We reconstruct
    them back into the returned string so the rest of the pipeline can parse normally.
    """
    base_url = _get_base_url()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'dummy')}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    content = msg.get("content") or ""

    # vLLM hermes parser moves tool calls from content into tool_calls field.
    # Reconstruct them as <tool_call>...</tool_call> so _parse_tool_call() can handle them.
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        try:
            args = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args = {}
        content = content + f'\n<tool_call>{{"name": "{name}", "arguments": {json.dumps(args)}}}</tool_call>'

    return content.strip()


def _image_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _b64_image_url(b64: str, mime: str = "image/jpeg") -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _call_tool_zoom_offline(image_b64: str, param) -> dict:
    """Call OpenThinkIMG's ZoomInSubfigure offline worker (pure PIL/OpenCV, no server)."""
    # Add the tool_server parent directory so offline_workers can be imported
    _tool_server_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    if _tool_server_root not in sys.path:
        sys.path.insert(0, _tool_server_root)
    try:
        from tool_server.tool_workers.offline_workers.zoom_in_worker import generate
        return generate({"image": image_b64, "param": param})
    except ImportError as e:
        logger.error("Failed to import zoom_in_worker from OpenThinkIMG: %s", e)
        return {"text": f"[ZoomIn unavailable: {e}]", "error_code": 1, "edited_image": None}
    except Exception as e:
        logger.error("zoom_in_worker.generate() failed: %s", e)
        return {"text": f"[ZoomIn error: {e}]", "error_code": 1, "edited_image": None}


def _call_tool_remote(tool_server_url: str, tool_name: str, arguments: dict) -> dict:
    """Call a tool on the remote tool server."""
    try:
        resp = requests.post(
            f"{tool_server_url}/call",
            json={"tool": tool_name, "arguments": arguments},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "text": data.get("result", data.get("text", "")),
            "edited_image": data.get("edited_image"),
            "error_code": 0,
        }
    except Exception as e:
        logger.warning("Remote tool call failed (%s): %s", tool_name, e)
        return {"text": f"[Tool unavailable: {e}]", "error_code": 1, "edited_image": None}


# ---------------------------------------------------------------------------
# JSON tool call parsing (original OpenThinkIMG format)
# ---------------------------------------------------------------------------

def _parse_tool_call(response: str) -> Optional[dict]:
    """Parse tool call from model response.

    Handles two formats:
    1. Qwen3 native: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. OpenThinkIMG JSON: {"thought": "...", "actions": [{"name": "...", "arguments": {...}}]}
    """
    text = response.strip()

    # --- Format 1: Qwen3 native <tool_call>...</tool_call> ---
    tc_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if tc_match:
        try:
            tc = json.loads(tc_match.group(1))
            if "name" in tc:
                # Extract thinking content from outside the tool_call tags
                think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                thought = think_match.group(1).strip() if think_match else ""
                return {
                    "thought": thought,
                    "actions": [{"name": tc["name"], "arguments": tc.get("arguments", {})}],
                }
        except json.JSONDecodeError:
            pass

    # --- Format 2: OpenThinkIMG JSON {"thought": ..., "actions": [...]} ---
    # Strip markdown code fences if present
    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if "actions" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON block containing "actions"
    match = re.search(r'\{[^{}]*"actions"[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if "actions" in data:
                return data
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Inference — JSON-based tool call loop (original OpenThinkIMG format)
# ---------------------------------------------------------------------------

def run_single(
    model: str,
    question: str,
    image_path: Optional[str] = None,
    tool_server_url: Optional[str] = None,
    max_turns: int = 8,
) -> str:
    """Run multi-turn VQA inference using the original OpenThinkIMG JSON tool call format.

    Supports offline ZoomInSubfigure via OpenThinkIMG's zoom_in_worker when no
    tool_server_url is provided. Other tools (SegmentRegionAroundPoint,
    BioMedParseTextSeg) require a remote tool server.
    """
    system_prompt = _load_system_prompt()

    # Track image states: "img_original", "img_round_N", "img_last"
    image_states: dict = {}
    original_b64: Optional[str] = None
    mime = "image/jpeg"

    if image_path and os.path.exists(image_path):
        original_b64 = _image_to_b64(image_path)
        suffix = Path(image_path).suffix.lower().lstrip(".")
        mime = f"image/{suffix if suffix in ('png', 'jpeg', 'jpg', 'gif', 'webp') else 'jpeg'}"
        image_states["img_original"] = original_b64

    # Build initial user message (text + image)
    user_content: list = [{"type": "text", "text": question}]
    if original_b64:
        user_content.append(_b64_image_url(original_b64, mime))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    last_edited_b64: Optional[str] = None

    for turn in range(max_turns):
        response = _call_model(model, messages)
        logger.info("Turn %d: %.300s", turn, response[:300])

        parsed = _parse_tool_call(response)
        if not parsed:
            if turn == 0 and original_b64:
                # Model skipped tool calls on first turn — re-prompt to force ZoomInSubfigure.
                # The system prompt requires at least one ZoomInSubfigure before Terminate.
                logger.info("Turn 0: no tool call found, re-prompting to force ZoomInSubfigure...")
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": (
                        "You MUST use ZoomInSubfigure at least once before answering. "
                        "Please call ZoomInSubfigure to examine the relevant region of the image first. "
                        "Output your tool call in the format: "
                        '<tool_call>{"name": "ZoomInSubfigure", "arguments": {"image": "img_original", "param": [x1, y1, x2, y2]}}</tool_call>'
                    ),
                })
                continue
            # No valid JSON tool call — treat as final text answer
            return response

        actions = parsed.get("actions", [])
        if not actions:
            return response

        action = actions[0]
        action_name = action.get("name", "")
        arguments = action.get("arguments", {})

        # Terminate → return the final answer
        # But the system prompt requires at least one ZoomInSubfigure before Terminate.
        # If this is the first turn and the model tries to Terminate immediately,
        # auto-zoom on the center region to give the model a closer view.
        if action_name == "Terminate":
            if turn == 0 and original_b64:
                logger.info("Turn 0: model tried Terminate without ZoomInSubfigure, auto-zooming center...")
                # Execute ZoomInSubfigure on center region automatically
                center_param = [200, 200, 800, 800]
                print(f"  [Auto-ZoomInSubfigure] image=img_original, param={center_param}")
                if tool_server_url:
                    zoom_result = _call_tool_remote(
                        tool_server_url, "ZoomInSubfigure",
                        {"image": original_b64, "param": center_param},
                    )
                else:
                    zoom_result = _call_tool_zoom_offline(original_b64, center_param)
                observation_text = zoom_result.get("text", "Zoomed into center region.")
                edited_b64 = zoom_result.get("edited_image")
                print(f"  [Observation] {observation_text}")

                if edited_b64:
                    image_states["img_round_0"] = edited_b64
                    image_states["img_last"] = edited_b64
                    last_edited_b64 = edited_b64

                # Inject the zoom observation and let the model reconsider
                messages.append({"role": "assistant", "content":
                    f'<tool_call>{{"name": "ZoomInSubfigure", "arguments": {{"image": "img_original", "param": {center_param}}}}}</tool_call>'
                })
                obs_content: list = [{"type": "text", "text": f"Observation: {observation_text}\n[Output Image ID: img_round_0]"}]
                if edited_b64:
                    obs_content.append(_b64_image_url(edited_b64, mime))
                messages.append({"role": "user", "content": obs_content})
                continue
            ans = arguments.get("ans", response)
            print(f"  [Terminate] ans={ans}")
            return str(ans)

        # Resolve image reference
        img_ref = arguments.get("image", "img_original")
        if img_ref == "img_original":
            current_b64 = image_states.get("img_original", original_b64)
        elif img_ref == "img_last":
            current_b64 = last_edited_b64 or original_b64
        elif re.match(r"img_round_\d+", img_ref):
            current_b64 = image_states.get(img_ref, last_edited_b64 or original_b64)
        else:
            current_b64 = original_b64

        param = arguments.get("param", arguments.get("bbox", [0, 0, 1000, 1000]))

        # Execute tool
        observation_text = ""
        edited_b64: Optional[str] = None

        if action_name == "ZoomInSubfigure":
            print(f"  [Tool call] ZoomInSubfigure(image={img_ref!r}, param={param})")
            if tool_server_url:
                result = _call_tool_remote(
                    tool_server_url, "ZoomInSubfigure",
                    {"image": current_b64 or "", "param": param},
                )
            else:
                result = _call_tool_zoom_offline(current_b64 or "", param)
            observation_text = result.get("text", "")
            edited_b64 = result.get("edited_image")
            print(f"  [Observation] {observation_text}")

        elif tool_server_url:
            # Other tools (SegmentRegionAroundPoint, BioMedParseTextSeg) via remote server
            print(f"  [Tool call] {action_name}(image={img_ref!r}, param={arguments.get('param', '')})")
            result = _call_tool_remote(tool_server_url, action_name, arguments)
            observation_text = result.get("text", "")
            edited_b64 = result.get("edited_image")
            print(f"  [Observation] {observation_text}")

        else:
            observation_text = (
                f"[Tool {action_name!r} requires --tool_server_url. "
                "Only ZoomInSubfigure is available offline.]"
            )
            print(f"  [Skipped] {action_name} — no tool server URL")

        # Update image state
        if edited_b64:
            round_key = f"img_round_{turn}"
            image_states[round_key] = edited_b64
            image_states["img_last"] = edited_b64
            last_edited_b64 = edited_b64

        # Append assistant response and tool observation
        messages.append({"role": "assistant", "content": response})
        obs_content: list = [{"type": "text", "text": f"Observation: {observation_text}"}]
        if edited_b64:
            round_id = f"img_round_{turn}"
            obs_content[0]["text"] += f"\n[Output Image ID: {round_id}]"
            obs_content.append(_b64_image_url(edited_b64, mime))
        messages.append({"role": "user", "content": obs_content})

    return response


# ---------------------------------------------------------------------------
# Eval mode
# ---------------------------------------------------------------------------

def run_eval(args) -> None:
    with open(args.data_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        samples = samples[:args.limit]

    logger.info(f"Evaluating {len(samples)} samples")
    results = []
    correct = 0

    for i, sample in enumerate(samples):
        question = sample.get("question", "")
        label = sample.get("answer", sample.get("label", ""))
        img_rel = sample.get("image_path", sample.get("image", ""))
        img_path = os.path.join(args.image_dir, img_rel) if img_rel and args.image_dir else None

        response = run_single(
            args.model, question, img_path,
            tool_server_url=args.tool_server_url if args.tool_server_url else None,
        )

        is_correct = str(label).lower().strip() in response.lower()
        correct += int(is_correct)
        results.append({"id": i, "question": question, "label": label,
                         "response": response, "correct": is_correct})

        if (i + 1) % 50 == 0:
            logger.info(f"[{i+1}/{len(samples)}] acc={correct/(i+1):.4f}")

    acc = correct / len(samples) if samples else 0.0
    logger.info(f"Final accuracy: {acc:.4f} ({correct}/{len(samples)})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"accuracy": acc, "n_correct": correct,
                       "n_total": len(samples), "results": results}, f, indent=2)
        logger.info(f"Results saved to {args.output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meissa Framework II: Interleaved Thinking with Images"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MEISSA_MODEL", "CYX1998/Meissa-4B"),
        help="Model name/path served via OPENAI_BASE_URL",
    )
    parser.add_argument(
        "--tool_server_url",
        default=os.environ.get("TOOL_SERVER_URL", ""),
        help="URL of the visual tool server (e.g. http://localhost:8080). "
             "If omitted, ZoomInSubfigure runs offline via OpenThinkIMG.",
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run SLAKE demo question with ZoomInSubfigure (offline or via server)")
    parser.add_argument("--data_path", default=None, help="Path to eval JSONL file")
    parser.add_argument("--image_dir", default=None, help="Root directory of images")
    parser.add_argument("--output", default=None, help="Path to save JSON results")
    parser.add_argument("--limit", type=int, default=None, help="Max samples (for debug)")
    args = parser.parse_args()

    if args.demo:
        print("=" * 60)
        print("Framework II: Interleaved Thinking with Images (Demo)")
        print(f"Model: {args.model}")
        if args.tool_server_url:
            print(f"Tool server: {args.tool_server_url}")
        else:
            print("Tool mode: offline ZoomInSubfigure (OpenThinkIMG zoom_in_worker)")
        print("=" * 60)
        print(f"\nQuestion: {_DEMO_QUESTION}")
        print(f"Image: {_DEMO_IMAGE}")

        # Resolve demo image path
        demo_image = _DEMO_IMAGE
        if not os.path.exists(demo_image):
            logger.warning("Demo image not found: %s — running without image", demo_image)
            demo_image = None

        print()
        answer = run_single(
            args.model,
            _DEMO_QUESTION,
            image_path=demo_image,
            tool_server_url=args.tool_server_url or None,
            max_turns=8,
        )
        print(f"\nAnswer: {answer}")
        print(f"Expected: liver")
        return

    if not args.data_path:
        parser.error("--data_path is required for eval mode (or use --demo)")

    run_eval(args)


if __name__ == "__main__":
    main()
