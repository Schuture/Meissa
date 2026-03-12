"""
generate_sft_data.py — Main script for generating agentic SFT training data
from MedAgentSim medical datasets.

Supports:
- Multi-turn agentic simulation (doctor requests exams/tests → gets results → diagnoses)
- vLLM backend (Qwen3-VL) and Gemini backend
- SLURM array job chunking (--total_chunks / --chunk_idx)
- Filtering wrong samples from previous runs (--filter_wrong_from)
- Checkpoint/resume per scenario
"""

import os
import sys
import json
import re
import random
import argparse
import logging
import time

from tqdm import tqdm

# Add project root to path so we can import medsim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medsim.core.scenario import (
    ScenarioLoaderMedQA, ScenarioLoaderMedQAExtended,
    ScenarioLoaderMIMICIV, ScenarioLoaderNEJM, ScenarioLoaderNEJMExtended,
    ScenarioLoaderMedQATrain, ScenarioLoaderMIMICCXR,
)
from sft_utils import (
    configure_gemini, query_gemini_with_retry, query_vllm,
    build_system_prompt, build_patient_presentation, extract_scenario_data,
    parse_tool_call, lookup_exam, lookup_test,
    evaluate_correctness,
    conversations_to_openai_messages, conversations_to_gemini_messages,
    filter_wrong_scenario_ids, format_sft_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# Dataset Loaders
# ============================================================

DATASET_LOADERS = {
    "MedQA": ScenarioLoaderMedQA,
    "MedQA_Ext": ScenarioLoaderMedQAExtended,
    "MIMICIV": ScenarioLoaderMIMICIV,
    "NEJM": ScenarioLoaderNEJM,
    "NEJM_Ext": ScenarioLoaderNEJMExtended,
    "MedQA_Train": ScenarioLoaderMedQATrain,
    "MIMIC_CXR": ScenarioLoaderMIMICCXR,
}

DATASET_TYPES = {
    "MedQA": "osce",
    "MedQA_Ext": "osce",
    "MIMICIV": "osce",
    "NEJM": "nejm",
    "NEJM_Ext": "nejm",
    "MedQA_Train": "osce",
    "MIMIC_CXR": "osce_multimodal",
}


def get_dataset_type(dataset_name):
    return DATASET_TYPES.get(dataset_name, "osce")


# ============================================================
# Multi-turn Simulation
# ============================================================

def _clean_model_response(raw_response):
    """Clean common artifacts from model responses."""
    # Strip format instruction leakage (Qwen3 Instruct often appends this)
    raw_response = re.sub(r'\s*\[END OF FORMAT INSTRUCTIONS\]\s*$', '', raw_response)

    # Handle Thinking model output: wrap reasoning in <think> tags.
    # The Thinking model (without --enable-thinking) outputs reasoning without
    # the opening <think> tag but includes </think> before the JSON action.
    if '</think>' in raw_response and '<think>' not in raw_response:
        think_end_pos = raw_response.find('</think>')
        reasoning = raw_response[:think_end_pos].strip()
        action = raw_response[think_end_pos + len('</think>'):].strip()
        if reasoning and action:
            raw_response = f"<think>\n{reasoning}\n</think>\n{action}"

    return raw_response.strip()


def _handle_test_result(test_data, test_name, dataset_type, images_list):
    """Handle a RequestTest tool call, returning observation text.
    For osce_multimodal, if the test is Chest_XRay, include <image> tag."""

    if dataset_type == "osce_multimodal" and test_name and \
       any(k in test_name.lower() for k in ["chest_xray", "chest xray", "chest x-ray", "cxr", "chest_x_ray"]):
        # Direct dict access — do NOT use lookup_in_dict which converts dicts to text
        cxr_data = None
        for key in test_data:
            if any(k in key.lower() for k in ["chest_xray", "chest xray", "chest x-ray", "cxr"]):
                cxr_data = test_data[key]
                break

        if isinstance(cxr_data, dict) and "image_path" in cxr_data:
            image_path = cxr_data["image_path"]
            findings = cxr_data.get("findings", "No findings available.")
            if os.path.exists(image_path):
                images_list.append(image_path)
                return f"<image>\nChest X-Ray Report:\n{findings}"
            else:
                return f"Chest X-Ray Report:\n{findings}\n(Image file not found: {image_path})"
        elif isinstance(cxr_data, str):
            return f"Chest X-Ray:\n{cxr_data}"

    # Default: regular test lookup
    from sft_utils import lookup_test
    return lookup_test(test_data, test_name)


def run_simulation_vllm(scenario, scenario_id, dataset_name, args):
    """Run multi-turn agentic simulation using vLLM backend."""
    dataset_type = get_dataset_type(dataset_name)
    available_exams, available_tests, exam_data, test_data, ground_truth, image_url = \
        extract_scenario_data(scenario, dataset_type)

    system_prompt = build_system_prompt(available_exams, available_tests, dataset_type)
    patient_text = build_patient_presentation(scenario, dataset_type)
    conversations = [{"from": "human", "value": patient_text}]
    images_list = []  # Track CXR images used in this scenario

    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")
    diagnosis = ""
    _last_test_name = None  # Track repeated test requests
    _repeat_count = 0

    for turn in range(args.max_turns):
        # Convert to OpenAI messages and query
        messages = conversations_to_openai_messages(system_prompt, conversations)
        try:
            raw_response = query_vllm(
                base_url, args.model, messages,
                max_tokens=args.max_tokens, temperature=0.6
            )
        except Exception as e:
            logger.error(f"vLLM query failed at turn {turn} for scenario {scenario_id}: {e}")
            raw_response = '<think>\nFailed to generate response.\n</think>\n\n{"name": "Terminate", "arguments": {"diagnosis": "Unknown"}}'

        # Clean and append model response
        raw_response = _clean_model_response(raw_response)
        conversations.append({"from": "function_call", "value": raw_response})

        # Parse tool call
        tool_name, tool_args, success = parse_tool_call(raw_response)

        if not success:
            # Model didn't produce valid tool call — force terminate
            logger.warning(f"Scenario {scenario_id} turn {turn}: failed to parse tool call, forcing terminate")
            conversations.append({"from": "observation", "value": "Invalid tool call format. Please use the correct JSON format."})
            continue

        if tool_name == "Terminate":
            diagnosis = tool_args.get("diagnosis", "Unknown")
            conversations.append({"from": "observation", "value": "Diagnosis recorded."})
            conversations.append({
                "from": "gpt",
                "value": f"<think>\nI will provide my final diagnosis.\n</think>\n[FINAL] {diagnosis}"
            })
            break

        elif tool_name == "RequestPhysicalExam":
            exam_name = tool_args.get("exam", "")
            result = lookup_exam(exam_data, exam_name)
            conversations.append({"from": "observation", "value": result})

        elif tool_name == "RequestTest":
            test_name = tool_args.get("test", "")
            # Detect repeated same-test requests
            if turn > 0 and _last_test_name == test_name:
                _repeat_count += 1
                if _repeat_count >= 2:
                    conversations.append({"from": "observation",
                        "value": f"You already requested {test_name} and received the results. "
                                 "Please use a different action or provide your diagnosis."})
                    continue
            else:
                _repeat_count = 0
            _last_test_name = test_name
            result = _handle_test_result(test_data, test_name, dataset_type, images_list)
            conversations.append({"from": "observation", "value": result})

        elif tool_name == "RequestExamResults":
            # NEJM: return all exam results as one block
            if isinstance(exam_data, str):
                conversations.append({"from": "observation", "value": exam_data})
            else:
                from sft_utils import format_dict_as_text
                conversations.append({"from": "observation", "value": format_dict_as_text(exam_data)})

        else:
            conversations.append({"from": "observation", "value": f"Unknown action: {tool_name}"})

    else:
        # Max turns reached without Terminate
        if not diagnosis:
            diagnosis = "Unknown (max turns reached)"
            conversations.append({
                "from": "gpt",
                "value": f"<think>\nMax turns reached. Providing best guess.\n</think>\n[FINAL] {diagnosis}"
            })

    # Evaluate correctness
    genai = None
    eval_model = None
    if args.eval_model:
        try:
            genai = configure_gemini(args.gemini_api_key)
            eval_model = args.eval_model
        except Exception:
            pass

    is_correct, method = evaluate_correctness(diagnosis, ground_truth, genai, eval_model)
    logger.info(f"Scenario {scenario_id}: diagnosis='{diagnosis}' | gt='{ground_truth}' | correct={is_correct} ({method})")

    result = format_sft_sample(
        scenario_id, dataset_name, ground_truth, diagnosis, is_correct,
        conversations, system_prompt, source=args.model, eval_method=method
    )
    if images_list:
        result["images"] = images_list
    return result


def run_simulation_gemini(scenario, scenario_id, dataset_name, args, genai):
    """Run multi-turn agentic simulation using Gemini backend."""
    dataset_type = get_dataset_type(dataset_name)
    available_exams, available_tests, exam_data, test_data, ground_truth, image_url = \
        extract_scenario_data(scenario, dataset_type)

    system_prompt = build_system_prompt(available_exams, available_tests, dataset_type)
    patient_text = build_patient_presentation(scenario, dataset_type)
    conversations = [{"from": "human", "value": patient_text}]
    images_list = []  # Track CXR images used in this scenario

    diagnosis = ""
    _last_test_name = None  # Track repeated test requests
    _repeat_count = 0

    for turn in range(args.max_turns):
        # Convert to Gemini messages and query
        gemini_messages = conversations_to_gemini_messages(conversations)
        try:
            raw_response = query_gemini_with_retry(
                genai, args.gemini_model, system_prompt, gemini_messages,
                max_tokens=args.max_tokens, temperature=0.6, max_retries=8,
                fatal_on_429=True
            )
        except Exception as e:
            logger.error(f"Gemini query failed at turn {turn} for scenario {scenario_id}: {e}")
            raw_response = '<think>\nFailed to generate response.\n</think>\n\n{"name": "Terminate", "arguments": {"diagnosis": "Unknown"}}'

        # Clean and append model response
        raw_response = _clean_model_response(raw_response)
        conversations.append({"from": "function_call", "value": raw_response})

        tool_name, tool_args, success = parse_tool_call(raw_response)

        if not success:
            logger.warning(f"Scenario {scenario_id} turn {turn}: failed to parse tool call, forcing terminate")
            conversations.append({"from": "observation", "value": "Invalid tool call format. Please use the correct JSON format."})
            continue

        if tool_name == "Terminate":
            diagnosis = tool_args.get("diagnosis", "Unknown")
            conversations.append({"from": "observation", "value": "Diagnosis recorded."})
            conversations.append({
                "from": "gpt",
                "value": f"<think>\nI will provide my final diagnosis.\n</think>\n[FINAL] {diagnosis}"
            })
            break

        elif tool_name == "RequestPhysicalExam":
            exam_name = tool_args.get("exam", "")
            result = lookup_exam(exam_data, exam_name)
            conversations.append({"from": "observation", "value": result})

        elif tool_name == "RequestTest":
            test_name = tool_args.get("test", "")
            # Detect repeated same-test requests
            if turn > 0 and _last_test_name == test_name:
                _repeat_count += 1
                if _repeat_count >= 2:
                    conversations.append({"from": "observation",
                        "value": f"You already requested {test_name} and received the results. "
                                 "Please use a different action or provide your diagnosis."})
                    continue
            else:
                _repeat_count = 0
            _last_test_name = test_name
            result = _handle_test_result(test_data, test_name, dataset_type, images_list)
            conversations.append({"from": "observation", "value": result})

        elif tool_name == "RequestExamResults":
            if isinstance(exam_data, str):
                conversations.append({"from": "observation", "value": exam_data})
            else:
                from sft_utils import format_dict_as_text
                conversations.append({"from": "observation", "value": format_dict_as_text(exam_data)})

        else:
            conversations.append({"from": "observation", "value": f"Unknown action: {tool_name}"})

        # Rate limiting for Gemini
        time.sleep(0.5)

    else:
        if not diagnosis:
            diagnosis = "Unknown (max turns reached)"
            conversations.append({
                "from": "gpt",
                "value": f"<think>\nMax turns reached. Providing best guess.\n</think>\n[FINAL] {diagnosis}"
            })

    # Evaluate correctness
    is_correct, method = evaluate_correctness(diagnosis, ground_truth, genai, args.eval_model or args.gemini_model)
    logger.info(f"Scenario {scenario_id}: diagnosis='{diagnosis}' | gt='{ground_truth}' | correct={is_correct} ({method})")

    result = format_sft_sample(
        scenario_id, dataset_name, ground_truth, diagnosis, is_correct,
        conversations, system_prompt, source=args.gemini_model, eval_method=method
    )
    if images_list:
        result["images"] = images_list
    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate agentic SFT data from MedAgentSim datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_LOADERS.keys()),
                        help="Dataset to use")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (for vLLM: served model name; for gemini: gemini model name)")
    parser.add_argument("--backend", type=str, required=True, choices=["vllm", "gemini"],
                        help="Inference backend")
    parser.add_argument("--gemini_model", type=str, default="gemini-3-flash-preview",
                        help="Gemini model name (when backend=gemini)")
    parser.add_argument("--gemini_api_key", type=str, default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env)")
    parser.add_argument("--eval_model", type=str, default=None,
                        help="Model for LLM-based correctness evaluation")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Max generation tokens per turn")
    parser.add_argument("--max_turns", type=int, default=8,
                        help="Max turns per scenario")
    parser.add_argument("--num_samples", type=int, default=999999,
                        help="Max samples per chunk")
    parser.add_argument("--total_chunks", type=int, default=1,
                        help="Total number of chunks")
    parser.add_argument("--chunk_idx", type=int, default=0,
                        help="Current chunk index (0-based)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for chunk results")
    parser.add_argument("--filter_wrong_from", type=str, default=None,
                        help="Dir with previous results; only re-run wrong samples")

    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    loader = DATASET_LOADERS[args.dataset]()
    logger.info(f"Loaded {loader.num_scenarios} scenarios")

    # Build scenario ID list
    all_scenario_ids = list(range(loader.num_scenarios))

    # Shuffle with fixed seed (consistent across all chunks)
    random.seed(1)
    random.shuffle(all_scenario_ids)
    logger.info(f"Shuffled {len(all_scenario_ids)} scenarios with seed=1")

    # Filter wrong samples if requested (Step 2)
    if args.filter_wrong_from:
        wrong_ids = filter_wrong_scenario_ids(args.filter_wrong_from)
        all_scenario_ids = [sid for sid in all_scenario_ids if sid in wrong_ids]
        logger.info(f"Filtered to {len(all_scenario_ids)} wrong samples for re-inference")

    # Chunk splitting
    total_len = len(all_scenario_ids)
    if args.total_chunks > 1:
        chunk_size = total_len // args.total_chunks
        start = args.chunk_idx * chunk_size
        end = total_len if args.chunk_idx == args.total_chunks - 1 else start + chunk_size
        chunk_scenarios = all_scenario_ids[start:end]
        logger.info(f"Running chunk {args.chunk_idx + 1}/{args.total_chunks}")
        logger.info(f"Data slice: {start} to {end} (Total: {len(chunk_scenarios)})")
    else:
        chunk_scenarios = all_scenario_ids
        logger.info(f"Running full dataset ({len(chunk_scenarios)} samples)")

    # Apply num_samples limit
    if args.num_samples < len(chunk_scenarios):
        logger.info(f"Limiting to first {args.num_samples} samples per chunk")
        chunk_scenarios = chunk_scenarios[:args.num_samples]

    # Output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"chunk_{args.chunk_idx}.jsonl")

    # Resume: load already processed scenario_ids
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        processed_ids.add(record["scenario_id"])
                    except json.JSONDecodeError:
                        pass
        logger.info(f"Resuming: {len(processed_ids)} scenarios already processed")

    # Configure Gemini if needed
    genai = None
    if args.backend == "gemini":
        genai = configure_gemini(args.gemini_api_key)

    # Main loop
    logger.info(f"Starting inference ({args.backend}) on {len(chunk_scenarios)} scenarios...")

    for i, scenario_id in enumerate(tqdm(chunk_scenarios, desc=f"Chunk {args.chunk_idx}")):
        if scenario_id in processed_ids:
            continue

        scenario = loader.get_scenario(id=scenario_id)

        try:
            if args.backend == "vllm":
                result = run_simulation_vllm(scenario, scenario_id, args.dataset, args)
            else:
                result = run_simulation_gemini(scenario, scenario_id, args.dataset, args, genai)
        except Exception as e:
            logger.error(f"Scenario {scenario_id} failed: {e}")
            result = format_sft_sample(
                scenario_id, args.dataset,
                scenario.diagnosis_information(), "Error", False,
                [], "", source=args.model
            )

        # Append to output file immediately (checkpoint)
        with open(output_file, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    logger.info(f"Chunk {args.chunk_idx} finished. Output: {output_file}")


if __name__ == "__main__":
    main()
