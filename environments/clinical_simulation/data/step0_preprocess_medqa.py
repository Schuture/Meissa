"""
preprocess_medqa.py — Convert raw MedQA MCQ questions into structured OSCE format
using Gemini API.

This follows the same approach as MedAgentSim's original conversion using GPT-4o:
split each MCQ into patient_info (what patient reports) + physical_exams + test_results
+ correct_diagnosis.

Usage:
    python preprocess_medqa.py \
        --input_file /path/to/MedQA/data_clean/questions/US/train.jsonl \
        --output_file datasets/medqa_train_osce.jsonl \
        --total_chunks 8 --chunk_idx 0 \
        --num_samples 999999
"""

import os
import sys
import json
import re
import argparse
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_utils import configure_gemini, query_gemini_with_retry

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OSCE_CONVERSION_PROMPT = """You are a medical education expert. Convert this USMLE-style multiple-choice question into a structured OSCE (Objective Structured Clinical Examination) case.

QUESTION:
{question}

CORRECT ANSWER: {answer}

INSTRUCTIONS:
1. Extract the patient's presenting information (what they would TELL the doctor): demographics, history, symptoms, past medical history, social history, review of systems.
2. Extract physical examination findings (what the doctor would FIND on exam): vital signs, specific exam findings.
3. Extract test/lab results (what diagnostic tests would SHOW): blood tests, imaging, special tests.
4. The correct diagnosis should be a concise disease/condition name.
5. CRITICAL: Do NOT put diagnostic test results or physical exam findings in the patient presentation. The doctor must REQUEST these.
6. If the question doesn't mention certain fields, use reasonable defaults or leave empty.
7. Output ONLY valid JSON, no markdown code blocks, no explanation.

OUTPUT FORMAT (strict JSON):
{{
  "OSCE_Examination": {{
    "Objective_for_Doctor": "Diagnose the patient's condition based on clinical evaluation",
    "Patient_Actor": {{
      "Demographics": "<age>-year-old <gender>",
      "History": "<brief history of present illness>",
      "Symptoms": {{
        "Primary_Symptom": "<main complaint>",
        "Secondary_Symptoms": ["<symptom1>", "<symptom2>"]
      }},
      "Past_Medical_History": "<relevant PMH>",
      "Social_History": "<relevant social history>",
      "Review_of_Systems": "<relevant ROS findings>"
    }},
    "Physical_Examination_Findings": {{
      "Vital_Signs": {{<key>: <value>}},
      "<Exam_Category>": {{<finding>: <value>}}
    }},
    "Test_Results": {{
      "<Test_Category>": {{<test>: <result>}}
    }},
    "Correct_Diagnosis": "<diagnosis>"
  }}
}}"""


def extract_json_from_response(text):
    """Extract JSON object from Gemini response, handling markdown blocks."""
    # Try to find JSON in markdown code block
    md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    # Find the outermost { ... }
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    continue
    return None


def validate_osce_structure(data):
    """Validate that the converted OSCE data has required fields."""
    if not isinstance(data, dict):
        return False
    osce = data.get("OSCE_Examination", data)
    if "OSCE_Examination" not in data and "Patient_Actor" not in data:
        return False
    if "OSCE_Examination" in data:
        osce = data["OSCE_Examination"]
    required = ["Patient_Actor", "Physical_Examination_Findings", "Test_Results", "Correct_Diagnosis"]
    return all(k in osce for k in required)


def convert_single_mcq(genai, gemini_model, question_data):
    """Convert a single MedQA MCQ to OSCE format using Gemini."""
    prompt = OSCE_CONVERSION_PROMPT.format(
        question=question_data["question"],
        answer=question_data["answer"],
    )

    messages = [{"role": "user", "parts": [prompt]}]
    response = query_gemini_with_retry(
        genai, gemini_model,
        "You are a medical education expert specializing in OSCE case design. Output ONLY valid JSON.",
        messages,
        max_tokens=4096,
        temperature=0.3,
        max_retries=8,
    )

    parsed = extract_json_from_response(response)
    if parsed is None:
        raise ValueError(f"Failed to parse JSON from response: {response[:200]}...")

    if not validate_osce_structure(parsed):
        raise ValueError(f"Invalid OSCE structure: missing required fields")

    # Ensure top-level wrapper
    if "OSCE_Examination" not in parsed:
        parsed = {"OSCE_Examination": parsed}

    # Ensure Objective_for_Doctor exists
    if "Objective_for_Doctor" not in parsed["OSCE_Examination"]:
        parsed["OSCE_Examination"]["Objective_for_Doctor"] = "Diagnose the patient's condition based on clinical evaluation"

    return parsed


def load_processed_indices(output_file):
    """Load already-processed line indices from output file for resume support."""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        idx = record.get("_source_idx")
                        if idx is not None:
                            processed.add(idx)
                    except json.JSONDecodeError:
                        pass
    return processed


def main():
    parser = argparse.ArgumentParser(description="Convert MedQA MCQ to OSCE format using Gemini")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to raw MedQA train.jsonl")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output path for OSCE JSONL")
    parser.add_argument("--gemini_model", type=str, default="gemini-3-flash-preview",
                        help="Gemini model to use for conversion")
    parser.add_argument("--total_chunks", type=int, default=1,
                        help="Total chunks for parallel processing")
    parser.add_argument("--chunk_idx", type=int, default=0,
                        help="Current chunk index (0-based)")
    parser.add_argument("--num_samples", type=int, default=999999,
                        help="Max samples to process in this run")
    parser.add_argument("--rate_limit_delay", type=float, default=0.5,
                        help="Delay between Gemini API calls (seconds)")
    args = parser.parse_args()

    # Load input data
    logger.info(f"Loading input: {args.input_file}")
    with open(args.input_file, 'r') as f:
        all_questions = [json.loads(line) for line in f if line.strip()]
    total = len(all_questions)
    logger.info(f"Loaded {total} questions")

    # Chunk splitting
    if args.total_chunks > 1:
        chunk_size = total // args.total_chunks
        start = args.chunk_idx * chunk_size
        end = total if args.chunk_idx == args.total_chunks - 1 else start + chunk_size
        indices = list(range(start, end))
        logger.info(f"Chunk {args.chunk_idx}/{args.total_chunks}: indices {start}-{end} ({len(indices)} questions)")
    else:
        indices = list(range(total))

    # Apply num_samples limit
    if args.num_samples < len(indices):
        indices = indices[:args.num_samples]
        logger.info(f"Limited to {args.num_samples} samples")

    # Determine output file (per-chunk)
    if args.total_chunks > 1:
        base, ext = os.path.splitext(args.output_file)
        output_file = f"{base}_chunk_{args.chunk_idx}{ext}"
    else:
        output_file = args.output_file

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Resume: load already processed
    processed_indices = load_processed_indices(output_file)
    if processed_indices:
        logger.info(f"Resuming: {len(processed_indices)} already processed, skipping")

    # Configure Gemini
    genai = configure_gemini()
    logger.info(f"Gemini configured, model: {args.gemini_model}")

    # Process
    success_count = 0
    fail_count = 0

    for i, idx in enumerate(indices):
        if idx in processed_indices:
            continue

        q = all_questions[idx]
        try:
            osce_data = convert_single_mcq(genai, args.gemini_model, q)

            # Add source metadata for tracing
            osce_data["_source_idx"] = idx
            osce_data["_source_answer_idx"] = q.get("answer_idx", "")
            osce_data["_source_meta_info"] = q.get("meta_info", "")

            with open(output_file, 'a') as f:
                f.write(json.dumps(osce_data, ensure_ascii=False) + '\n')

            success_count += 1
            if success_count % 50 == 0:
                logger.info(f"Progress: {success_count} converted, {fail_count} failed "
                           f"({i + 1}/{len(indices)})")

        except Exception as e:
            fail_count += 1
            logger.warning(f"Index {idx} failed: {e}")
            # Write failure record for tracking
            fail_record = {
                "_source_idx": idx,
                "_conversion_error": str(e),
                "OSCE_Examination": None,
            }
            with open(output_file, 'a') as f:
                f.write(json.dumps(fail_record, ensure_ascii=False) + '\n')

        # Rate limiting
        if args.rate_limit_delay > 0:
            time.sleep(args.rate_limit_delay)

    logger.info(f"Done. Success: {success_count}, Failed: {fail_count}")
    logger.info(f"Output: {output_file}")


if __name__ == "__main__":
    main()
