"""
medsim/main.py — Framework IV: Multi-turn Clinical Simulation entry point.

Usage (demo — no data required):
    python medsim/main.py --model Qwen/Qwen3-VL-4B-Instruct --mode demo

Usage (eval — requires dataset):
    python medsim/main.py \\
        --model Qwen/Qwen3-VL-4B-Instruct \\
        --mode eval \\
        --dataset MedQA \\
        --num_scenarios 50 \\
        --total_inferences 20

Environment variables:
    MEISSA_MODEL       Model path or HuggingFace ID (overrides --model)
    OPENAI_BASE_URL    Base URL of a running vLLM server (e.g. http://localhost:8001/v1)
    VLLM_PORT          vLLM port when OPENAI_BASE_URL is not set (default: 8001)
    DATA_ROOT          Root directory of datasets (needed for --mode eval)
"""

import argparse
import json
import os
import sys
import time
import logging

# Ensure the local medsim package (clinical_simulation/) takes precedence over
# any installed MedAgentSim package that may be on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo scenario — MIMIC-IV scenario_265 (Psoriatic Arthritis)
# Model correctly diagnosed in evaluation with 6 tool calls and systematic
# differential diagnosis (CBC -> ANA -> X-ray -> RF -> skin exam -> diagnosis).
# ---------------------------------------------------------------------------
_DEMO_SCENARIO = {
    "OSCE_Examination": {
        "Objective_for_Doctor": "Diagnose the patient based on clinical evaluation.",
        "Patient_Actor": {
            "Demographics": "45-year-old male",
            "History": (
                "The patient reports developing a widespread skin rash over the past two weeks. "
                "The rash is non-itchy, raised, and reddish. He also complains of joint pain "
                "primarily in his knees and wrists which started a few days before the rash "
                "appeared. He denies any recent travel or contact with new medications or substances."
            ),
            "Symptoms": {
                "Primary_Symptom": "Non-itchy, widespread skin rash",
                "Secondary_Symptoms": [
                    "Joint pain in knees and wrists",
                    "No fever",
                    "No recent travel",
                    "No new medications",
                ],
            },
            "Past_Medical_History": (
                "History of psoriasis diagnosed 5 years ago, treated with topical steroids. "
                "No other significant illnesses."
            ),
            "Social_History": "Non-smoker, does not consume alcohol. Works as an accountant.",
            "Review_of_Systems": "Denies fever, chills, gastrointestinal symptoms, or respiratory symptoms.",
        },
        "Physical_Examination_Findings": {
            "Integumentary_Examination": (
                "Skin: Presence of widespread erythematous plaques with scales, most prominent "
                "on the elbows, knees, and scalp. Nails: Pitting of the nails observed. "
                "Joint Examination: Mild swelling and tenderness in the knees and wrists "
                "without notable redness or warmth."
            ),
        },
        "Test_Results": {
            "Complete_Blood_Count": "WBC: 10,000 /uL; Hemoglobin: 14.0 g/dL; Platelets: 240,000 /uL",
            "Antinuclear_Antibody_Test": "Negative",
            "X-ray_Joints": "Findings: No evidence of joint space narrowing or erosions.",
            "Rheumatoid_Factor": "Negative",
        },
        "Correct_Diagnosis": "Psoriatic Arthritis",
    }
}


def _query_vllm_sft(base_url: str, model_name: str, messages: list,
                    max_tokens: int = 1024, temperature: float = 0.6) -> str:
    """Call vLLM and return the full response text.

    With --tool-call-parser hermes + --enable-auto-tool-choice, vLLM may extract
    <tool_call> blocks from content into tool_calls. We reconstruct them.
    """
    import re as _re
    import requests as _requests
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = _requests.post(url, json=payload, timeout=120,
                          headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    content = msg.get("content") or ""
    # Reconstruct tool calls that vLLM moved to tool_calls field
    for tc in (msg.get("tool_calls") or []):
        func = tc.get("function", {})
        name = func.get("name", "")
        try:
            args = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args = {}
        content += f'\n{{"name": "{name}", "arguments": {json.dumps(args)}}}'
    return content.strip()


def _parse_sft_tool_call(raw_response: str):
    """Parse {"name": ..., "arguments": ...} from model response.

    Returns (tool_name, arguments_dict, success).
    """
    import re as _re
    # Look for {"name": ..., "arguments": {...}} pattern
    match = _re.search(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*\})',
        raw_response, _re.DOTALL
    )
    if match:
        try:
            name = match.group(1)
            args = json.loads(match.group(2))
            return name, args, True
        except (json.JSONDecodeError, TypeError):
            pass
    return None, {}, False


def run_demo(model_name: str, total_inferences: int = 12) -> None:
    """Run the SFT evaluation loop — single-agent tool-calling format.

    The model receives the patient presentation and calls:
      RequestPhysicalExam → get exam findings
      RequestTest         → get test results
      Terminate           → provide final diagnosis

    This mirrors the evaluate_sft.py loop used for evaluation, which is the
    actual format the Meissa SFT model was trained on.
    """
    osce = _DEMO_SCENARIO["OSCE_Examination"]
    patient = osce["Patient_Actor"]
    ground_truth = osce["Correct_Diagnosis"]

    # Extract available exams and tests
    exam_data = osce.get("Physical_Examination_Findings", {})
    test_data = osce.get("Test_Results", {})
    available_exams = list(exam_data.keys())
    available_tests = list(test_data.keys())

    # Build system prompt (matches sft_utils.build_system_prompt format)
    exams_str = ", ".join(available_exams) if available_exams else "None"
    tests_str = ", ".join(available_tests) if available_tests else "None"
    system_prompt = f"""[BEGIN OF GOAL]
You are an expert medical diagnostician evaluating a patient. You will be given the patient's presenting complaint and history. Use the available tools to request physical examinations and medical tests to gather evidence, then provide your final diagnosis.

You must reason step-by-step using <think>...</think> tags before each tool call.
[END OF GOAL]

[BEGIN OF ACTIONS]
Name: RequestPhysicalExam
Description: Request a specific physical examination finding for the patient.
Arguments: {{'exam': 'Name of the physical examination (choose from Available Physical Examinations)'}}
Examples:
{{"name": "RequestPhysicalExam", "arguments": {{"exam": "Respiratory"}}}}

Name: RequestTest
Description: Request a medical test or laboratory result for the patient.
Arguments: {{'test': 'Name of the medical test (choose from Available Medical Tests)'}}
Examples:
{{"name": "RequestTest", "arguments": {{"test": "Chest_X-Ray"}}}}

Name: Terminate
Description: Provide your final diagnosis and conclude the evaluation.
Arguments: {{'diagnosis': 'Your final diagnosis — the disease or condition name only.'}}
Examples:
{{"name": "Terminate", "arguments": {{"diagnosis": "Myasthenia gravis"}}}}
[END OF ACTIONS]

[BEGIN OF TASK INSTRUCTIONS]
1. Only select actions from ACTIONS.
2. Call at most one action at a time.
3. After receiving each observation, reason about what you've learned.
4. Request only the most relevant exams/tests.
5. Always finish by calling Terminate with your final diagnosis.
6. Output pure JSON for the tool call — no markdown wrappers.
[END OF TASK INSTRUCTIONS]

[BEGIN OF AVAILABLE DATA]
Physical Examinations: {exams_str}
Medical Tests: {tests_str}
[END OF AVAILABLE DATA]

[BEGIN OF FORMAT INSTRUCTIONS]
Your output must follow this format:
<think>
your clinical reasoning here
</think>

{{"name": "action_name", "arguments": {{"arg": "value"}}}}
[END OF FORMAT INSTRUCTIONS]"""

    # Build patient presentation
    hist = patient.get("History", "")
    symptoms = patient.get("Symptoms", {})
    pmh = patient.get("Past_Medical_History", "")
    social = patient.get("Social_History", "")
    ros = patient.get("Review_of_Systems", "")

    sym_parts = []
    if isinstance(symptoms, dict):
        if "Primary_Symptom" in symptoms:
            sym_parts.append(symptoms["Primary_Symptom"])
        for s in symptoms.get("Secondary_Symptoms", []):
            sym_parts.append(s)
    elif isinstance(symptoms, list):
        sym_parts = symptoms
    sym_str = "; ".join(sym_parts) if sym_parts else ""

    patient_text = f"Patient: {patient.get('Demographics', '')}\n"
    if hist:
        patient_text += f"History: {hist}\n"
    if sym_str:
        patient_text += f"Symptoms: {sym_str}\n"
    if pmh:
        patient_text += f"Past Medical History: {pmh}\n"
    if social:
        patient_text += f"Social History: {social}\n"
    if ros:
        patient_text += f"Review of Systems: {ros}\n"

    base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8877/v1")
    conversations = [{"role": "user", "content": patient_text.strip()}]

    print("=" * 60)
    print("Framework IV: Multi-turn Clinical Simulation (Demo)")
    print(f"Model: {model_name}")
    print("Format: Single-agent SFT tool-calling (evaluate_sft.py format)")
    print("=" * 60)
    print(f"\nPatient: {patient.get('Demographics', '')}")
    print(f"Chief Complaint: {symptoms.get('Primary_Symptom', '') if isinstance(symptoms, dict) else ''}\n")

    diagnosis = ""
    for turn in range(total_inferences):
        messages = [{"role": "system", "content": system_prompt}] + conversations
        try:
            raw = _query_vllm_sft(base_url, model_name, messages, temperature=0.3)
        except Exception as e:
            logger.error(f"vLLM query failed at turn {turn}: {e}")
            break

        # Strip <think>...</think> for display, keep for parsing
        import re as _re
        think_match = _re.search(r"<think>(.*?)</think>", raw, _re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""
        if thinking:
            print(f"[Turn {turn+1}] <think> {thinking[:200]}{'...' if len(thinking)>200 else ''} </think>")

        tool_name, tool_args, success = _parse_sft_tool_call(raw)

        if not success:
            logger.warning(f"Turn {turn}: failed to parse tool call from: {raw[:200]}")
            conversations.append({"role": "assistant", "content": raw})
            conversations.append({"role": "user", "content":
                'Invalid format. Please output: {"name": "tool_name", "arguments": {...}}'})
            continue

        conversations.append({"role": "assistant", "content": raw})
        print(f"[Turn {turn+1}] Tool: {tool_name}({tool_args})")

        if tool_name == "Terminate":
            diagnosis = tool_args.get("diagnosis", "Unknown")
            break
        elif tool_name == "RequestPhysicalExam":
            exam = tool_args.get("exam", "")
            # Fuzzy match against available exams
            result = None
            for k, v in exam_data.items():
                if exam.lower() in k.lower() or k.lower() in exam.lower():
                    result = f"{k}: {v}" if isinstance(v, str) else f"{k}: {json.dumps(v)}"
                    break
            observation = result or f"No findings available for '{exam}'."
        elif tool_name == "RequestTest":
            test = tool_args.get("test", "")
            result = None
            for k, v in test_data.items():
                if test.lower().replace(" ", "_").replace("-", "_") in k.lower().replace(" ", "_").replace("-", "_") or \
                   k.lower().replace(" ", "_").replace("-", "_") in test.lower().replace(" ", "_").replace("-", "_"):
                    result = f"{k}: {v}"
                    break
            observation = result or f"Test '{test}' not available."
        else:
            observation = f"Unknown tool '{tool_name}'."

        print(f"           Observation: {observation}")
        conversations.append({"role": "user", "content": observation})

    correct = bool(diagnosis.strip()) and (
        diagnosis.lower().strip() in ground_truth.lower() or
        ground_truth.lower() in diagnosis.lower()
    )
    print(f"\nDiagnosis:      {diagnosis}")
    print(f"Ground truth:   {ground_truth}")
    print(f"Result:         {'CORRECT ✓' if correct else 'INCORRECT ✗'}")


def run_eval(args) -> None:
    from medsim.core.agent import MeasurementAgent, PatientAgent, DoctorAgent
    from medsim.core.scenario import (
        ScenarioLoaderMedQA,
        ScenarioLoaderNEJM,
        ScenarioLoaderMIMICIV,
    )
    from medsim.query_model import BAgent
    from medsim.agents import compare_results

    model_name = args.model
    loaders = {
        "MedQA": ScenarioLoaderMedQA,
        "NEJM": ScenarioLoaderNEJM,
        "MIMICIV": ScenarioLoaderMIMICIV,
    }
    if args.dataset not in loaders:
        raise ValueError(f"Unknown dataset: {args.dataset}. Choose from {list(loaders)}")

    scenario_loader = loaders[args.dataset]()
    num_scenarios = args.num_scenarios or scenario_loader.num_scenarios

    total_correct, total_presents = 0, 0

    for _scenario_id in range(min(num_scenarios, scenario_loader.num_scenarios)):
        total_presents += 1
        scenario = scenario_loader.get_scenario(id=_scenario_id)

        mod_pipe = BAgent(model_name=model_name)
        meas_agent = MeasurementAgent(backend_str=model_name)
        patient_agent = PatientAgent(backend_str=model_name)
        doctor_agent = DoctorAgent(backend_str=model_name)

        patient_agent.update_scenario(scenario=scenario, bias_present="None")
        doctor_agent.update_scenario(
            scenario=scenario,
            bias_present="None",
            max_infs=args.total_inferences,
            img_request=False,
        )
        meas_agent.update_scenario(scenario=scenario)

        pi_dialogue = ""
        for _inf_id in range(args.total_inferences):
            if _inf_id == args.total_inferences - 1:
                pi_dialogue += "\nThis is the final question. Please provide a diagnosis.\n"

            doctor_dialogue = doctor_agent.inference_doctor(
                pi_dialogue, image_requested=False, scenario_id=_scenario_id
            )
            logger.info("Doctor: %s", doctor_dialogue)

            if "DIAGNOSIS READY" in doctor_dialogue:
                correct = compare_results(
                    doctor_dialogue, scenario.diagnosis_information(), mod_pipe
                )
                if correct:
                    total_correct += 1
                logger.info(
                    "Scenario %d: %s (%d/%d)",
                    _scenario_id,
                    "CORRECT" if correct else "INCORRECT",
                    total_correct,
                    total_presents,
                )
                break

            if "REQUEST TEST" in doctor_dialogue:
                pi_dialogue = meas_agent.inference_measurement(doctor_dialogue)
            else:
                pi_dialogue = patient_agent.inference_patient(doctor_dialogue)

            time.sleep(0.5)

    acc = total_correct / total_presents if total_presents else 0
    print(f"\nFinal accuracy: {total_correct}/{total_presents} = {acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meissa Framework IV: Multi-turn Clinical Simulation"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MEISSA_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
        help="Model name/path served via OPENAI_BASE_URL, or loaded locally as fallback",
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "eval"],
        default="demo",
        help="'demo' runs a single built-in scenario; 'eval' runs the full dataset",
    )
    parser.add_argument(
        "--dataset",
        default="MedQA",
        choices=["MedQA", "NEJM", "MIMICIV"],
        help="Dataset for eval mode (requires DATA_ROOT to be set)",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=None,
        help="Number of scenarios to run (eval mode only; default: all)",
    )
    parser.add_argument(
        "--total_inferences",
        type=int,
        default=20,
        help="Max turns per scenario",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(args.model, total_inferences=min(args.total_inferences, 8))
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
