#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import random
import uuid
import ast
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from medrax.utils import load_prompts_from_file 

# =============================================================================
#  Utils & Helpers
# =============================================================================

_IMAGE_PATH_TOKEN = "<image_path>"

def safe_json_dumps(x: Any) -> str:
    if x is None: return ""
    elif isinstance(x, str): return x
    else:
        try: return json.dumps(x, ensure_ascii=False)
        except: return str(x)

def extract_section_prompt(prompt_file: str, section_name: str) -> str:
    prompts = load_prompts_from_file(prompt_file)
    if section_name not in prompts:
        raise KeyError(f"Section [{section_name}] not found.")
    return prompts[section_name].strip()

def extract_final_answer_v2(sample: Dict[str, Any]) -> str:
    # 1. 优先尝试从 assistant_final 字段获取
    val = (sample.get("assistant_final") or "").strip()
    if val and val.lower() != "unknown":
        return val
    
    # 2. 尝试从 raw text 解析
    raw = (sample.get("assistant_raw") or sample.get("assistant_answer") or "").strip()
    m = re.search(r"\[FINAL\]\s*(.+?)\s*$", raw, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    
    return "" 

def generate_random_simulated_path(extension=".jpg") -> str:
    strategies = [
        lambda: f"{uuid.uuid4().hex[:8]}{extension}",
        lambda: f"/mnt/data/{uuid.uuid4().hex[:6]}/{uuid.uuid4().hex[:8]}{extension}",
        lambda: f"p{random.randint(10,19)}/p{random.randint(10000000,19999999)}/s{random.randint(50000000,59999999)}/{uuid.uuid4()}{extension}",
        lambda: f"uploads/user_{random.randint(1,999)}/chest_xray{extension}",
        lambda: f"data_v2/batch_{random.randint(0,100)}/img-{random.randint(0,1000)}{extension}"
    ]
    return random.choice(strategies)()

# --- Text Cleaning (Fixed Bug Here) ---

def clean_observation_content(content: str) -> str:
    """
    清洗 tool_result 的内容。
    输入通常是: "('REPORT TEXT...', {'model_thought': ...})" 
    或者字典结果: "({'key': 'val'}, {'model_thought': ...})"
    我们需要提取前面的结果部分，去掉后面的元数据字典。
    """
    if not content: return ""
    content = content.strip()
    
    # 1. 尝试使用 ast.literal_eval 解析 Python Tuple
    try:
        # 这是一个比较安全的解析方式
        parsed = ast.literal_eval(content)
        if isinstance(parsed, tuple) and len(parsed) > 0:
            # 无论第一个元素是字符串还是字典，都转为字符串返回
            # 如果是字典，ensure_ascii=False 保证中文正常显示（如果有）
            result = parsed[0]
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
    except:
        pass
        
    # 2. Regex/String Fallback
    # 如果 AST 解析失败（可能是因为内容包含特殊字符导致语法错误），尝试手动切割
    # 典型的 Tuple 字符串格式: (RESULT, METADATA)
    # METADATA 通常以 {'image_path': ... 开始，或者 {'model_thought': ...
    
    if content.startswith("(") and content.endswith(")"):
        # 寻找分割点： ", {"
        # 我们从右边开始找，因为 Result 内容里可能有 ", {"，但 Metadata 一定在最后
        last_comma_brace = content.rfind(", {")
        
        # 简单的启发式检查：如果找到了 ", {"，且位置比较靠后
        if last_comma_brace > 1:
            # 提取 Result 部分：去掉开头的 "(" 和 分割点之后的 ", {..."
            candidate = content[1:last_comma_brace]
            
            # 如果 Result 是被引号包围的字符串 "..." 或 '...'，去掉引号
            if (candidate.startswith("'") and candidate.endswith("'")) or \
               (candidate.startswith('"') and candidate.endswith('"')):
                # 再次检查长度，防止切片错误
                if len(candidate) >= 2:
                    candidate = candidate[1:-1]
                    # 处理常见的转义字符
                    candidate = candidate.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
            
            return candidate.strip()

    # 如果都失败了，为了不报错，返回原始内容（虽然可能包含脏数据，但比 crash 好）
    # 但在 SFT 场景下，宁可要原始数据也不要空数据
    return content

def _anonymize_paths_in_text(s: str, simulated_path: str) -> str:
    """
    将文本中的真实路径、/mnt/ 路径替换为模拟路径或 Token
    """
    if not s or not isinstance(s, str): return s
    
    # 1. 匹配 MIMIC 风格路径
    mimic_pat = r"p\d+/p\d+/s\d+/[a-z0-9-]+\.(?:jpg|png|jpeg|dcm)"
    s = re.sub(mimic_pat, simulated_path, s, flags=re.IGNORECASE)
    
    # 2. 匹配 /mnt/ 绝对路径
    mnt_pat = r"/mnt/[^\s'\"{},]+" 
    s = re.sub(mnt_pat, simulated_path, s)
    
    return s

def clean_thought_text(text: str) -> str:
    if not text: return ""
    return text.strip()

def contains_hallucinated_symptoms(sample: Dict[str, Any]) -> bool:
    """
    检查样本的思考过程（Thought）中是否包含了幻觉症状。
    逻辑：如果 Thought 中出现了 symptom_keywords 中的词，但这些词并没有出现在 Question 中，
    则判定为幻觉。
    """
    # 定义幻觉敏感词（针对 MIMIC-CXR-VQA 的常见脑补词）
    symptom_keywords = [
        "cough", "fever", "shortness of breath", "dyspnea", 
        "chest pain", "symptom", "history of", "chills", 
        "weight loss", "night sweats", "clinical history"
    ]
    
    question_text = (sample.get("question") or "").lower()
    
    # 辅助检查函数
    def _is_hallucination(text: str) -> bool:
        if not text: return False
        text_lower = text.lower()
        for kw in symptom_keywords:
            if kw in text_lower:
                # 只有当这个关键词完全不在 Question 里出现时，才算幻觉
                # 比如 Question: "Does the patient with cough have..." -> "cough" 不是幻觉
                if kw not in question_text:
                    return True
        return False

    # 1. 检查 Tool Trace 中的 Thought
    trace = sample.get("tool_trace", [])
    for ev in trace:
        if ev.get("event") == "tool_call":
            args = ev.get("args", {})
            thought = args.get("thought", "")
            if _is_hallucination(thought):
                return True

    # 2. 检查 Final Answer 前的 Recap/Thought (assistant_raw)
    raw_response = str(sample.get("assistant_raw") or sample.get("assistant_answer") or "")
    # 只需要检查 [FINAL] 之前的部分，因为 [FINAL] 之后是答案
    if "[FINAL]" in raw_response:
        thought_part = raw_response.split("[FINAL]")[0]
        if _is_hallucination(thought_part):
            return True
            
    return False

# --- Tools Manifest ---

def _schema_has_bad_items(schema: Any) -> bool:
    if isinstance(schema, dict):
        if schema.get("type") == "array" and isinstance(schema.get("items"), dict) and len(schema["items"]) == 0: return True
        for _, v in schema.items():
            if _schema_has_bad_items(v): return True
    elif isinstance(schema, list):
        return any(_schema_has_bad_items(v) for v in schema)
    return False

def _extract_args_schema_brief(args_schema: Any) -> Dict[str, Any]:
    if not isinstance(args_schema, dict): return {}
    out: Dict[str, Any] = {}
    if "properties" in args_schema: out["properties"] = args_schema["properties"]
    if "required" in args_schema: out["required"] = args_schema["required"]
    out["type"] = args_schema.get("type", "object")
    return out

def build_tools_field_from_manifest(manifest: Dict[str, Any], exclude_tools: Optional[List[str]] = None, drop_bad_schema: bool = True) -> str:
    tools = manifest.get("tools", [])
    if not tools: return "[]"
    exclude = set([t.strip() for t in (exclude_tools or []) if t.strip()])
    out_tools: List[Dict[str, Any]] = []
    for t in tools:
        name = (t.get("name") or "").strip()
        if not name or name in exclude: continue
        args_schema = t.get("args_schema")
        # 移除 thought 参数定义，因为这是内部隐式参数
        if "properties" in args_schema and "thought" in args_schema["properties"]:
            del args_schema["properties"]["thought"]
            if "required" in args_schema and "thought" in args_schema["required"]:
                args_schema["required"].remove("thought")

        if drop_bad_schema and _schema_has_bad_items(args_schema): continue
        params = _extract_args_schema_brief(args_schema)
        out_tools.append({"name": name, "description": (t.get("description") or "").strip(), "parameters": params if params else {"type": "object", "properties": {}}})
    return json.dumps(out_tools, ensure_ascii=False)

# =============================================================================
#  Logic: Filtering & Construction
# =============================================================================

def is_repetitive_call(current_tool: str, current_args: Dict, previous_calls: List[Tuple[str, Dict]]) -> bool:
    """
    检查是否与上一次调用完全一致（工具名相同且剔除thought后的参数相同）。
    """
    if not previous_calls:
        return False
    
    last_tool, last_args = previous_calls[-1]
    
    if current_tool != last_tool:
        return False
    
    # 比较参数 (已经移除了 thought)
    return current_args == last_args

def build_sharegpt_conversations(sample: Dict[str, Any], image_root: str) -> Tuple[Optional[List[Dict[str, str]]], Dict[str, Any]]:
    # --- 1. 基础信息提取 ---
    real_rel_path = (sample.get("image_path") or "").strip()
    simulated_path = generate_random_simulated_path()
    
    final_val = extract_final_answer_v2(sample)
    if not final_val or final_val.lower() == "unknown":
        return None, {"fail_reason": "final_answer_unknown"}
    
    question = (sample.get("question") or "").strip()
    
    # --- 2. 处理 Tool Trace ---
    raw_trace = sample.get("tool_trace", [])
    conv: List[Dict[str, str]] = []
    
    # 添加 Human Message
    conv.append({"from": "human", "value": f"<image>\nImage Context: {simulated_path}\nQuestion: {question}"})
    
    previous_calls: List[Tuple[str, Dict]] = [] # 用于重复检测
    
    for i, ev in enumerate(raw_trace):
        evt_type = ev.get("event")
        
        if evt_type == "tool_call":
            tool_name = ev.get("tool_name")
            raw_args = ev.get("args", {}).copy()
            
            # 提取 Thinking
            thought = raw_args.pop("thought", None)
            
            # [Filter] 如果没有 thought，丢弃样本
            if not thought or not isinstance(thought, str) or not thought.strip():
                return None, {"fail_reason": "missing_thought_in_args"}
            
            # 清理参数中的路径
            clean_args = {}
            for k, v in raw_args.items():
                if isinstance(v, str):
                    clean_args[k] = _anonymize_paths_in_text(v, simulated_path)
                elif isinstance(v, list):
                    clean_args[k] = [(_anonymize_paths_in_text(x, simulated_path) if isinstance(x, str) else x) for x in v]
                else:
                    clean_args[k] = v
            
            # [Filter] 重复调用检查
            if is_repetitive_call(tool_name, clean_args, previous_calls):
                return None, {"fail_reason": "repetitive_tool_loop"}
            
            previous_calls.append((tool_name, clean_args))
            
            # 构造 Assistant (Think + Tool Call)
            fc_value = f"<think>\n{thought.strip()}\n</think>\n" + json.dumps({"name": tool_name, "arguments": clean_args}, ensure_ascii=False)
            conv.append({"from": "function_call", "value": fc_value})
            
        elif evt_type == "tool_result":
            # 清理 Observation
            raw_content = ev.get("content_text") or ev.get("content_preview") or ""
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)
                
            # [重要] 提取纯文本/JSON内容，去除 ('result', {meta}) 的元组包裹
            clean_content = clean_observation_content(raw_content)
            
            # 路径脱敏
            clean_content = _anonymize_paths_in_text(clean_content, simulated_path)
            
            conv.append({"from": "observation", "value": clean_content})
            
    # --- 3. 构建 Final Response ---
    raw_final_text = (sample.get("assistant_raw") or "").strip()
    
    if "[FINAL]" in raw_final_text:
        parts = raw_final_text.split("[FINAL]")
        final_thought = parts[0].strip()
    else:
        final_thought = raw_final_text
    
    final_thought = _anonymize_paths_in_text(final_thought, simulated_path)
    
    if not final_thought:
        final_thought = "Based on the findings, I can determine the answer."
        
    final_gpt_value = f"<think>\n{final_thought}\n</think>\n[FINAL] {final_val}"
    conv.append({"from": "gpt", "value": final_gpt_value})
    
    return conv, {"fail_reason": "ok"}

# =============================================================================
#  MAIN PROCESSING LOGIC
# =============================================================================

def main():
    global _IMAGE_PATH_TOKEN

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="Main log file")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--input-simple-jsonl", type=str, nargs='+', default=None, help="List of simple sample files")
    parser.add_argument("--max-simple-samples", type=int, default=300)

    parser.add_argument("--prompt-file", default="medrax/docs/system_prompts_distillation.txt")
    parser.add_argument("--prompt-section", default="MEDICAL_ASSISTANT_DISTILLATION_SFT")
    parser.add_argument("--tool-manifest", type=str, default=None)
    parser.add_argument("--exclude-tools", type=str, default="image_visualizer")
    
    # 过滤参数
    parser.add_argument("--filter-empty-gt", action="store_true", default=True)

    args = parser.parse_args()

    # --- 统计计数器 ---
    stats_dropped = defaultdict(int)
    stats_source = defaultdict(int)
    stats_tool_counts = defaultdict(int)
    seen_indices = set()
    kept_count = 0
    total_processed = 0

    # 加载 System Prompt 和 Tools
    try:
        raw_system_text = extract_section_prompt(args.prompt_file, args.prompt_section)
        system_text = "\n".join([l for l in raw_system_text.split('\n') if "[CONTROL]" not in l]).strip()
    except Exception as e:
        print(f"Warning: Could not load system prompt: {e}. Using default.")
        system_text = "You are a helpful medical assistant."

    tools_field = "[]"
    if args.tool_manifest:
        with open(args.tool_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        tools_field = build_tools_field_from_manifest(manifest, exclude_tools=args.exclude_tools.split(","))

    def process_line(sample, source_type):
        nonlocal total_processed
        total_processed += 1

        # 1. 基础状态检查
        if sample.get("status") != "ok":
            stats_dropped["bad_status"] += 1
            return None

        # 2. [Filter] 必须是正确样本
        is_correct = sample.get("is_correct", False)
        if not is_correct:
            stats_dropped["incorrect_answer"] += 1
            return None

        # 3. [Filter] 过滤无效的 Ground Truth
        if args.filter_empty_gt:
            gt_raw = sample.get("gt_answer_raw")
            gt_norm = sample.get("gt_answer_norm")
            has_gt = (isinstance(gt_raw, list) and len(gt_raw) > 0) or \
                     (isinstance(gt_norm, list) and len(gt_norm) > 0)
            if not has_gt:
                stats_dropped["empty_gt"] += 1
                return None

        # --- [NEW] 4. 幻觉检查 (Hallucination Filter) ---
        # 必须把这一步放在非常前面，保证数据的纯净度
        if contains_hallucinated_symptoms(sample):
            stats_dropped["hallucinated_symptoms"] += 1
            return None

        # --- [NEW] 5. "Yes" 样本降采样 (Downsample 'Yes') ---
        # 仅针对 Gemini 生成的数据，且答案为 Yes 的情况
        final_ans = extract_final_answer_v2(sample).lower().strip().strip('.').strip()
        # 检查是否是单纯的 "yes"
        if source_type == "gemini" and final_ans in ["yes"]:
            # 保留概率 1/3 (0.33)，即扔掉 0.67
            if random.random() > 0.33: 
                stats_dropped["downsample_yes_class_imbalance"] += 1
                return None

        # 6. [Filter] 工具数量过滤逻辑
        trace = sample.get("tool_trace", [])
        actual_tool_calls = [t for t in trace if t.get("event") == "tool_call"]
        n_tools = len(actual_tool_calls)

        # 规则 A: 超过 4 次的全部删掉
        if n_tools > 4:
            stats_dropped["filter_tool_gt_4"] += 1
            return None

        # 规则 B: 等于 4 次的砍掉一半
        if n_tools == 4:
            if random.random() < 0.5:
                stats_dropped["filter_tool_eq_4_halved"] += 1
                return None
        
        # 7. 路径检查
        rel = (sample.get("image_path") or "").strip()
        if not rel:
            stats_dropped["missing_image_path"] += 1
            return None

        # 8. 构建对话
        res = build_sharegpt_conversations(sample, image_root=args.image_root)
        
        if res is None or res[0] is None:
            reason = res[1]["fail_reason"] if res else "unknown_build_error"
            stats_dropped[reason] += 1
            return None
        
        conv, build_stats = res

        return {
            "conversations": conv,
            "system": system_text,
            "tools": tools_field,
            "images": [rel],
            "meta": {
                "source": source_type,
                "idx": sample.get("idx"),
                "num_tools": n_tools,
                # "ans_type": final_ans # 方便后续debug看分布
            }
        }

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        # === Phase 1: Process GEMINI Log ===
        print(f"Reading: {args.input_jsonl}...")
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try: 
                    sample = json.loads(line)
                except: 
                    continue
                
                out = process_line(sample, "gemini")
                
                if out:
                    idx = str(sample.get("idx", ""))
                    if idx: seen_indices.add(idx)
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    kept_count += 1
                    stats_source["gemini"] += 1
                    stats_tool_counts[out["meta"]["num_tools"]] += 1

        # === Phase 2: Process SIMPLE (Optional) ===
        if args.input_simple_jsonl:
            simple_files = args.input_simple_jsonl
            simple_total_kept = 0
            
            for s_file in simple_files:
                if simple_total_kept >= args.max_simple_samples: break
                print(f"Reading simple: {s_file}...")
                
                if not os.path.exists(s_file): continue
                
                with open(s_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if simple_total_kept >= args.max_simple_samples: break
                        try:
                            sample = json.loads(line)
                        except: continue

                        idx = str(sample.get("idx", ""))
                        if idx and idx in seen_indices:
                            continue 
                        
                        out = process_line(sample, "simple")
                        
                        if out:
                            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                            kept_count += 1
                            simple_total_kept += 1
                            stats_source["simple"] += 1
                            stats_tool_counts[out["meta"]["num_tools"]] += 1
                            if idx: seen_indices.add(idx)

    # === Report ===
    print("\n" + "="*50)
    print(f"DONE. Final SFT samples: {kept_count}")
    print(f"Total processed lines: {total_processed}")
    print("-" * 50)
    print("Source Distribution:")
    for k, v in stats_source.items():
        print(f"  {k}: {v}")
    
    print("\nDrop Statistics:")
    sorted_drops = sorted(stats_dropped.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_drops:
        print(f"  {reason:30s}: {count}")

    print("\nTool Usage Distribution:")
    for k in sorted(stats_tool_counts.keys()):
        print(f"  {k} tools: {stats_tool_counts[k]}")
    print("="*50)

if __name__ == "__main__":
    main()