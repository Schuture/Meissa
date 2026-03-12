#!/usr/bin/env python3
"""
Convert MDAgents trace files to LLaMA Factory SFT format.

Generates 5 types of SFT data capturing the full multi-agent behavior chain:
  Type 1: Difficulty Assessment   — Question → Difficulty classification (ALL difficulties)
  Type 2: Expert Recruitment      — Question → Expert team composition (intermediate only)
  Type 3: Expert Analysis         — Question + Role → Expert initial opinion (intermediate only)
  Type 4: Multi-Agent Debate      — Other opinions → Debate decision + arguments (intermediate only)
  Type 5: Synthesis / Final Answer — Comprehensive answer with <think> reasoning (ALL difficulties)

All GPT messages use <think>reasoning</think>\\nanswer format for consistent SFT training.

Usage:
    python convert_traces_to_sft.py \\
        --inputs file1.jsonl:100 file2.jsonl:50 \\
        --output-dir sft_output/ \\
        --output-prefix combined \\
        --filter-truncated \\
        --types 1 2 3 4 5
"""

import json
import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


# ═══════════════════════════════════════════════════════════════════
#  Quality Filter
# ═══════════════════════════════════════════════════════════════════

class QualityFilter:
    """Comprehensive quality filtering for MDAgents traces."""

    TERMINAL_PUNCTUATION = set('.!?。！？)）]】"\'*`')

    HALLUCINATION_KEYWORDS = [
        "cough", "fever", "shortness of breath", "dyspnea",
        "chest pain", "symptom", "history of", "chills",
        "weight loss", "night sweats", "clinical history",
        "patient reports", "patient complains", "patient states",
    ]

    MIN_RESPONSE_LENGTH = 10  # characters

    def __init__(self):
        self.drop_stats = defaultdict(int)

    # ── Completeness ──────────────────────────────────────────

    def is_sample_complete(self, sample) -> bool:
        """Check all agents finished successfully (no API errors)."""
        agents = sample.get('agents', [])
        if not agents:
            return False

        for agent in agents:
            messages = agent.get('messages', [])
            if not messages:
                return False

            last_message = messages[-1]
            if last_message.get('role') != 'assistant':
                return False

            content = last_message.get('content', '')
            if isinstance(content, str) and 'Error in' in content and 'after 8 retries' in content:
                return False

        return True

    # ── Truncation ────────────────────────────────────────────

    def is_truncated(self, text: str) -> bool:
        """Check if a single text response appears truncated."""
        if not text or len(text.strip()) < 5:
            return True

        text = text.rstrip()

        # Unclosed markdown bold markers
        if text.count('**') % 2 != 0:
            return True

        # If the response contains "Answer:" with content after it,
        # the model completed its structured response — not truncated
        if re.search(r'\bAnswer:\s*\S', text):
            return False

        # For longer texts, check if it ends with terminal punctuation
        if len(text) > 50:
            last_char = text[-1]
            if last_char not in self.TERMINAL_PUNCTUATION:
                return True

        return False

    def check_truncation(self, sample) -> bool:
        """Check if the final agent response in the sample is truncated."""
        agents = sample.get('agents', [])
        if not agents:
            return True
        # Only check the last agent's last assistant message (the actual response)
        last_agent = agents[-1]
        for msg in reversed(last_agent.get('messages', [])):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if isinstance(content, str):
                    return self.is_truncated(content)
                break
        return False

    def check_truncation_for_role(self, sample, role_keyword) -> bool:
        """Check truncation for a specific agent role (not just last agent).
        Returns True if the agent with matching role has a truncated response."""
        for agent in sample.get('agents', []):
            if role_keyword in agent.get('agent_role', '').lower():
                for msg in reversed(agent.get('messages', [])):
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        if isinstance(content, str):
                            return self.is_truncated(content)
                        break
        return False

    # Per-SFT-type truncation: which agent role to check for each type
    TYPE_TRUNCATION_TARGETS = {
        '1':  None,          # difficulty assessor — almost never truncated, skip check
        '2':  None,          # recruiter outputs structured lists ending with "Hierarchy: Independent", not terminal punct
        '3':  None,          # per-expert check done inside converter
        '4':  None,          # debate multi-turn, skip check
        '5':  'last',        # moderator / final answer agent (last agent)
        '1r': 'recap',       # recap agent
        '2r': 'recap',
        '5r': 'recap',
    }

    def check_truncation_for_type(self, sample, sft_type) -> bool:
        """Per-type truncation check — only checks the agent relevant to
        each SFT type instead of blanket-dropping the entire sample."""
        target = self.TYPE_TRUNCATION_TARGETS.get(sft_type)
        if target is None:
            return False  # no truncation check for this type
        if target == 'last':
            return self.check_truncation(sample)
        return self.check_truncation_for_role(sample, target)

    # ── Hallucination ─────────────────────────────────────────

    def check_hallucination(self, sample) -> bool:
        """Check if agents mention symptoms/conditions not in the question."""
        question_text = (sample.get('question', '')).lower()

        for agent in sample.get('agents', []):
            for msg in agent.get('messages', []):
                if msg.get('role') == 'assistant':
                    content = str(msg.get('content', '')).lower()
                    for kw in self.HALLUCINATION_KEYWORDS:
                        if kw in content and kw not in question_text:
                            return True
        return False

    # ── Structure validation ──────────────────────────────────

    def validate_agent_structure(self, sample) -> Tuple[bool, str]:
        """Validate that the trace has expected agents for its difficulty."""
        agents = sample.get('agents', [])
        difficulty = sample.get('difficulty', 'unknown')

        if difficulty == 'basic':
            if len(agents) < 1:
                return False, "basic_too_few_agents"

        elif difficulty == 'intermediate':
            if len(agents) < 5:
                return False, "intermediate_too_few_agents"
            has_recruiter = any(
                a.get('agent_role', '').lower() == 'recruiter' for a in agents
            )
            has_summarizer = any(
                'summarizing and synthesizing' in a.get('instruction', '').lower()
                for a in agents
            )
            has_moderator = any(
                a.get('agent_role', '').lower() == 'moderator' for a in agents
            )
            if not has_recruiter:
                return False, "intermediate_missing_recruiter"
            if not has_summarizer:
                return False, "intermediate_missing_summarizer"
            if not has_moderator:
                return False, "intermediate_missing_moderator"

        return True, "ok"

    # ── Response quality ──────────────────────────────────────

    # Short debate protocol responses (yes/no/numbers for expert selection)
    _DEBATE_PROTOCOL_RE = re.compile(
        r'^(yes|no|(?:\d+(?:\s*,\s*\d+)*))$', re.IGNORECASE
    )

    def check_response_quality(self, sample) -> bool:
        """Check that all assistant responses meet minimum quality.

        Skips short debate protocol responses (yes/no/agent numbers) that
        are valid in intermediate multi-agent traces.
        """
        for agent in sample.get('agents', []):
            messages = agent.get('messages', [])
            if not messages:
                continue
            last_msg = messages[-1]
            if last_msg.get('role') == 'assistant':
                content = last_msg.get('content', '')
                if not content or not isinstance(content, str):
                    return False
                stripped = content.strip()
                # Allow short debate protocol responses
                if self._DEBATE_PROTOCOL_RE.match(stripped):
                    continue
                if len(stripped) < self.MIN_RESPONSE_LENGTH:
                    return False
        return True

    # ── Master filter ─────────────────────────────────────────

    def filter_sample(self, sample, config=None) -> Tuple[bool, str]:
        """
        Run all enabled filters on a sample.
        Returns (passes, drop_reason).
        """
        config = config or {}

        # 1. Completeness (always on)
        if not self.is_sample_complete(sample):
            self.drop_stats["incomplete_api_failure"] += 1
            return False, "incomplete_api_failure"

        # 2. Agent structure (always on)
        valid, reason = self.validate_agent_structure(sample)
        if not valid:
            self.drop_stats[reason] += 1
            return False, reason

        # 3. Response quality (always on)
        if not self.check_response_quality(sample):
            self.drop_stats["low_quality_response"] += 1
            return False, "low_quality_response"

        # 4. Truncation (opt-in via --filter-truncated)
        if config.get('filter_truncated', False):
            if self.check_truncation(sample):
                self.drop_stats["truncated_response"] += 1
                return False, "truncated_response"

        # 5. Hallucination (opt-in via --filter-hallucination)
        if config.get('filter_hallucination', False):
            if self.check_hallucination(sample):
                self.drop_stats["hallucinated_symptoms"] += 1
                return False, "hallucinated_symptoms"

        return True, "ok"


# ═══════════════════════════════════════════════════════════════════
#  Conversion Statistics
# ═══════════════════════════════════════════════════════════════════

class ConversionStats:
    """Track all conversion statistics."""

    TYPE_NAMES = {
        'type1': 'Difficulty Assessment',
        'type2': 'Expert Recruitment',
        'type3': 'Expert Analysis',
        'type4': 'Multi-Agent Debate',
        'type5': 'Synthesis / Final Answer',
        'type1r': 'Difficulty Recap',
        'type2r': 'Recruitment Recap',
        'type5r': 'Synthesis Recap',
    }

    def __init__(self):
        self.total_processed = 0
        self.total_passed_filter = 0
        self.drop_reasons = defaultdict(int)
        self.type_counts = defaultdict(int)
        self.difficulty_counts = defaultdict(int)
        self.expert_role_counts = defaultdict(int)
        self.debate_participated = 0
        self.debate_declined = 0

    def record_drop(self, reason):
        self.drop_reasons[reason] += 1

    def record_generated(self, type_name, meta=None):
        self.type_counts[type_name] += 1
        if meta:
            diff = meta.get('difficulty', '')
            if diff:
                self.difficulty_counts[diff] += 1
            role = meta.get('expert_role', '')
            if role:
                self.expert_role_counts[role] += 1
            if meta.get('type') == 'multi_agent_debate':
                if meta.get('participated', False):
                    self.debate_participated += 1
                else:
                    self.debate_declined += 1

    def print_report(self):
        total_dropped = sum(self.drop_reasons.values())
        total_generated = sum(self.type_counts.values())

        print("\n" + "=" * 60)
        print("CONVERSION STATISTICS")
        print("=" * 60)

        print(f"\nTotal samples processed:   {self.total_processed}")
        print(f"Passed quality filter:     {self.total_passed_filter}")
        print(f"Dropped by filters:        {total_dropped}")
        print(f"Total SFT samples output:  {total_generated}")

        if self.drop_reasons:
            print("\n--- Drop Reasons ---")
            for reason, count in sorted(self.drop_reasons.items(), key=lambda x: -x[1]):
                pct = 100 * count / max(self.total_processed, 1)
                print(f"  {reason:40s}: {count:6d} ({pct:5.1f}%)")

        print("\n--- SFT Type Distribution ---")
        for key in ['type1', 'type2', 'type3', 'type4', 'type5', 'type1r', 'type2r', 'type5r']:
            count = self.type_counts.get(key, 0)
            if count > 0:
                name = self.TYPE_NAMES.get(key, key)
                print(f"  {name:30s}: {count:6d}")

        if self.difficulty_counts:
            print("\n--- Difficulty Distribution ---")
            for diff, count in sorted(self.difficulty_counts.items()):
                print(f"  {diff:15s}: {count:6d}")

        if self.expert_role_counts:
            print("\n--- Expert Roles (top 10) ---")
            for role, count in sorted(self.expert_role_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  {role:30s}: {count:6d}")

        if self.debate_participated or self.debate_declined:
            total_debate = self.debate_participated + self.debate_declined
            print(f"\n--- Debate Statistics ---")
            print(f"  Total debate samples:    {total_debate}")
            print(f"  Participated in debate:  {self.debate_participated}")
            print(f"  Declined debate:         {self.debate_declined}")
            if total_debate > 0:
                pct = 100 * self.debate_participated / total_debate
                print(f"  Participation rate:      {pct:.1f}%")

        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
#  Trace Converter (6 SFT types)
# ═══════════════════════════════════════════════════════════════════

def _format_thinking_response(raw_response):
    """将各种格式的 LLM 输出转换为 '<think>xxx</think>\nyyy' 格式。

    支持的输入格式：
      1. 'Thought: xxx\n(Final) Answer: yyy'
      2. 'Thought: xxx' (无 Answer 标记 — 取最后一行为答案)
      3. '**Key Knowledge:**\nxxx\n**Total Analysis:**\nyyy' (intermediate 综合)
    不匹配任何格式时原样返回。
    """
    # Already formatted
    if '<think>' in raw_response:
        return raw_response

    # Pattern 0: lowercase "thought" without colon (malformed LLM output)
    # e.g. "thought\nThe study investigated..." where "thought" is a stray word
    m = re.match(r'[Tt]hought\s*\n(.+)', raw_response, re.DOTALL)
    if m:
        content = m.group(1).strip()
        lines = content.rsplit('\n', 1)
        if len(lines) == 2:
            thought = lines[0].strip()
            answer = lines[1].strip()
            return f"<think>{thought}</think>\n{answer}"
        # Single block of text — all goes into think, answer is empty
        return f"<think>{content}</think>\n"

    # Pattern 1: Thought: xxx ... (Final) Answer: yyy
    m = re.search(r'Thought:\s*(.+?)(?:Final\s+)?Answer:\s*(.+)', raw_response, re.DOTALL | re.IGNORECASE)
    if m:
        thought = m.group(1).strip()
        answer = m.group(2).strip()
        return f"<think>{thought}</think>\n{answer}"

    # Pattern 2: Thought: xxx (no explicit Answer marker — take last line as answer)
    m = re.match(r'Thought:\s*(.+)', raw_response, re.DOTALL)
    if m:
        content = m.group(1).strip()
        lines = content.rsplit('\n', 1)
        if len(lines) == 2:
            thought = lines[0].strip()
            answer = lines[1].strip()
            return f"<think>{thought}</think>\n{answer}"

    # Pattern 3: Key Knowledge + Total Analysis (intermediate synthesizer)
    # Handles **Key Knowledge:** ... **Total Analysis:** ... format
    clean = re.sub(r'\*{2,}', '', raw_response)  # strip markdown bold
    m = re.search(r'Key\s*Knowledge:?\s*(.+?)Total\s*Analysis:?\s*(.+)', clean, re.DOTALL | re.IGNORECASE)
    if m:
        knowledge = m.group(1).strip()
        analysis = m.group(2).strip()
        return f"<think>{knowledge}</think>\n{analysis}"

    return raw_response


def _fix_format_instructions(text):
    """Replace Thought:/Answer: format instructions with <think> format in human messages.

    LLaMA Factory requires <think>...</think> format for training, so human format
    instructions must match the GPT response format.
    """
    # Basic synthesis: "Provide your response in the following format:\nThought: ...\nAnswer: ..."
    text = re.sub(
        r'Provide your response in the following format:\s*\n\s*Thought:\s*<[^>]+>\s*\n\s*(?:Final\s+)?Answer:\s*<[^>]+>',
        'Provide your response in the following format:\n<think>your step-by-step reasoning</think>\nyour concise answer',
        text, flags=re.IGNORECASE
    )
    # Intermediate synthesis: "You should output in exactly the same format as: Key Knowledge:; Total Analysis:..."
    text = re.sub(
        r'You should output in exactly the same format as:\s*Key Knowledge.*?(?:Be concise[^.]*\.)?',
        'Provide your analysis in the following format:\n<think>Key Knowledge: [3-5 bullet points]\nTotal Analysis: [2-3 sentences]</think>\nyour final answer',
        text, flags=re.IGNORECASE | re.DOTALL
    )
    return text


def _get_default_system_prompt(images):
    """Return appropriate system prompt based on whether sample has images."""
    if images:
        return ("You are a helpful medical assistant that answers questions based on "
                "medical images. First think step by step, then provide a very brief "
                "answer (1-5 words).")
    return ("You are a helpful assistant that answers multiple choice questions "
            "about medical knowledge.")


def _extract_concise_answer(answer_text, question=''):
    """Post-process moderator answer to extract only the concise core answer.

    - MCQ (detected by (A)-(E) in question): extract "(X) option text"
    - yes/no/maybe answers (PubMedQA): extract just the label
    - Short answers (≤10 words): return as-is
    """
    if not answer_text or not answer_text.strip():
        return answer_text

    answer = answer_text.strip()

    # Already concise — skip
    if len(answer.split()) <= 10:
        return answer

    is_mcq = bool(re.search(r'\([A-E]\)', question))

    if is_mcq:
        # Find the LAST occurrence of (X) in the answer — that's typically the conclusion
        matches = list(re.finditer(r'\(([A-E])\)', answer))
        if matches:
            letter = matches[-1].group(1)
            # Get the full option text from the question
            opt_m = re.search(rf'\({letter}\)\s*([^(]+?)(?=\s*\([A-E]\)|$)', question)
            if opt_m:
                return f"({letter}) {opt_m.group(1).strip()}"
            return f"({letter})"

        # Try "answer is X" or "X)" patterns
        m = re.search(r'(?:correct answer is|answer is)\s*\(?([A-E])\)?', answer, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            opt_m = re.search(rf'\({letter}\)\s*([^(]+?)(?=\s*\([A-E]\)|$)', question)
            if opt_m:
                return f"({letter}) {opt_m.group(1).strip()}"
            return f"({letter})"

        # No letter found at all — return as-is
        return answer

    # Check for yes/no/maybe (PubMedQA-style)
    answer_lower = answer.lower().strip()
    if answer_lower.rstrip('.') in ('yes', 'no', 'maybe'):
        return answer

    # Long answer with yes/no/maybe — extract just the label
    for pattern in [
        r'(?:the answer is|answer:?)\s*(yes|no|maybe)',
        r'(?:conclusion|therefore)[,:]?\s*(yes|no|maybe)',
        r'\b(yes|no|maybe)\b\s*[.!]?\s*$',  # at the very end
    ]:
        m = re.search(pattern, answer_lower)
        if m:
            label = m.group(1)
            return label

    return answer


def _recover_option_letter(answer_text, question):
    """For MedQA: match moderator paragraph against question options to recover
    the (X) letter format when the moderator didn't include it.

    Uses containment check first, then token overlap on the last portion of the
    answer where the conclusion is typically stated.
    Returns the original answer_text if no confident match is found.
    """
    options = re.findall(r'\(([A-E])\)\s*([^(]+?)(?=\s*\([A-E]\)|$)', question)
    if not options:
        return answer_text

    answer_lower = answer_text.lower().strip()
    # Focus on last portion where the answer is typically stated
    last_part = answer_lower[-300:] if len(answer_lower) > 300 else answer_lower

    best_letter = None
    best_score = 0
    best_opt = ""

    for letter, opt_text in options:
        opt_clean = opt_text.strip().lower()
        score = 0
        # 1) Exact containment
        if opt_clean in answer_lower:
            score = len(opt_clean) * 2
            if opt_clean in last_part:
                score *= 2  # prioritize matches near the end
        else:
            # 2) Token overlap (>=60% of option words appear in last part)
            opt_words = set(re.findall(r'[a-z]{3,}', opt_clean))
            ans_words = set(re.findall(r'[a-z]{3,}', last_part))
            if opt_words:
                overlap = opt_words & ans_words
                ratio = len(overlap) / len(opt_words)
                if ratio >= 0.6:
                    score = int(ratio * len(opt_clean))

        if score > best_score:
            best_score = score
            best_letter = letter
            best_opt = opt_text.strip()

    if best_letter and best_score > 5:
        return f"({best_letter}) {best_opt}"

    return answer_text


def _format_difficulty_with_think(raw_response):
    """将难度评估输出转换为 '<think>reasoning</think>\nN) level' 格式。
    支持多种格式：
      - 'N) level\n\nreasoning...'
      - '**Difficulty: N) level**\n\n**Reasoning:** ...'
      - 'some text...\n1) **basic**:...'
    """
    # Strip markdown bold markers first to normalize the text
    text = raw_response.strip()
    clean = re.sub(r'\*{2,}', '', text).strip()

    # Step 1: Find the difficulty level in cleaned text
    level_pattern = r'(\d\)\s*(?:basic|intermediate|advanced))'
    level_match = re.search(level_pattern, clean, re.IGNORECASE)
    if not level_match:
        # Fallback: wrap entire response
        return f"<think>{clean}</think>"

    level = level_match.group(1).strip()

    # Step 2: Extract reasoning
    # Try explicit "Reasoning:" section first
    reasoning_match = re.search(r'Reasoning:?\s*(.+)', clean, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        if reasoning:
            return f"<think>{reasoning}</think>\n{level}"

    # Try: everything after the difficulty level line as reasoning
    after_level = clean[level_match.end():].strip()
    # Strip leading punctuation/colons
    after_level = re.sub(r'^[\s:#\-]+', '', after_level).strip()
    if after_level:
        return f"<think>{after_level}</think>\n{level}"

    # Try: everything before the difficulty level as reasoning
    before_level = clean[:level_match.start()].strip()
    # Remove prefixes like "Difficulty:", "Difficulty Assessment:"
    before_level = re.sub(r'^.*(?:Difficulty|Assessment)\s*:?\s*', '', before_level, flags=re.IGNORECASE).strip()
    if before_level:
        return f"<think>{before_level}</think>\n{level}"

    return f"<think>Based on the nature of this medical query, it can be classified accordingly.</think>\n{level}"


class TraceConverter:
    """Converts MDAgents traces to 5 types of LLaMA Factory SFT data."""

    def __init__(self):
        pass

    # ── Agent identification helpers ──────────────────────────

    def _find_difficulty_assessor(self, agents):
        """Find the difficulty assessor agent."""
        for agent in agents:
            if 'decide the difficulty' in agent.get('instruction', '').lower():
                return agent
        return None

    def _find_recruiter(self, agents):
        """Find the recruiter agent."""
        for agent in agents:
            if agent.get('agent_role', '').lower() == 'recruiter':
                return agent
        return None

    def _find_domain_experts(self, agents):
        """Find all domain expert agents (excluding utility agents)."""
        skip_roles = {'medical expert', 'recruiter', 'medical assistant', 'moderator', 'recap'}
        skip_instructions = {'helpful medical agent', 'decide the difficulty'}
        experts = []
        for agent in agents:
            role = agent.get('agent_role', '').lower()
            instruction = agent.get('instruction', '').lower()
            # Experts have dynamic roles like "vascular surgeon", "endocrinologist"
            if role and role not in skip_roles:
                experts.append(agent)
            # Edge case: role might still be "medical expert" but with expert-like instruction
            # We skip these since they're exemplar generators or assessors
        return experts

    def _find_summarizer(self, agents):
        """Find the summarizer/medical assistant agent."""
        for agent in agents:
            if 'summarizing and synthesizing' in agent.get('instruction', '').lower():
                return agent
        return None

    def _find_moderator(self, agents):
        """Find the moderator agent."""
        for agent in agents:
            if agent.get('agent_role', '').lower() == 'moderator':
                return agent
        return None

    def _find_final_answer_agent_basic(self, agents):
        """Find the final answer agent in basic traces (longest conversation, excluding assessor)."""
        max_len = 0
        final_agent = None
        for agent in agents:
            instruction = agent.get('instruction', '').lower()
            if 'decide the difficulty' in instruction:
                continue
            messages = agent.get('messages', [])
            if len(messages) > max_len:
                max_len = len(messages)
                final_agent = agent
        return final_agent

    def _extract_images(self, sample):
        """Extract all unique img_path from sample (top-level or in messages)."""
        images = []
        seen = set()
        # Check top-level img_path (added in newer traces)
        top_img = sample.get('img_path')
        if top_img and top_img not in seen:
            images.append(top_img)
            seen.add(top_img)
        # Check agent messages
        for agent in sample.get('agents', []):
            for msg in agent.get('messages', []):
                img_path = msg.get('img_path')
                if img_path and img_path not in seen:
                    images.append(img_path)
                    seen.add(img_path)
        return images

    @staticmethod
    def _extract_text_from_content(content):
        """Extract plain text from a message content (handles multipart lists)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get('type') == 'text':
                        parts.append(part['text'])
                    # skip image_url parts
                elif isinstance(part, str):
                    parts.append(part)
            return '\n'.join(parts)
        return str(content)

    def _build_sft_sample(self, conversations, images, meta, system=None):
        """Create a standardized SFT sample dict."""
        result = {
            "conversations": conversations,
            "images": images,
            "meta": meta,
        }
        if system:
            result["system"] = system
        return result

    # ── Difficulty prompt template (mirrors determine_difficulty in utils.py) ──
    _DIFFICULTY_PROMPT_TEMPLATE = (
        "Now, given the medical query (and potentially an image), you need to "
        "decide the difficulty/complexity of it:\n{question}.\n\n"
        "Please indicate the difficulty/complexity of the medical query among below options:\n"
        "1) basic: a single medical agent can output an answer based on the visual and text info.\n"
        "2) intermediate: number of medical experts with different expertise should "
        "discuss and make final decision.\n"
        "3) advanced: multiple teams of clinicians from different departments need to "
        "collaborate with each other to make final decision."
    )

    _DIFFICULTY_RESPONSE_BASIC = (
        "<think>This is a straightforward medical question that can be answered by "
        "a single medical expert based on the available visual and text information, "
        "without requiring multi-expert discussion or cross-department collaboration.</think>\n"
        "1) basic"
    )

    _DIFFICULTY_RESPONSE_INTERMEDIATE = (
        "<think>This medical question requires expertise from multiple medical "
        "specialists with different backgrounds to discuss and reach a consensus, "
        "as it involves complex clinical reasoning that benefits from multi-expert "
        "deliberation.</think>\n"
        "2) intermediate"
    )

    # ── Type 1: Difficulty Assessment ─────────────────────────

    def convert_type1_difficulty_assessment(self, sample, source_file):
        """
        Question → Difficulty classification + reasoning.
        Applicable to: ALL traces (basic + intermediate + advanced).
        For basic without assessor agent: synthesize the response.
        """
        agents = sample.get('agents', [])
        assessor = self._find_difficulty_assessor(agents)
        question = sample.get('question', '')
        images = self._extract_images(sample)
        difficulty = sample.get('difficulty', 'unknown')

        # Build human prompt
        human_value = ""
        if images:
            human_value += "<image>\n"
        human_value += self._DIFFICULTY_PROMPT_TEMPLATE.format(question=question)

        # Build GPT response
        if assessor:
            # Use actual assessor output, convert to <think> format
            messages = assessor.get('messages', [])
            assistant_response = None
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    assistant_response = msg.get('content', '')
                    break
            if not assistant_response:
                return None
            gpt_value = _format_difficulty_with_think(assistant_response)
        elif difficulty == 'basic':
            # Synthesize for basic traces without assessor
            gpt_value = self._DIFFICULTY_RESPONSE_BASIC
        elif difficulty == 'intermediate':
            # Synthesize for intermediate traces without assessor
            gpt_value = self._DIFFICULTY_RESPONSE_INTERMEDIATE
        else:
            # No assessor and unknown difficulty — skip
            return None

        return self._build_sft_sample(
            conversations=[
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            images=images,
            meta={
                "sample_id": str(sample.get('id', 'unknown')),
                "type": "difficulty_assessment",
                "difficulty": difficulty,
                "source_file": os.path.basename(source_file),
            },
        )

    # ── Type 2: Expert Recruitment ────────────────────────────

    def convert_type2_recruitment(self, sample, source_file):
        """
        Question → Expert team composition (roles, expertise, hierarchy).
        Applicable to: intermediate traces only.
        """
        if sample.get('difficulty') != 'intermediate':
            return None

        agents = sample.get('agents', [])
        recruiter = self._find_recruiter(agents)
        if not recruiter:
            return None

        messages = recruiter.get('messages', [])
        assistant_response = None
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                assistant_response = msg.get('content', '')
                break
        if not assistant_response:
            return None

        question = sample.get('question', '')
        images = self._extract_images(sample)

        human_value = ""
        if images:
            human_value += "<image>\n"
        human_value += (
            f"{question}\n\n"
            "You can recruit 3 experts in different medical expertise. "
            "What kind of experts will you recruit to solve this medical query? "
            "Please specify their roles, areas of expertise, and communication hierarchy."
        )

        gpt_value = _format_thinking_response(assistant_response)
        # Fallback: if still no <think> tags, wrap with generic reasoning
        if '<think>' not in gpt_value:
            gpt_value = (
                "<think>Based on the medical query, the following expert team "
                "is needed for comprehensive analysis.</think>\n"
                + assistant_response.strip()
            )

        return self._build_sft_sample(
            conversations=[
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            images=images,
            meta={
                "sample_id": str(sample.get('id', 'unknown')),
                "type": "expert_recruitment",
                "difficulty": "intermediate",
                "source_file": os.path.basename(source_file),
            },
        )

    # ── Type 3: Expert Analysis ───────────────────────────────

    def convert_type3_expert_analysis(self, sample, source_file) -> List[dict]:
        """
        Question + Expert Role → Expert initial opinion.
        Returns a list (one SFT sample per expert, typically 3).
        Applicable to: intermediate traces only.
        """
        if sample.get('difficulty') != 'intermediate':
            return []

        agents = sample.get('agents', [])
        experts = self._find_domain_experts(agents)
        if not experts:
            return []

        question = sample.get('question', '')
        images = self._extract_images(sample)
        results = []

        for expert in experts:
            role = expert.get('agent_role', 'unknown')
            instruction = expert.get('instruction', '')
            messages = expert.get('messages', [])

            # Expert messages: [0]=system, [1]=user(question+exemplars), [2]=assistant(initial opinion)
            if len(messages) < 3:
                continue

            # Find the first assistant response (initial opinion)
            initial_opinion = None
            for msg in messages:
                if msg.get('role') == 'assistant':
                    initial_opinion = msg.get('content', '')
                    break
            if not initial_opinion:
                continue

            # Skip truncated expert responses
            if isinstance(initial_opinion, str) and QualityFilter().is_truncated(initial_opinion):
                continue

            human_value = ""
            if images:
                human_value += "<image>\n"
            human_value += (
                f"You are a {role}. "
                f"Given the following medical query, provide your expert analysis.\n\n"
                f"{question}"
            )

            results.append(self._build_sft_sample(
                conversations=[
                    {"from": "human", "value": human_value},
                    {"from": "gpt", "value": _format_thinking_response(initial_opinion)},
                ],
                images=images,
                meta={
                    "sample_id": str(sample.get('id', 'unknown')),
                    "type": "expert_analysis",
                    "expert_role": role,
                    "difficulty": "intermediate",
                    "source_file": os.path.basename(source_file),
                },
            ))

        return results

    # ── Type 4: Multi-Agent Debate ────────────────────────────

    def convert_type4_debate(self, sample, source_file) -> List[dict]:
        """
        Other experts' opinions → Debate decision + arguments (multi-turn conversation).
        Returns a list (one SFT sample per expert's debate interaction).
        Applicable to: intermediate traces only.

        Expert message structure:
          [0] system
          [1] user (question + exemplars)
          [2] assistant (initial opinion)
          [3] user (debate prompt: "Given opinions... want to talk? yes/no")
          [4] assistant ("yes" / "no")
          [5] user (choose expert number)     — only if yes
          [6] assistant (number)              — only if yes
          [7] user (deliver opinion prompt)   — only if yes
          [8] assistant (persuasive argument) — only if yes
          ... may repeat for multiple rounds
        """
        if sample.get('difficulty') != 'intermediate':
            return []

        agents = sample.get('agents', [])
        experts = self._find_domain_experts(agents)
        if not experts:
            return []

        images = self._extract_images(sample)
        results = []

        for expert in experts:
            role = expert.get('agent_role', 'unknown')
            messages = expert.get('messages', [])

            # Debate starts at index 3 (after system, question, initial opinion)
            if len(messages) < 4:
                continue

            debate_messages = messages[3:]
            conversations = []

            # Build multi-turn conversation from debate messages
            i = 0
            while i < len(debate_messages):
                msg = debate_messages[i]
                if msg.get('role') == 'user':
                    # Look for the next assistant response
                    if i + 1 < len(debate_messages) and debate_messages[i + 1].get('role') == 'assistant':
                        human_content = msg.get('content', '')
                        gpt_content = debate_messages[i + 1].get('content', '')
                        # Handle "No\n\nThought: reasoning..." format before
                        # _format_thinking_response (which doesn't handle leading No)
                        _no_thought_m = re.match(
                            r'(?:Talk to .*?(?:yes/no)\s*:?\s*)?[Nn]o\.?\s*\n+\s*Thought:\s*(.+)',
                            gpt_content, re.DOTALL)
                        if _no_thought_m:
                            reasoning = _no_thought_m.group(1).strip()
                            # Strip trailing "Answer: ..." if present
                            reasoning = re.split(
                                r'(?:Final\s+)?Answer:', reasoning,
                                maxsplit=1, flags=re.IGNORECASE)[0].strip()
                            gpt_content = f"<think>{reasoning}</think>\nNo"
                        else:
                            # Apply <think> formatting to substantive responses
                            gpt_content = _format_thinking_response(gpt_content)
                        # Add template <think> for bare "No" debate responses
                        if gpt_content.strip().lower() in ('no', 'no.'):
                            gpt_content = (
                                "<think>After reviewing the other experts' opinions, "
                                "I find that their analyses are consistent with my "
                                "initial assessment. The existing evidence and reasoning "
                                "are sufficient to support the conclusion, so further "
                                "discussion is not necessary.</think>\nNo"
                            )
                        conversations.append({"from": "human", "value": human_content})
                        conversations.append({"from": "gpt", "value": gpt_content})
                        i += 2
                    else:
                        i += 1  # Skip orphan user message
                else:
                    i += 1

            if not conversations:
                continue

            # Prepend <image> token to first human turn (image was in original question)
            if images and conversations:
                conversations[0]["value"] = "<image>\n" + conversations[0]["value"]

            # Check first GPT response to determine if expert participated in debate
            first_gpt = conversations[1]["value"] if len(conversations) >= 2 else ""
            participated = 'yes' in first_gpt.lower().strip()[:20]

            results.append(self._build_sft_sample(
                conversations=conversations,
                images=images,
                meta={
                    "sample_id": str(sample.get('id', 'unknown')),
                    "type": "multi_agent_debate",
                    "expert_role": role,
                    "participated": participated,
                    "num_turns": len(conversations) // 2,
                    "difficulty": "intermediate",
                    "source_file": os.path.basename(source_file),
                },
            ))

        return results

    # ── Type 5: Synthesis / Final Answer ──────────────────────

    def convert_type5_synthesis(self, sample, source_file):
        """
        Final comprehensive answer with <think> reasoning.
        Applicable to: ALL traces (basic + intermediate + advanced).

        Basic:        exemplars (if any) as human context → final answer
        Intermediate: expert opinions → Key Knowledge + Total Analysis
        """
        difficulty = sample.get('difficulty', 'unknown')
        agents = sample.get('agents', [])
        images = self._extract_images(sample)

        if difficulty == 'basic':
            return self._convert_type5_basic(sample, agents, images, source_file)
        elif difficulty in ('intermediate', 'advanced'):
            return self._convert_type5_intermediate(sample, agents, images, source_file)
        return None

    def _convert_type5_basic(self, sample, agents, images, source_file):
        """Type 5 for basic traces: single agent answer with exemplars in human context."""
        final_agent = self._find_final_answer_agent_basic(agents)
        if not final_agent:
            return None

        messages = final_agent.get('messages', [])

        # Extract system instruction
        system_instruction = None
        non_system_msgs = []
        for msg in messages:
            if msg['role'] == 'system':
                system_instruction = self._extract_text_from_content(msg.get('content', ''))
            else:
                non_system_msgs.append(msg)

        if len(non_system_msgs) < 2:
            return None

        # Last user+assistant pair is the actual Q&A
        # Everything before is exemplar context
        actual_user = self._extract_text_from_content(non_system_msgs[-2].get('content', ''))
        actual_assistant = self._extract_text_from_content(non_system_msgs[-1].get('content', ''))

        if non_system_msgs[-2].get('role') != 'user' or non_system_msgs[-1].get('role') != 'assistant':
            return None

        # Build human message: just the actual question (no few-shot exemplars)
        human_parts = []
        if images:
            human_parts.append("<image>")
        human_parts.append(_fix_format_instructions(actual_user))
        human_value = "\n".join(human_parts)

        # GPT response with <think> formatting
        gpt_value = _format_thinking_response(actual_assistant)

        return self._build_sft_sample(
            conversations=[
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            images=images,
            meta={
                "sample_id": str(sample.get('id', 'unknown')),
                "type": "synthesis",
                "difficulty": "basic",
                "source_file": os.path.basename(source_file),
            },
            system=system_instruction,
        )

    def _convert_type5_intermediate(self, sample, agents, images, source_file):
        """Type 5 for intermediate traces: moderator answer + summarizer reasoning.

        Uses the moderator's concise answer as the output and the summarizer's
        analysis as the <think> reasoning block.  Falls back to summarizer-only
        if no moderator is present.
        """
        moderator = self._find_moderator(agents)
        summarizer = self._find_summarizer(agents)

        if not moderator and not summarizer:
            return None

        # --- Extract moderator answer ---
        mod_response = None
        if moderator:
            for msg in moderator.get('messages', []):
                if msg.get('role') == 'assistant':
                    mod_response = self._extract_text_from_content(msg.get('content', ''))

        # --- Extract summarizer analysis ---
        sum_user_content = None
        sum_analysis = None
        if summarizer:
            for msg in summarizer.get('messages', []):
                if msg.get('role') == 'user':
                    sum_user_content = msg.get('content', '')
                elif msg.get('role') == 'assistant':
                    sum_analysis = self._extract_text_from_content(msg.get('content', ''))

        # Need at least one source of answer
        if not mod_response and not sum_analysis:
            return None

        # --- Build human prompt: original question + expert reports ---
        question = sample.get('question', '')
        human_value = ""
        if images:
            human_value += "<image>\n"
        # Prepend original question so the model sees what it's answering
        if question:
            human_value += question + "\n\n"
        if sum_user_content:
            human_value += _fix_format_instructions(sum_user_content)
        else:
            human_value += "Synthesize the expert opinions above."

        # --- Build gpt value ---
        if mod_response and sum_analysis:
            # Best case: summarizer reasoning as <think>, moderator answer as output
            # Clean summarizer: strip Thought:/Answer: markers to get pure analysis
            clean_sum = re.sub(r'^(?:Thought|Key\s*Knowledge):?\s*', '', sum_analysis,
                               flags=re.IGNORECASE)
            clean_sum = re.split(r'(?:Final\s+)?Answer:', clean_sum,
                                 maxsplit=1, flags=re.IGNORECASE)[0].strip()

            # Extract moderator's concise answer
            mod_formatted = _format_thinking_response(mod_response)
            if '<think>' in mod_formatted and '</think>' in mod_formatted:
                answer_part = mod_formatted.split('</think>', 1)[1].strip()
            else:
                answer_part = mod_formatted.strip()

            # For MedQA: if answer_part doesn't contain option letter, try to
            # match against question options to recover the letter format
            question = sample.get('question', '')
            if question and 'Options:' in question and not re.search(r'\(?[A-E]\)', answer_part):
                answer_part = _recover_option_letter(answer_part, question)

            # Post-process: truncate verbose answer to concise core
            answer_part = _extract_concise_answer(answer_part, question)

            gpt_value = f"<think>{clean_sum}</think>\n{answer_part}"
        elif mod_response:
            # Moderator only (no summarizer)
            gpt_value = _format_thinking_response(mod_response)
        else:
            # Fallback: summarizer only (no moderator)
            gpt_value = _format_thinking_response(sum_analysis)

        # System prompt: infer from whether sample has images
        system_instruction = _get_default_system_prompt(images)

        return self._build_sft_sample(
            conversations=[
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            images=images,
            meta={
                "sample_id": str(sample.get('id', 'unknown')),
                "type": "synthesis",
                "difficulty": sample.get('difficulty', 'intermediate'),
                "source_file": os.path.basename(source_file),
            },
            system=system_instruction,
        )

    # ── Recap Types (1R, 2R, 5R) ─────────────────────────────────

    def _extract_recap_section(self, agents, section_name):
        """从 recap agent 的 response 中提取指定 section。

        支持两种格式：
          1. 带闭合标签: [SECTION]...[/SECTION]
          2. 无闭合标签: [SECTION]...（到下一个 [TAG] 或结尾）
        """
        for agent in agents:
            if agent.get('agent_role', '').lower() == 'recap':
                for msg in agent.get('messages', []):
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        # Try with closing tag first
                        pattern = rf'\[{section_name}\](.*?)\[/{section_name}\]'
                        m = re.search(pattern, content, re.DOTALL)
                        if m:
                            return m.group(1).strip()
                        # Fallback: no closing tag — capture until next [TAG] or end
                        pattern2 = rf'\[{section_name}\]\s*(.*?)(?=\n\s*\[(?:DIFFICULTY|RECRUITMENT|SYNTHESIS)_RECAP\]|\[/|$)'
                        m2 = re.search(pattern2, content, re.DOTALL)
                        if m2:
                            text = m2.group(1).strip()
                            if len(text) > 20:  # avoid empty/trivial matches
                                return text
        return None

    def convert_type1r_difficulty_recap(self, sample, source_file):
        """Type 1R: 与 Type 1 完全相同格式，只替换 <think> 为 hindsight recap。"""
        recap_think = self._extract_recap_section(sample.get('agents', []), 'DIFFICULTY_RECAP')
        if not recap_think:
            return None

        difficulty = sample.get('difficulty', '')
        question = sample.get('question', '')
        images = self._extract_images(sample)

        human_value = ""
        if images:
            human_value += "<image>\n"
        human_value += (
            f"Now, given the medical query (and potentially an image), you need to "
            f"decide the difficulty/complexity of it:\n{question}.\n\n"
            f"Please indicate the difficulty/complexity of the medical query among below options:\n"
            f"1) basic: a single medical agent can output an answer based on the visual and text info.\n"
            f"2) intermediate: number of medical experts with different expertise should discuss and make final decision.\n"
            f"3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."
        )

        level_map = {'basic': '1) basic', 'intermediate': '2) intermediate', 'advanced': '3) advanced'}
        level = level_map.get(difficulty, '2) intermediate')
        gpt_value = f"<think>{recap_think}</think>\n{level}"

        return self._build_sft_sample(
            conversations=[
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            images=images,
            meta={
                "sample_id": str(sample.get('id', 'unknown')),
                "type": "difficulty_recap",
                "difficulty": difficulty,
                "source_file": os.path.basename(source_file),
            },
        )

    def convert_type2r_recruitment_recap(self, sample, source_file):
        """Type 2R: 与 Type 2 完全相同格式，只替换 <think> 为 hindsight recap。仅 intermediate。"""
        if sample.get('difficulty', '') != 'intermediate':
            return None

        recap_think = self._extract_recap_section(sample.get('agents', []), 'RECRUITMENT_RECAP')
        if not recap_think:
            return None

        agents = sample.get('agents', [])
        question = sample.get('question', '')
        images = self._extract_images(sample)

        # 从 recruiter agent 提取原始 expert list
        recruiter_response = ""
        for agent in agents:
            if agent.get('agent_role', '').lower() == 'recruiter':
                for msg in agent.get('messages', []):
                    if msg.get('role') == 'assistant':
                        recruiter_response = msg.get('content', '')
                        break
        if not recruiter_response:
            return None

        human_value = ""
        if images:
            human_value += "<image>\n"
        human_value += (
            f"{question}\n\nYou can recruit 3 experts in different medical expertise. "
            f"What kind of experts will you recruit to solve this medical query? "
            f"Please specify their roles, areas of expertise, and communication hierarchy."
        )

        gpt_value = f"<think>{recap_think}</think>\n{recruiter_response}"

        return self._build_sft_sample(
            conversations=[
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            images=images,
            meta={
                "sample_id": str(sample.get('id', 'unknown')),
                "type": "recruitment_recap",
                "difficulty": "intermediate",
                "source_file": os.path.basename(source_file),
            },
        )

    def convert_type5r_synthesis_recap(self, sample, source_file):
        """Type 5R: 与 Type 5 完全相同格式，只替换 <think> 为 hindsight recap。"""
        recap_think = self._extract_recap_section(sample.get('agents', []), 'SYNTHESIS_RECAP')
        if not recap_think:
            return None

        difficulty = sample.get('difficulty', '')
        agents = sample.get('agents', [])
        question = sample.get('question', '')
        images = self._extract_images(sample)

        if difficulty == 'intermediate':
            # 从 moderator 提取简洁答案，从 summarizer 提取 user prompt
            human_value, answer_part = "", ""

            # Extract moderator's concise answer
            moderator = self._find_moderator(agents)
            if moderator:
                for msg in moderator.get('messages', []):
                    if msg.get('role') == 'assistant':
                        raw = self._extract_text_from_content(msg.get('content', ''))
                        formatted = _format_thinking_response(raw)
                        if '<think>' in formatted and '</think>' in formatted:
                            answer_part = formatted.split('</think>', 1)[1].strip()
                        else:
                            answer_part = formatted.strip()

            # Fallback to summarizer answer if no moderator
            if not answer_part:
                for agent in agents:
                    if agent.get('agent_role', '').lower() == 'medical assistant':
                        for msg in agent.get('messages', []):
                            if msg.get('role') == 'assistant':
                                raw = self._extract_text_from_content(msg.get('content', ''))
                                formatted = _format_thinking_response(raw)
                                if '</think>' in formatted:
                                    answer_part = formatted.split('</think>', 1)[1].strip()

            # Extract summarizer's user prompt (contains expert reports)
            for agent in agents:
                if agent.get('agent_role', '').lower() == 'medical assistant':
                    for msg in agent.get('messages', []):
                        if msg.get('role') == 'user':
                            user_content = msg.get('content', '')
                            if 'reports from different' in user_content.lower():
                                human_value = _fix_format_instructions(user_content)

            if not human_value:
                human_value = "Synthesize the expert opinions above."
            if not answer_part:
                return None

            # Prepend original question so model sees what it's answering
            if question:
                human_value = question + "\n\n" + human_value

            # For MedQA: recover option letter if missing
            if question and 'Options:' in question and not re.search(r'\(?[A-E]\)', answer_part):
                answer_part = _recover_option_letter(answer_part, question)

            # Post-process: truncate verbose answer to concise core
            answer_part = _extract_concise_answer(answer_part, question)

            if images:
                human_value = "<image>\n" + human_value

            gpt_value = f"<think>{recap_think}</think>\n{answer_part}"

            # System prompt for intermediate
            system_instruction = _get_default_system_prompt(images)

            return self._build_sft_sample(
                conversations=[
                    {"from": "human", "value": human_value},
                    {"from": "gpt", "value": gpt_value},
                ],
                images=images,
                meta={
                    "sample_id": str(sample.get('id', 'unknown')),
                    "type": "synthesis_recap",
                    "difficulty": "intermediate",
                    "source_file": os.path.basename(source_file),
                },
                system=system_instruction,
            )

        elif difficulty == 'basic':
            # 从 medical expert 提取 human msg 和 answer
            human_value, answer_part, system_instruction = "", "", ""
            for agent in agents:
                if agent.get('agent_role', '').lower() == 'medical expert':
                    for msg in agent.get('messages', []):
                        if msg.get('role') == 'system':
                            system_instruction = msg.get('content', '')
                    user_msgs = [m for m in agent.get('messages', []) if m.get('role') == 'user']
                    if user_msgs:
                        human_value = user_msgs[-1].get('content', '')
                    asst_msgs = [m for m in agent.get('messages', []) if m.get('role') == 'assistant']
                    if asst_msgs:
                        raw = asst_msgs[-1].get('content', '')
                        formatted = _format_thinking_response(raw)
                        if '</think>' in formatted:
                            answer_part = formatted.split('</think>', 1)[1].strip()
                        else:
                            answer_part = raw.strip()
                    break

            if not human_value or not answer_part:
                return None

            # Build human message: just the actual question (no few-shot exemplars)
            full_human = _fix_format_instructions(human_value)
            if images:
                full_human = "<image>\n" + full_human

            gpt_value = f"<think>{recap_think}</think>\n{answer_part}"

            return self._build_sft_sample(
                conversations=[
                    {"from": "human", "value": full_human},
                    {"from": "gpt", "value": gpt_value},
                ],
                images=images,
                system=system_instruction if system_instruction else None,
                meta={
                    "sample_id": str(sample.get('id', 'unknown')),
                    "type": "synthesis_recap",
                    "difficulty": "basic",
                    "source_file": os.path.basename(source_file),
                },
            )

        return None


# ═══════════════════════════════════════════════════════════════════
#  Utility Functions
# ═══════════════════════════════════════════════════════════════════

def load_correct_ids(results_dirs, dataset):
    """从一个或多个 JSON 结果目录加载正确样本的 ID 集合。

    复用 utils.py 中的 _check_correct() 进行正确性判断。
    results_dirs 可以是单个路径字符串或路径列表。
    """
    # Import from utils.py in the same directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import _check_correct

    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]

    correct_ids = set()
    total = 0
    for results_dir in results_dirs:
        dir_correct = 0
        dir_total = 0
        for jf in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
            if '_traces' in jf or 'exampler_cache' in jf:
                continue
            with open(jf, 'r', encoding='utf-8') as f:
                results = json.load(f)
            for r in results:
                total += 1
                dir_total += 1
                if _check_correct(r, dataset):
                    correct_ids.add(str(r['id']))
                    dir_correct += 1
        print(f"[INFO] Correct filter: {dir_correct}/{dir_total} correct from {results_dir}")
    print(f"[INFO] Total correct IDs: {len(correct_ids)}/{total}")
    return correct_ids


def parse_input_spec(spec_str):
    """Parse input specification like 'file.jsonl:100'. Returns (filepath, limit)."""
    if ':' in spec_str:
        filepath, limit_str = spec_str.rsplit(':', 1)
        try:
            limit = int(limit_str)
        except ValueError:
            filepath = spec_str
            limit = None
    else:
        filepath = spec_str
        limit = None
    return filepath, limit


def load_traces_from_files(input_specs):
    """Load traces from multiple files with per-file limits."""
    all_traces = []

    for spec in input_specs:
        filepath, limit = parse_input_spec(spec)
        if not os.path.exists(filepath):
            print(f"[WARN] File not found: {filepath}, skipping...")
            continue

        print(f"[INFO] Loading from {filepath} (limit: {limit if limit else 'all'})")

        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and count >= limit:
                    break
                try:
                    sample = json.loads(line)
                    all_traces.append((sample, filepath))
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"[WARN] Failed to parse line in {filepath}: {e}")

        print(f"[INFO] Loaded {count} samples from {filepath}")

    return all_traces


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

OUTPUT_FILE_MAP = {
    '1': 'type1_difficulty.jsonl',
    '2': 'type2_recruitment.jsonl',
    '3': 'type3_expert_analysis.jsonl',
    '4': 'type4_debate.jsonl',
    '5': 'type5_synthesis.jsonl',
    '1r': 'type1r_difficulty_recap.jsonl',
    '2r': 'type2r_recruitment_recap.jsonl',
    '5r': 'type5r_synthesis_recap.jsonl',
}


def main():
    parser = argparse.ArgumentParser(
        description='Convert MDAgents traces to LLaMA Factory SFT format (5 types)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_traces_to_sft.py \\
    --inputs outputs/pubmedqa_train_gemini/gemini-3-flash-preview/*_traces.jsonl \\
    --output-dir sft_output/ \\
    --output-prefix pubmedqa_gemini

  python convert_traces_to_sft.py \\
    --inputs file1.jsonl:100 file2.jsonl:50 \\
    --output-dir sft_output/ \\
    --output-prefix combined \\
    --filter-truncated --filter-hallucination

Output files (in --output-dir):
  {prefix}_type1_difficulty.jsonl       Difficulty Assessment (all difficulties)
  {prefix}_type2_recruitment.jsonl      Expert Recruitment (intermediate)
  {prefix}_type3_expert_analysis.jsonl  Expert Analysis (intermediate)
  {prefix}_type4_debate.jsonl           Multi-Agent Debate (intermediate)
  {prefix}_type5_synthesis.jsonl        Synthesis / Final Answer (all difficulties)
        """,
    )

    parser.add_argument('--inputs', nargs='+', required=True,
                        help='Input trace files with optional limits (file.jsonl:100)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for SFT files')
    parser.add_argument('--output-prefix', default='sft',
                        help='Prefix for output filenames (default: sft)')

    # Filter controls
    parser.add_argument('--filter-truncated', action='store_true', default=False,
                        help='Filter out samples with truncated responses')
    parser.add_argument('--filter-hallucination', action='store_true', default=False,
                        help='Filter out samples with hallucinated symptoms')
    parser.add_argument('--filter-correct-from', nargs='+', default=None,
                        help='Results dir(s): only convert traces for correctly answered samples. '
                             'Accepts multiple directories (e.g., basic_dir intermediate_dir).')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['medqa', 'pubmedqa', 'pathvqa', 'mimic-cxr-vqa'],
                        help='Dataset name (required when using --filter-correct-from)')

    # Type selection
    parser.add_argument('--types', nargs='+', default=['1', '2', '3', '4', '5', '1r', '2r', '5r'],
                        help='Which SFT types to generate (default: all, including recap types 1r/2r/5r)')

    args = parser.parse_args()
    selected_types = set(args.types)

    # Validate --filter-correct-from requires --dataset
    if args.filter_correct_from and not args.dataset:
        parser.error("--dataset is required when using --filter-correct-from")

    print("=" * 60)
    print("MDAgents Trace → LLaMA Factory SFT Converter (v3)")
    print("=" * 60)
    print(f"  Types:              {', '.join(sorted(selected_types))}")
    print(f"  Filter truncated:   {args.filter_truncated}")
    print(f"  Filter hallucination: {args.filter_hallucination}")
    if args.filter_correct_from:
        for d in args.filter_correct_from:
            print(f"  Filter correct from: {d}")
        print(f"  Dataset:            {args.dataset}")
    print()

    # ── Initialize ────────────────────────────────────────────
    quality_filter = QualityFilter()
    converter = TraceConverter()
    stats = ConversionStats()

    filter_config = {
        'filter_truncated': args.filter_truncated,
        'filter_hallucination': args.filter_hallucination,
    }

    type_results = {t: [] for t in selected_types}

    # ── Load correct IDs (if filtering) ───────────────────────
    correct_ids = None
    if args.filter_correct_from:
        print("[STEP 0] Loading correct sample IDs for filtering...")
        correct_ids = load_correct_ids(args.filter_correct_from, args.dataset)
        print()

    # ── Load ──────────────────────────────────────────────────
    print("[STEP 1] Loading traces...")
    traces = load_traces_from_files(args.inputs)
    print(f"[INFO] Loaded {len(traces)} total samples\n")

    # ── Convert ───────────────────────────────────────────────
    print("[STEP 2] Filtering and converting...")

    # Split filter config: hard filters (completeness, structure, quality) always
    # apply at sample level.  Truncation is applied per-SFT-type so that a
    # truncated Moderator (Type 5) doesn't kill valid Type 2/3/4 data.
    hard_filter_config = {
        k: v for k, v in filter_config.items() if k != 'filter_truncated'
    }
    use_per_type_truncation = filter_config.get('filter_truncated', False)

    for sample, source_file in traces:
        stats.total_processed += 1

        # Correctness filtering (only keep correct samples)
        if correct_ids is not None:
            sample_id = str(sample.get('id', ''))
            if sample_id not in correct_ids:
                stats.record_drop("incorrect_answer")
                continue

        # Hard quality filtering (completeness, structure, response quality)
        passes, reason = quality_filter.filter_sample(sample, hard_filter_config)
        if not passes:
            stats.record_drop(reason)
            continue
        stats.total_passed_filter += 1

        # Helper: check per-type truncation before conversion
        def _type_ok(sft_type):
            if not use_per_type_truncation:
                return True
            if quality_filter.check_truncation_for_type(sample, sft_type):
                stats.record_drop(f"truncated_{sft_type}")
                return False
            return True

        # Generate each requested type
        try:
            if '1' in selected_types and _type_ok('1'):
                result = converter.convert_type1_difficulty_assessment(sample, source_file)
                if result:
                    type_results['1'].append(result)
                    stats.record_generated('type1', result.get('meta'))

            if '2' in selected_types and _type_ok('2'):
                result = converter.convert_type2_recruitment(sample, source_file)
                if result:
                    type_results['2'].append(result)
                    stats.record_generated('type2', result.get('meta'))

            if '3' in selected_types and _type_ok('3'):
                results = converter.convert_type3_expert_analysis(sample, source_file)
                for r in results:
                    type_results['3'].append(r)
                    stats.record_generated('type3', r.get('meta'))

            if '4' in selected_types and _type_ok('4'):
                results = converter.convert_type4_debate(sample, source_file)
                for r in results:
                    type_results['4'].append(r)
                    stats.record_generated('type4', r.get('meta'))

            if '5' in selected_types and _type_ok('5'):
                result = converter.convert_type5_synthesis(sample, source_file)
                if result:
                    type_results['5'].append(result)
                    stats.record_generated('type5', result.get('meta'))

            # Recap types (1R, 2R, 5R) — only for samples with a recap agent
            if '1r' in selected_types and _type_ok('1r'):
                result = converter.convert_type1r_difficulty_recap(sample, source_file)
                if result:
                    type_results['1r'].append(result)
                    stats.record_generated('type1r', result.get('meta'))

            if '2r' in selected_types and _type_ok('2r'):
                result = converter.convert_type2r_recruitment_recap(sample, source_file)
                if result:
                    type_results['2r'].append(result)
                    stats.record_generated('type2r', result.get('meta'))

            if '5r' in selected_types and _type_ok('5r'):
                result = converter.convert_type5r_synthesis_recap(sample, source_file)
                if result:
                    type_results['5r'].append(result)
                    stats.record_generated('type5r', result.get('meta'))

        except Exception as e:
            print(f"[ERROR] Failed to process sample {sample.get('id')}: {e}")
            stats.record_drop("conversion_error")

    # ── Write output ──────────────────────────────────────────
    print(f"\n[STEP 3] Writing output files to {args.output_dir}/")
    os.makedirs(args.output_dir, exist_ok=True)

    for type_key in sorted(selected_types):
        samples = type_results.get(type_key, [])
        filename = f"{args.output_prefix}_{OUTPUT_FILE_MAP[type_key]}"
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f"  {filename:45s} {len(samples):6d} samples")

    # ── Merge filter stats into main stats ────────────────────
    for reason, count in quality_filter.drop_stats.items():
        stats.drop_reasons[reason] = count

    stats.print_report()
    print("\n[DONE] Conversion complete!")


if __name__ == '__main__':
    main()
