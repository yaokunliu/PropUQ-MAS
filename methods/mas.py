from typing import Dict, List, Optional, Union
import re

from . import default_agents
from models import ModelWrapper
from prompts import (
    build_agent_messages_hierarchical_text_mas,
    build_agent_messages_hierarchical_text_mas_gemma,
    build_agent_messages_sequential_text_mas,
    build_agent_messages_sequential_text_mas_gemma,
)
from utils import (
    extract_gsm8k_answer,
    extract_mcq_choice,
    normalize_answer,
    extract_markdown_python_block,
    run_with_timeout,
    strip_structured_uncertainty_blocks,
)
import argparse


class MASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens_each: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.model = model
        self.max_new_tokens_each = max_new_tokens_each
        self.max_new_tokens_judger = max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.args = args
        self.method_name = "mas"
        self.task = args.task

    @staticmethod
    def _clamp01(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _anchor_to_score(level: str) -> Optional[float]:
        if not level:
            return None
        return {
            "VL": 0.0,
            "L": 0.25,
            "M": 0.5,
            "H": 0.75,
            "VH": 1.0,
        }.get(level.upper())

    @classmethod
    def _extract_agent_uncertainty(
        cls,
        text: str,
        *,
        preserve_anchor: bool = False,
    ) -> Optional[Union[float, str]]:
        if not text:
            return None

        for match in re.finditer(r"<agent_uncertainty\b([^>]*)/?>", text, flags=re.IGNORECASE):
            attrs = match.group(1)
            score_match = re.search(
                r'\bscore="([01](?:\.\d+)?|0?\.\d+)"',
                attrs,
                flags=re.IGNORECASE,
            )
            if score_match:
                return cls._clamp01(float(score_match.group(1)))

            level_match = re.search(
                r'\blevel="(VL|L|M|H|VH)"',
                attrs,
                flags=re.IGNORECASE,
            )
            if level_match:
                level = level_match.group(1).upper()
                return level if preserve_anchor else cls._clamp01(cls._anchor_to_score(level))

        score_match = re.search(
            r"(?im)^\s*(?:agent'?s?\s+uncertainty|self[_\s-]*uncertainty)\s*[:=]\s*([01](?:\.\d+)?|0?\.\d+)\s*$",
            text,
        )
        if score_match:
            return cls._clamp01(float(score_match.group(1)))

        level_match = re.search(
            r"(?im)^\s*(?:agent'?s?\s+uncertainty|self[_\s-]*uncertainty)\s*[:=]\s*(VL|L|M|H|VH)\s*$",
            text,
        )
        if level_match:
            level = level_match.group(1).upper()
            return level if preserve_anchor else cls._clamp01(cls._anchor_to_score(level))

        confidence_match = re.search(
            r'<agent_confidence\s+score="([01](?:\.\d+)?|0?\.\d+)"\s*(?:/>|>)',
            text,
            flags=re.IGNORECASE,
        )
        if confidence_match:
            return cls._clamp01(1.0 - float(confidence_match.group(1)))

        return None

    @classmethod
    def _extract_message_adoption_scores(
        cls,
        text: str,
        *,
        preserve_anchor: bool = False,
    ) -> Dict[str, Union[float, str]]:
        if not text:
            return {}

        parsed: Dict[str, Union[float, str]] = {}
        for match in re.finditer(r"<message_adoption\b([^>]*)/?>", text, flags=re.IGNORECASE):
            attrs = match.group(1)
            agent_match = re.search(r'\bagent="([^"]+)"', attrs, flags=re.IGNORECASE)
            if not agent_match:
                continue
            score: Optional[Union[float, str]] = None
            score_match = re.search(
                r'\bscore="([01](?:\.\d+)?|0?\.\d+)"',
                attrs,
                flags=re.IGNORECASE,
            )
            if score_match:
                score = cls._clamp01(float(score_match.group(1)))
            else:
                level_match = re.search(
                    r'\blevel="(VL|L|M|H|VH)"',
                    attrs,
                    flags=re.IGNORECASE,
                )
                if level_match:
                    level = level_match.group(1).upper()
                    score = level if preserve_anchor else cls._clamp01(cls._anchor_to_score(level))
            if score is None:
                continue
            parsed[agent_match.group(1).strip()] = score
        return parsed

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _context_text_for_next_agent(self, text: str) -> str:
        mode = getattr(self.args, "mas_inter_agent_think", "strip")
        cleaned = text.strip() if mode == "pass" else self._strip_think_blocks(text)
        return self._sanitize_uncertainty_for_peer(cleaned)

    @staticmethod
    def _strip_uncertainty_metadata(text: str) -> str:
        if not text:
            return text

        out = re.sub(
            r"<agent_uncertainty\b[^>]*>.*?</agent_uncertainty>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        out = re.sub(r"<agent_uncertainty\b[^>]*/>", "", out, flags=re.IGNORECASE)
        out = re.sub(r"<peer_influence\b[^>]*/>", "", out, flags=re.IGNORECASE)
        out = re.sub(r"<message_uncertainty\b[^>]*/>", "", out, flags=re.IGNORECASE)
        out = re.sub(r"<message_adoption\b[^>]*/>", "", out, flags=re.IGNORECASE)
        out = re.sub(
            r"<message_uncertainty_explanation\b[^>]*>.*?</message_uncertainty_explanation>",
            "",
            out,
            flags=re.IGNORECASE | re.DOTALL,
        )
        out = re.sub(r"<agent_confidence\b[^>]*>.*?</agent_confidence>", "", out, flags=re.IGNORECASE | re.DOTALL)
        out = re.sub(r"<agent_confidence\b[^>]*/>", "", out, flags=re.IGNORECASE)
        out = re.sub(r"<message_confidence\b[^>]*/>", "", out, flags=re.IGNORECASE)

        meta_keys = (
            "agent_uncertainty",
            "message_adoption",
            "peer_influence",
            "message_uncertainty",
            "agent_confidence",
            "message_confidence",
        )
        key_group = "|".join(meta_keys)
        out = re.sub(rf"(?im)^[ \t>*`-]*<?(?:{key_group})\b[^\n\r]*$", "", out)
        out = re.sub(rf"(?i)[ \t]*<?(?:{key_group})\b[^\n\r]*", "", out)
        out = re.sub(
            r"(?im)^[ \t>*`-]*(?:agent'?s?\s+uncertainty|self[_\s-]*uncertainty|message\s+adoption)\s*[:=]\s*[^\n\r]*$",
            "",
            out,
        )
        out = re.sub(
            r"(?im)^[ \t>*`-]*message[_\s-]*adoption[_\s-]*weight\s*[:=]\s*[^\n\r]*$",
            "",
            out,
        )
        tag_group = (
            "evidence_gap|memory_gap|reasoning_gap|ambiguous_task|context_conflict|tool_risk|weak_signal|guess|"
            "verify|critique|selective_use|balanced_use|follow|execute|refine|low_relevance"
        )
        out = re.sub(rf"(?im)^[ \t>*`-]*tag\s*[:=]\s*(?:{tag_group})\s*$", "", out)
        out = re.sub(
            r"(?i)[ \t]*(?:agent'?s?\s+uncertainty|self[_\s-]*uncertainty|message\s+adoption)\s*[:=]\s*[^\n\r]*",
            "",
            out,
        )
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()

    def _sanitize_uncertainty_for_peer(self, text: str) -> str:
        return self._strip_uncertainty_metadata(text)

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        contexts = ["" for _ in range(batch_size)]
        history_contexts = ["" for _ in range(batch_size)]
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        for agent in self.agents:
            current_max_new_tokens = (
                self.max_new_tokens_judger if agent.role == "judger" else self.max_new_tokens_each
            )

            model_name = str(getattr(self.args, "model_name", "")).lower()
            use_instruct_template = "gemma" in model_name
            if self.args.prompt == "hierarchical":
                prompt_builder = (
                    build_agent_messages_hierarchical_text_mas_gemma
                    if use_instruct_template
                    else build_agent_messages_hierarchical_text_mas
                )
                batch_messages = [
                    prompt_builder(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=self.args,
                    )
                    for idx, item in enumerate(items)
                ]
            else:
                prompt_builder = (
                    build_agent_messages_sequential_text_mas_gemma
                    if use_instruct_template
                    else build_agent_messages_sequential_text_mas
                )
                batch_messages = [
                    prompt_builder(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=self.args,
                    )
                    for idx, item in enumerate(items)
                ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if self.model.use_vllm:
                generated_texts = self.model.vllm_generate_text_batch(
                    prompts,
                    max_new_tokens=current_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            else:
                generated_texts, _ = self.model.generate_text_batch(
                    input_ids,
                    attention_mask,
                    max_new_tokens=current_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            agent_name_map_for_prompt_hierarchical = {
                "Planner": "Math Agent",
                "Critic": "Science Agent",
                "Refiner": "Code Agent",
                "Judger": "Task Summrizer",
                "planner": "Math Agent",
                "critic": "Science Agent",
                "refiner": "Code Agent",
                "judger": "Task Summrizer",
            }

            for idx in range(batch_size):
                text_out = generated_texts[idx].strip()
                context_text = self._context_text_for_next_agent(text_out)
                preserve_anchor = getattr(self.args, "uncertainty_mode", None) == "anchor"
                self_uncertainty = self._extract_agent_uncertainty(
                    text_out,
                    preserve_anchor=preserve_anchor,
                )
                message_adoption = self._extract_message_adoption_scores(
                    text_out,
                    preserve_anchor=preserve_anchor,
                )

                if self.args.prompt == "hierarchical":
                    formatted_output = f"[{agent_name_map_for_prompt_hierarchical[agent.name]}]:\n{context_text}\n\n"
                else:
                    formatted_output = f"[{agent.name}]:\n{context_text}\n\n"

                if agent.role != "judger":
                    history_contexts[idx] = f"{history_contexts[idx]}{formatted_output}"
                    if self.args.prompt == "hierarchical":
                        contexts[idx] = f"{contexts[idx]}{formatted_output}"
                    else:
                        contexts[idx] = formatted_output
                else:
                    final_texts[idx] = text_out
                mask = attention_mask[idx].bool()
                trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
                agent_traces[idx].append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "input": prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": tokens_batch[idx],
                        "output": text_out,
                        "self_uncertainty": self_uncertainty,
                        "message_adoption": message_adoption,
                    }
                )

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            final_text_for_eval = strip_structured_uncertainty_blocks(final_text)

            if self.task in ["mbppplus", "humanevalplus"]:
                pred = extract_markdown_python_block(final_text_for_eval)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)

                print(f"=========================================")
                print(f"Question {idx}")
                print(f"error_msg: {error_msg}")

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text_for_eval))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = pred_int == gold_int
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f"Value error in parsing answer. Pred: {pred}, Gold: {gold}"

            elif self.task == "gpqa":
                pred = normalize_answer(extract_mcq_choice(final_text_for_eval))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text_for_eval))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "context": history_contexts[idx],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
