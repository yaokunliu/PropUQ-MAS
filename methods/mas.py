from __future__ import annotations

import argparse
import re
from typing import Dict, List, Optional

from . import Agent, default_agents
from mas_graph import build_mas_graph
from models import ModelWrapper
from prompts import (
    build_agent_messages_graph_text_mas,
    build_agent_messages_hierarchical_text_mas,
    build_agent_messages_sequential_text_mas,
)
from utils import (
    extract_gsm8k_answer,
    extract_markdown_python_block,
    extract_mcq_choice,
    normalize_answer,
    run_with_timeout,
    strip_structured_uncertainty_blocks,
)

LOGIT_UQ_METHODS = ("MSP",)
MAS_PROMPT_CHOICES = ("norole", "role")

_SEQUENTIAL_AGENT_SPECS = (
    {"name": "Planner", "role": "planner", "context_label": "Planner", "prompt_label": "Planner Agent"},
    {"name": "Critic", "role": "critic", "context_label": "Critic", "prompt_label": "Critic Agent"},
    {"name": "Refiner", "role": "refiner", "context_label": "Refiner", "prompt_label": "Refiner Agent"},
    {"name": "Judger", "role": "judger", "context_label": "Judger", "prompt_label": "Judger Agent"},
)
_HIERARCHICAL_AGENT_SPECS = (
    {"name": "Planner", "role": "planner", "context_label": "Math Agent", "prompt_label": "Math Agent"},
    {"name": "Critic", "role": "critic", "context_label": "Science Agent", "prompt_label": "Science Agent"},
    {"name": "Refiner", "role": "refiner", "context_label": "Code Agent", "prompt_label": "Code Agent"},
    {"name": "Judger", "role": "judger", "context_label": "Task Summarizer", "prompt_label": "Task Summarizer"},
)


def _selected_uq_methods(args) -> List[str]:
    value = getattr(args, "local_uncertainty_modes", getattr(args, "local_uncertainty_mode", None))
    if value is None:
        return ["Verb"]
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    out: List[str] = []
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        if text not in out:
            out.append(text)
    return out or ["Verb"]


def _uses_logit_uq(args) -> bool:
    return any(method in LOGIT_UQ_METHODS for method in _selected_uq_methods(args))


def resolve_mas_prompt_mode(args) -> str:
    prompt = str(getattr(args, "mas_prompt", "norole") or "norole").strip().lower()
    if prompt != "role":
        return "graph"
    if getattr(args, "mas_node_num", None) != 4:
        return "graph"
    topology = str(getattr(args, "mas_topology", "") or "").strip().lower()
    if topology == "sequential":
        return "legacy_sequential"
    if topology == "hierarchical":
        return "legacy_hierarchical"
    return "graph"


def resolve_mas_prompt(args) -> str:
    return "role" if resolve_mas_prompt_mode(args) != "graph" else "norole"


def _legacy_agent_specs_for_mode(prompt_mode: str):
    if prompt_mode == "legacy_sequential":
        return _SEQUENTIAL_AGENT_SPECS
    if prompt_mode == "legacy_hierarchical":
        return _HIERARCHICAL_AGENT_SPECS
    return None


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
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.args = args
        self.method_name = "mas"
        self.task = args.task
        self.prompt_mode = resolve_mas_prompt_mode(args)
        self.graph = build_mas_graph(
            args.mas_topology,
            args.mas_node_num,
        )
        legacy_specs = _legacy_agent_specs_for_mode(self.prompt_mode)
        if legacy_specs is None:
            self.agents = default_agents(args.mas_node_num)
            self._context_labels = {agent.index: agent.name for agent in self.agents}
            self._prompt_labels = {agent.index: agent.name for agent in self.agents}
        else:
            self.agents = [
                Agent(index=idx, name=spec["name"], role=spec["role"])
                for idx, spec in enumerate(legacy_specs)
            ]
            self._context_labels = {
                idx: spec["context_label"] for idx, spec in enumerate(legacy_specs)
            }
            self._prompt_labels = {
                idx: spec["prompt_label"] for idx, spec in enumerate(legacy_specs)
            }

    def _is_legacy_prompt_mode(self) -> bool:
        return self.prompt_mode in {"legacy_sequential", "legacy_hierarchical"}

    def _context_label(self, agent_idx: int) -> str:
        return self._context_labels.get(agent_idx, f"Agent {agent_idx + 1}")

    def _prompt_label(self, agent_idx: int) -> str:
        return self._prompt_labels.get(agent_idx, f"Agent {agent_idx + 1}")

    @staticmethod
    def _clamp01(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _extract_local_uncertainty(
        cls,
        text: str,
    ) -> Optional[float]:
        if not text:
            return None

        for match in re.finditer(r"<local_uncertainty\b([^>]*)/?>", text, flags=re.IGNORECASE):
            attrs = match.group(1)
            score_match = re.search(r'\bscore="([01](?:\.\d+)?|0?\.\d+)"', attrs, flags=re.IGNORECASE)
            if score_match:
                return cls._clamp01(float(score_match.group(1)))

        score_match = re.search(
            r"(?im)^\s*local[_\s-]*uncertainty\s*[:=]\s*([01](?:\.\d+)?|0?\.\d+)\s*$",
            text,
        )
        if score_match:
            return cls._clamp01(float(score_match.group(1)))
        return None

    @classmethod
    def _extract_alpha_scores(
        cls,
        text: str,
    ) -> Dict[str, float]:
        if not text:
            return {}

        parsed: Dict[str, float] = {}
        for match in re.finditer(r"<alpha\b([^>]*)/?>", text, flags=re.IGNORECASE):
            attrs = match.group(1)
            agent_match = re.search(r'\bagent="([^"]+)"', attrs, flags=re.IGNORECASE)
            if not agent_match:
                continue
            score: Optional[float] = None
            score_match = re.search(r'\bscore="([01](?:\.\d+)?|0?\.\d+)"', attrs, flags=re.IGNORECASE)
            if score_match:
                score = cls._clamp01(float(score_match.group(1)))
            if score is not None:
                parsed[agent_match.group(1).strip()] = score
        return parsed

    def _default_alpha_scores(self, incoming_labels: List[str]) -> Dict[str, float]:
        return {label: 1.0 for label in incoming_labels}

    def _local_uncertainty_by_method(
        self,
        text: str,
        uq_stats: Dict | None,
    ) -> Dict[str, float | None]:
        selected = _selected_uq_methods(self.args)
        out: Dict[str, float | None] = {}
        for method in selected:
            if method == "Verb":
                out[method] = self._extract_local_uncertainty(text)
            elif method == "MSP":
                out[method] = None if uq_stats is None else uq_stats.get("msp")
        return out

    def _alpha(
        self,
        text: str,
        incoming_labels: List[str],
    ) -> Dict[str, float]:
        selected = _selected_uq_methods(self.args)
        if "Verb" in selected:
            return self._extract_alpha_scores(text)
        if any(method in LOGIT_UQ_METHODS for method in selected):
            return self._default_alpha_scores(incoming_labels)
        return {}

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _strip_uncertainty_metadata(text: str) -> str:
        if not text:
            return text
        out = re.sub(r"<local_uncertainty\b[^>]*>.*?</local_uncertainty>", "", text, flags=re.IGNORECASE | re.DOTALL)
        out = re.sub(r"<local_uncertainty\b[^>]*/>", "", out, flags=re.IGNORECASE)
        out = re.sub(r"<alpha\b[^>]*/>", "", out, flags=re.IGNORECASE)
        out = re.sub(
            r"(?im)^[ \t>*`-]*(?:local[_\s-]*uncertainty|alpha)\s*[:=]\s*[^\n\r]*$",
            "",
            out,
        )
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()

    def _context_text_for_next_agent(self, text: str) -> str:
        mode = getattr(self.args, "mas_inter_agent_think", "strip")
        cleaned = text.strip() if mode == "pass" else self._strip_think_blocks(text)
        return self._strip_uncertainty_metadata(cleaned)

    @staticmethod
    def _extract_answer_only(text: str) -> str:
        cleaned = text.strip()
        match = re.search(r"(?is)##\s*Answer\s*(.+)$", cleaned)
        if match:
            return match.group(1).strip()
        return cleaned

    def _incoming_context(self, trace_map: dict[int, dict]) -> str:
        blocks = []
        for source_idx, trace in sorted(trace_map.items()):
            answer_only = self._extract_answer_only(trace.get("peer_output", "") or trace.get("output", ""))
            if not answer_only:
                continue
            blocks.append(f"[Agent {source_idx + 1} Answer]\n{answer_only}")
        return "\n\n".join(blocks).strip()

    def _prediction_from_answer_text(self, text: str):
        clean_text = strip_structured_uncertainty_blocks(text or "")
        if self.task == "mbppplus":
            return extract_markdown_python_block(clean_text)
        if self.task == "gsm8k":
            return normalize_answer(extract_gsm8k_answer(clean_text))
        if self.task == "medqa":
            return normalize_answer(extract_mcq_choice(clean_text))
        return normalize_answer(extract_gsm8k_answer(clean_text))

    def _build_skipped_result(
        self,
        *,
        item: Dict,
        trace_map: dict[int, dict],
        history_context: str,
        graph_metadata: Dict,
        skip_reason: str,
    ) -> Dict:
        return {
            "question": item["question"],
            "gold": item.get("gold", ""),
            "solution": item.get("solution", ""),
            "context": history_context.strip(),
            "prediction": None,
            "raw_prediction": "",
            "agents": [trace_map[agent_idx] for agent_idx in sorted(trace_map)],
            "mas_graph": graph_metadata,
            "correct": False,
            "skipped": True,
            "skip_reason": skip_reason,
        }

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        selected_methods = _selected_uq_methods(self.args)
        trace_maps: List[dict[int, dict]] = [{} for _ in range(batch_size)]
        legacy_contexts = ["" for _ in range(batch_size)]
        history_contexts = ["" for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        skip_reasons: List[Optional[str]] = [None for _ in range(batch_size)]
        max_model_len = self.model.get_max_model_len()

        for agent in self.agents:
            active_indices = [idx for idx in range(batch_size) if skip_reasons[idx] is None]
            if not active_indices:
                break
            incoming_indices = self.graph.incoming(agent.index)
            incoming_labels = [self._prompt_label(idx) for idx in incoming_indices]
            outgoing_labels = [self._prompt_label(idx) for idx in self.graph.outgoing(agent.index)]
            if self.prompt_mode == "legacy_sequential":
                batch_messages = [
                    build_agent_messages_sequential_text_mas(
                        role=agent.role,
                        question=items[item_idx]["question"],
                        context=legacy_contexts[item_idx],
                        method=self.method_name,
                        args=self.args,
                    )
                    for item_idx in active_indices
                ]
            elif self.prompt_mode == "legacy_hierarchical":
                batch_messages = [
                    build_agent_messages_hierarchical_text_mas(
                        role=agent.role,
                        question=items[item_idx]["question"],
                        context=legacy_contexts[item_idx],
                        method=self.method_name,
                        args=self.args,
                    )
                    for item_idx in active_indices
                ]
            else:
                batch_messages = [
                    build_agent_messages_graph_text_mas(
                        agent_label=agent.name,
                        question=items[item_idx]["question"],
                        context=self._incoming_context(
                            {
                                idx: trace_maps[item_idx][idx]
                                for idx in incoming_indices
                                if idx in trace_maps[item_idx]
                            }
                        ),
                        incoming_agents=incoming_labels,
                        method=self.method_name,
                        args=self.args,
                    )
                    for item_idx in active_indices
                ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )
            active_prompt_lengths = attention_mask.sum(dim=1).to("cpu").tolist()

            overlong_positions = set()
            if max_model_len is not None:
                for pos, prompt_len in enumerate(active_prompt_lengths):
                    if prompt_len > max_model_len:
                        overlong_positions.add(pos)
                        item_idx = active_indices[pos]
                        skip_reasons[item_idx] = (
                            f"prompt_too_long: prompt length {prompt_len} exceeds model limit {max_model_len}"
                        )
                        trimmed_ids = input_ids[pos][attention_mask[pos].bool()].to("cpu").tolist()
                        trace_maps[item_idx][agent.index] = {
                            "index": agent.index,
                            "name": agent.name,
                            "role": agent.role,
                            "incoming_agents": incoming_labels,
                            "outgoing_agents": outgoing_labels,
                            "input": prompts[pos],
                            "input_ids": trimmed_ids,
                            "input_tokens": tokens_batch[pos],
                            "output": "",
                            "peer_output": "",
                            "local_uncertainty": None,
                            "local_uncertainty_by_method": {method: None for method in selected_methods},
                            "logits_uq_stats": None,
                            "alpha": self._alpha("", incoming_labels),
                            "skipped": True,
                            "skip_reason": skip_reasons[item_idx],
                        }

            kept_positions = [pos for pos in range(len(active_indices)) if pos not in overlong_positions]
            if not kept_positions:
                continue

            kept_prompts = [prompts[pos] for pos in kept_positions]
            kept_input_ids = input_ids[kept_positions]
            kept_attention_mask = attention_mask[kept_positions]
            kept_tokens_batch = [tokens_batch[pos] for pos in kept_positions]
            kept_item_indices = [active_indices[pos] for pos in kept_positions]
            if self.model.use_vllm:
                generated_texts, generation_details = self.model.vllm_generate_text_batch(
                    kept_prompts,
                    max_new_tokens=self.max_new_tokens_each,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    return_generation_details=True,
                )
            else:
                generated_texts, _, generation_details = self.model.generate_text_batch(
                    kept_input_ids,
                    kept_attention_mask,
                    max_new_tokens=self.max_new_tokens_each,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    return_generation_details=True,
                )

            parsed_outputs = []
            for kept_pos, item_idx in enumerate(kept_item_indices):
                text_out = generated_texts[kept_pos].strip()
                generation_detail = generation_details[kept_pos]
                peer_context = self._context_text_for_next_agent(text_out)
                if self._is_legacy_prompt_mode():
                    peer_output = peer_context
                else:
                    peer_output = self._extract_answer_only(peer_context)
                parsed_outputs.append(
                    {
                        "kept_pos": kept_pos,
                        "item_idx": item_idx,
                        "text_out": text_out,
                        "generation_detail": generation_detail,
                        "peer_output": peer_output,
                    }
                )

            for parsed_pos, parsed_output in enumerate(parsed_outputs):
                kept_pos = parsed_output["kept_pos"]
                item_idx = parsed_output["item_idx"]
                text_out = parsed_output["text_out"]
                generation_detail = parsed_output["generation_detail"]
                local_uncertainty_by_method = self._local_uncertainty_by_method(
                    text_out,
                    generation_detail.get("uq_stats"),
                )
                alpha = self._alpha(text_out, incoming_labels)
                local_uncertainty = local_uncertainty_by_method.get("Verb")
                trimmed_ids = kept_input_ids[kept_pos][kept_attention_mask[kept_pos].bool()].to("cpu").tolist()

                trace_maps[item_idx][agent.index] = {
                    "index": agent.index,
                    "name": agent.name,
                    "role": agent.role,
                    "incoming_agents": incoming_labels,
                    "outgoing_agents": outgoing_labels,
                    "input": kept_prompts[kept_pos],
                    "input_ids": trimmed_ids,
                    "input_tokens": kept_tokens_batch[kept_pos],
                    "output": text_out,
                    "peer_output": parsed_output["peer_output"],
                    "local_uncertainty": local_uncertainty,
                    "local_uncertainty_by_method": local_uncertainty_by_method,
                    "logits_uq_stats": generation_detail.get("uq_stats"),
                    "alpha": alpha,
                }

                if self.prompt_mode == "legacy_sequential":
                    formatted_output = f"[{self._context_label(agent.index)}]:\n{parsed_output['peer_output']}\n\n"
                    if agent.role != "judger":
                        history_contexts[item_idx] += formatted_output
                        legacy_contexts[item_idx] = formatted_output
                    else:
                        final_texts[item_idx] = text_out
                elif self.prompt_mode == "legacy_hierarchical":
                    formatted_output = f"[{self._context_label(agent.index)}]:\n{parsed_output['peer_output']}\n\n"
                    if agent.role != "judger":
                        history_contexts[item_idx] += formatted_output
                        legacy_contexts[item_idx] += formatted_output
                    else:
                        final_texts[item_idx] = text_out
                else:
                    history_contexts[item_idx] += f"[{agent.name}]\n{parsed_output['peer_output']}\n\n"
                if not self._is_legacy_prompt_mode() and agent.index == self.graph.node_num - 1:
                    final_texts[item_idx] = text_out

        results: List[Dict] = []
        graph_metadata = self.graph.to_metadata()
        for idx, item in enumerate(items):
            if skip_reasons[idx] is not None:
                results.append(
                    self._build_skipped_result(
                        item=item,
                        trace_map=trace_maps[idx],
                        history_context=history_contexts[idx],
                        graph_metadata=graph_metadata,
                        skip_reason=skip_reasons[idx],
                    )
                )
                continue
            final_text = final_texts[idx] or trace_maps[idx][self.graph.node_num - 1]["output"]
            pred = None
            final_text_for_eval = strip_structured_uncertainty_blocks(final_text)

            if self.task == "mbppplus":
                pred = pred if pred is not None else extract_markdown_python_block(final_text_for_eval)
                gold = item.get("gold", "")
                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    ok, error_msg = run_with_timeout(pred + "\n" + gold, timeout=10)
                print("=========================================")
                print(f"Question {idx}")
                print(f"error_msg: {error_msg}")
            elif self.task == "medqa":
                pred = pred if pred is not None else normalize_answer(extract_mcq_choice(final_text_for_eval))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None
            else:
                pred = pred if pred is not None else normalize_answer(extract_gsm8k_answer(final_text_for_eval))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "context": history_contexts[idx].strip(),
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": [trace_maps[idx][agent_idx] for agent_idx in sorted(trace_maps[idx])],
                    "mas_graph": graph_metadata,
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
