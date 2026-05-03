import re
from typing import Dict, List

from models import ModelWrapper
from prompts import build_agent_messages_single_agent, build_agent_messages_single_agent_gemma
from utils import (
    extract_gsm8k_answer,
    extract_mcq_choice,
    normalize_answer,
    extract_markdown_python_block,
    run_with_timeout,
    strip_structured_uncertainty_blocks,
)

LOGIT_UQ_METHODS = ("MSP", "NLL")


def _selected_uq_methods(args) -> List[str]:
    value = getattr(args, "uncertainty_modes", getattr(args, "uncertainty_mode", None))
    if value is None:
        return ["ASK4CONF"]
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    out: List[str] = []
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        if text.lower() == "continuous":
            text = "ASK4CONF"
        if text not in out:
            out.append(text)
    return out or ["ASK4CONF"]


def _uses_logit_uq(args) -> bool:
    return any(method in LOGIT_UQ_METHODS for method in _selected_uq_methods(args))


def _extract_agent_uncertainty(text: str):
    if not text:
        return None
    match = re.search(
        r'<agent_uncertainty\b[^>]*\bscore="([01](?:\.\d+)?|0?\.\d+)"[^>]*/?>',
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    match = re.search(
        r"(?im)^\s*(?:agent'?s?\s+uncertainty|self[_\s-]*uncertainty)\s*[:=]\s*([01](?:\.\d+)?|0?\.\d+)\s*$",
        text,
    )
    if match:
        return float(match.group(1))
    match = re.search(
        r'<agent_confidence\s+score="([01](?:\.\d+)?|0?\.\d+)"\s*(?:/>|>)',
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return 1.0 - float(match.group(1))
    return None


def _strip_think_blocks(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _nll_target_text(text: str) -> str:
    return (text or "").strip()


def _self_uncertainty_by_method(args, generated_text: str, uq_stats: Dict | None) -> Dict[str, float | None]:
    selected = _selected_uq_methods(args)
    out: Dict[str, float | None] = {}
    for method in selected:
        if method == "ASK4CONF":
            out[method] = _extract_agent_uncertainty(generated_text)
        elif method == "MSP":
            out[method] = None if uq_stats is None else uq_stats.get("msp")
        elif method == "NLL":
            out[method] = None if uq_stats is None else uq_stats.get("nll")
    return out


class SingleAgentMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        use_vllm: bool = False,
        args=None,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.use_vllm = use_vllm
        self.method_name = "single_agent"
        self.args = args
        self.task = args.task

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        selected_methods = _selected_uq_methods(self.args)
        model_name = str(getattr(self.args, "model_name", "")).lower()
        use_instruct_template = "gemma" in model_name
        prompt_builder = (
            build_agent_messages_single_agent_gemma
            if use_instruct_template
            else build_agent_messages_single_agent
        )
        batch_messages = [
            prompt_builder(question=item["question"], args=self.args)
            for item in items
        ]
        prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
            batch_messages, add_generation_prompt=True
        )
        if self.use_vllm:
            generated_batch, generation_details = self.model.vllm_generate_text_batch(
                prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                return_generation_details=True,
            )
        else:
            generated_batch, _, generation_details = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                return_generation_details=True,
            )

        results: List[Dict] = []

        for idx, item in enumerate(items):
            generated_text = generated_batch[idx]
            generation_detail = generation_details[idx]
            nll_target_stat = None
            if "NLL" in selected_methods:
                nll_target_stat = self.model.target_uq_stats_from_generation(
                    generation_text=generated_text,
                    generated_token_ids=list(generation_detail.get("generated_token_ids") or []),
                    token_logprobs=list(generation_detail.get("token_logprobs") or []),
                    target_text=_nll_target_text(generated_text),
                    prefer_last=True,
                )
            self_uncertainty_by_method = _self_uncertainty_by_method(
                self.args,
                generated_text,
                generation_detail.get("uq_stats"),
            )
            if "NLL" in selected_methods and nll_target_stat is not None:
                self_uncertainty_by_method["NLL"] = nll_target_stat.get("nll")
            generated_text_for_eval = strip_structured_uncertainty_blocks(generated_text)

            if self.task in ["mbppplus", "humanevalplus"]:
                pred = extract_markdown_python_block(generated_text_for_eval)
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
                pred = normalize_answer(extract_gsm8k_answer(generated_text_for_eval))
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
                pred = normalize_answer(extract_mcq_choice(generated_text_for_eval))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            else:
                pred = normalize_answer(extract_gsm8k_answer(generated_text_for_eval))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            mask = attention_mask[idx].bool()
            trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
            agent_trace = {
                "name": "SingleAgent",
                "role": "singleagent",
                "input": prompts[idx],
                "input_ids": trimmed_ids,
                "input_tokens": tokens_batch[idx],
                "output": generated_text,
                "nll_target_text": _nll_target_text(generated_text),
                "self_uncertainty": self_uncertainty_by_method.get("ASK4CONF"),
                "self_uncertainty_by_method": self_uncertainty_by_method,
                "logits_uq_stats": generation_detail.get("uq_stats"),
                "logits_uq_stats_nll_target": nll_target_stat,
                "message_adoption": {},
            }
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": generated_text,
                    "agents": [agent_trace],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
