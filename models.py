import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # Decoder-only models should use left padding for batched generation.
    tokenizer.padding_side = "left"


def _load_tokenizer(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except ValueError as exc:
        msg = str(exc)
        if "Tokenizer class" not in msg:
            raise
        print(
            f"[Tokenizer] Fast tokenizer load failed for {model_name} ({exc}); "
            "retrying with use_fast=False."
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    _ensure_pad_token(tokenizer)
    return tokenizer


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


LOGIT_UQ_METHODS = ["MSP"]


def _selected_uq_methods(args) -> List[str]:
    value = (
        getattr(args, "local_uncertainty_modes", getattr(args, "local_uncertainty_mode", None))
        if args is not None
        else None
    )
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


def _uses_msp_uq(args) -> bool:
    return "MSP" in _selected_uq_methods(args)


def _apply_repetition_penalty_(scores: torch.Tensor, sequences: List[List[int]], penalty: float) -> None:
    if penalty is None or penalty == 1.0:
        return
    for row_idx, seq in enumerate(sequences):
        if not seq:
            continue
        unique_ids = set(int(token_id) for token_id in seq)
        token_ids = torch.tensor(sorted(unique_ids), device=scores.device, dtype=torch.long)
        token_scores = scores[row_idx, token_ids]
        adjusted = torch.where(token_scores < 0, token_scores * penalty, token_scores / penalty)
        scores[row_idx, token_ids] = adjusted


def _top_p_sample(filtered_logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        probs = torch.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_remove, float("-inf"))
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_sorted).squeeze(-1)


def _uq_stats_from_raw_logits(
    chosen_logprobs: List[float],
) -> Dict[str, Optional[float]]:
    if not chosen_logprobs:
        return {
            "sequence_probability": None,
            "msp": None,
            "generated_token_count": 0,
        }
    total_logprob = float(sum(chosen_logprobs))
    mean_logprob = total_logprob / len(chosen_logprobs)
    mean_token_probability = float(torch.exp(torch.tensor(mean_logprob, dtype=torch.float64)).item())
    msp = max(0.0, min(1.0, 1.0 - mean_token_probability))
    return {
        "sequence_probability": None,
        "mean_token_probability": mean_token_probability,
        "msp": msp,
        "generated_token_count": len(chosen_logprobs),
    }


def _uq_stats_from_cumulative_logprob(
    cumulative_logprob: Optional[float],
    generated_token_count: int,
) -> Dict[str, Optional[float]]:
    if cumulative_logprob is None or generated_token_count <= 0:
        return {
            "sequence_probability": None,
            "mean_token_probability": None,
            "msp": None,
            "generated_token_count": generated_token_count,
        }
    mean_logprob = float(cumulative_logprob) / generated_token_count
    mean_token_probability = float(torch.exp(torch.tensor(mean_logprob, dtype=torch.float64)).item())
    return {
        "sequence_probability": None,
        "mean_token_probability": mean_token_probability,
        "msp": max(0.0, min(1.0, 1.0 - mean_token_probability)),
        "generated_token_count": generated_token_count,
    }


def _extract_vllm_token_logprob(entry, token_id: int) -> Optional[float]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        value = entry.get(int(token_id))
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        logprob = getattr(value, "logprob", None)
        if logprob is not None:
            return float(logprob)
    if isinstance(entry, (int, float)):
        return float(entry)
    logprob = getattr(entry, "logprob", None)
    if logprob is not None:
        return float(logprob)
    return None


def _extract_vllm_sampled_token_logprobs(
    sample_logprobs,
    token_ids: List[int],
) -> List[Optional[float]]:
    if not sample_logprobs or not token_ids:
        return []
    chosen: List[Optional[float]] = []
    for token_id, entry in zip(token_ids, list(sample_logprobs)):
        chosen.append(_extract_vllm_token_logprob(entry, int(token_id)))
    return chosen


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args=None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.max_model_len = None
        self.args = args
        self.seed = int(getattr(args, "seed", 42)) if args is not None else 42

        if self.use_vllm:
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            self.vllm_engine = LLM(
                model=model_name,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_util,
                seed=self.seed,
            )
            self.tokenizer = _load_tokenizer(model_name)
            self.max_model_len = self._resolve_max_model_len()
            return

        self.tokenizer = _load_tokenizer(model_name)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        self.max_model_len = self._resolve_max_model_len()

    def _resolve_max_model_len(self) -> Optional[int]:
        if self.vllm_engine is not None:
            llm_engine = getattr(self.vllm_engine, "llm_engine", None)
            if llm_engine is not None:
                model_config = getattr(llm_engine, "model_config", None)
                value = getattr(model_config, "max_model_len", None)
                if isinstance(value, int) and value > 0:
                    return value
            engine_model_config = getattr(self.vllm_engine, "model_config", None)
            value = getattr(engine_model_config, "max_model_len", None)
            if isinstance(value, int) and value > 0:
                return value

        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**6:
            return tokenizer_limit

        model = getattr(self, "model", None)
        config = getattr(model, "config", None)
        value = getattr(config, "max_position_embeddings", None)
        if isinstance(value, int) and value > 0:
            return value
        return None

    def get_max_model_len(self) -> Optional[int]:
        return self.max_model_len

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        return_generation_details: bool = False,
    ):
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            logprobs=(1 if _uses_msp_uq(self.args) else None),
            seed=self.seed,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations: List[str] = []
        details: List[Dict] = []
        for out in outputs:
            first = out.outputs[0]
            raw_text = first.text
            token_ids = list(getattr(first, "token_ids", []) or [])
            if not token_ids and raw_text:
                token_ids = self.tokenizer(raw_text, add_special_tokens=False)["input_ids"]
            generations.append(raw_text.strip())
            detail = {
                "text_raw": raw_text,
                "generated_token_ids": token_ids,
                "token_logprobs": _extract_vllm_sampled_token_logprobs(
                    getattr(first, "logprobs", None),
                    token_ids,
                ),
                "uq_stats": _uq_stats_from_cumulative_logprob(
                    getattr(first, "cumulative_logprob", None),
                    len(token_ids),
                ) if _uses_msp_uq(self.args) else None,
            }
            details.append(detail)
        if return_generation_details:
            return generations, details
        return generations

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        past_key_values: Optional[Tuple] = None,
        return_generation_details: bool = False,
    ):
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if past_key_values is not None:
            raise NotImplementedError("past_key_values is not supported in manual generation mode.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        batch_size = input_ids.shape[0]
        eos_token_id = self.tokenizer.eos_token_id
        generated_token_ids: List[List[int]] = [[] for _ in range(batch_size)]
        chosen_logprobs: List[List[float]] = [[] for _ in range(batch_size)]
        sequence_history = [
            input_ids[row][attention_mask[row].bool()].tolist()
            for row in range(batch_size)
        ]
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        full_attention_mask = attention_mask
        past = None

        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                use_cache=True,
                past_key_values=past,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :].float()
            past = outputs.past_key_values
            log_probs = torch.log_softmax(logits, dim=-1)

            sampling_logits = logits.clone()
            _apply_repetition_penalty_(sampling_logits, sequence_history, repetition_penalty)
            if temperature is not None and temperature > 0:
                sampling_logits = sampling_logits / temperature
            next_tokens = _top_p_sample(sampling_logits, top_p=top_p)

            if eos_token_id is not None:
                next_tokens = torch.where(
                    unfinished,
                    next_tokens,
                    torch.full_like(next_tokens, eos_token_id),
                )

            for row_idx in range(batch_size):
                if not unfinished[row_idx]:
                    continue
                token_id = int(next_tokens[row_idx].item())
                generated_token_ids[row_idx].append(token_id)
                sequence_history[row_idx].append(token_id)
                chosen_logprobs[row_idx].append(float(log_probs[row_idx, token_id].item()))

            if eos_token_id is not None:
                unfinished = unfinished & (next_tokens != eos_token_id)
                if not torch.any(unfinished):
                    break

            full_attention_mask = torch.cat(
                [
                    full_attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=self.device),
                ],
                dim=-1,
            )
            current_input_ids = next_tokens.unsqueeze(-1)
            current_attention_mask = full_attention_mask

        generations: List[str] = []
        details: List[Dict] = []
        for row_idx in range(batch_size):
            token_ids = generated_token_ids[row_idx]
            raw_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            generations.append(raw_text.strip())
            details.append(
                {
                    "text_raw": raw_text,
                    "generated_token_ids": token_ids,
                    "token_logprobs": list(chosen_logprobs[row_idx]),
                    "uq_stats": _uq_stats_from_raw_logits(chosen_logprobs[row_idx]),
                }
            )
        if return_generation_details:
            return generations, None, details
        return generations, None

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)
