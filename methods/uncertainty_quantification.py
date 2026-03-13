import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

ANCHOR_TO_SCORE = {
    "VL": 0.0,
    "L": 0.25,
    "M": 0.5,
    "H": 0.75,
    "VH": 1.0,
}


def parse_uncertainty_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().upper()
        if not stripped:
            return None
        if stripped in ANCHOR_TO_SCORE:
            return ANCHOR_TO_SCORE[stripped]
        return None
    return None


def extract_agent_uncertainty(text: str):
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
        r'<agent_uncertainty\b[^>]*\blevel="(VL|L|M|H|VH)"[^>]*/?>',
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return ANCHOR_TO_SCORE.get(match.group(1).upper())

    match = re.search(
        r"(?im)^\s*(?:agent'?s?\s+uncertainty|self[_\s-]*uncertainty)\s*[:=]\s*([01](?:\.\d+)?|0?\.\d+)\s*$",
        text,
    )
    if match:
        return float(match.group(1))

    match = re.search(
        r"(?im)^\s*(?:agent'?s?\s+uncertainty|self[_\s-]*uncertainty)\s*[:=]\s*(VL|L|M|H|VH)\s*$",
        text,
    )
    if match:
        return ANCHOR_TO_SCORE.get(match.group(1).upper())

    match = re.search(
        r'<agent_confidence\s+score="([01](?:\.\d+)?|0?\.\d+)"\s*(?:/>|>)',
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return 1.0 - float(match.group(1))

    return None


def _clamp01(x):
    if x is None:
        return None
    return max(0.0, min(1.0, float(x)))


def _normalize_agent_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _agent_aliases(name: str, role: str, prompt: str) -> List[str]:
    aliases = {name, role}
    if prompt == "sequential":
        aliases.add(f"{name} Agent")
    elif prompt == "hierarchical":
        hierarchical_alias = {
            "planner": "Math Agent",
            "critic": "Science Agent",
            "refiner": "Code Agent",
            "judger": "Task Summarizer",
        }.get(str(role).lower(), "")
        if hierarchical_alias:
            aliases.add(hierarchical_alias)
    return [alias for alias in aliases if alias]


def _incoming_agent_names(role: str, prompt: str, hazard_step_idx: int) -> List[str]:
    role = str(role).lower()
    if prompt == "sequential":
        return {
            "planner": [],
            "critic": ["Planner Agent"],
            "refiner": ["Critic Agent"],
            "judger": ["Refiner Agent"],
        }.get(role, [])
    if prompt == "hierarchical":
        if hazard_step_idx == 0:
            return []
        return {
            "planner": [],
            "critic": [],
            "refiner": [],
            "judger": ["Math Agent", "Science Agent", "Code Agent"],
        }.get(role, [])
    return []


def _hazard_time_step_index(role: str, prompt: str) -> int:
    if prompt == "hierarchical":
        return 1 if role == "judger" else 0
    role_order = {"planner": 0, "critic": 1, "refiner": 2, "judger": 3}
    return role_order.get(role, 0)


def _hazard_num_time_steps(prompt: str, agents: List[Dict]) -> int:
    if prompt == "hierarchical":
        return 2
    return len(agents)


def _compute_formula_uncertainty(
    *,
    self_uncertainty: Optional[float],
    message_adoption: Dict[str, float],
    previous_formula_uncertainties: Dict[str, float],
    incoming_agent_names: List[str],
) -> Optional[float]:
    base_uncertainty = _clamp01(self_uncertainty)
    factors = [1.0 - base_uncertainty] if base_uncertainty is not None else []
    has_signal = base_uncertainty is not None

    normalized_adoption = {
        _normalize_agent_label(agent_name): _clamp01(parse_uncertainty_value(score))
        for agent_name, score in message_adoption.items()
        if _clamp01(parse_uncertainty_value(score)) is not None
    }

    for incoming_name in incoming_agent_names:
        alpha = normalized_adoption.get(_normalize_agent_label(incoming_name))
        source_pi = previous_formula_uncertainties.get(_normalize_agent_label(incoming_name))
        if alpha is None or source_pi is None:
            continue
        has_signal = True
        factors.append(1.0 - alpha * source_pi)

    if not has_signal:
        return None

    prod = 1.0
    for factor in factors:
        prod *= factor
    return _clamp01(1.0 - prod)


def _compute_hazard_ht(step_formula_uncertainties: List[Optional[float]]) -> Optional[float]:
    pi_vals = [
        _clamp01(value)
        for value in step_formula_uncertainties
        if _clamp01(value) is not None
    ]
    if not pi_vals:
        return None
    mean_pi = sum(pi_vals) / len(pi_vals)
    max_pi = max(pi_vals)
    return _clamp01((mean_pi + max_pi) / 2.0)


def _aggregate_temporal_uncertainty(step_scores: List[Optional[float]]) -> Optional[float]:
    valid_scores = [
        _clamp01(score)
        for score in step_scores
        if _clamp01(score) is not None
    ]
    if not valid_scores:
        return None

    prod = 1.0
    for score in valid_scores:
        prod *= (1.0 - score)
    return _clamp01(1.0 - (prod ** (1.0 / len(valid_scores))))


def enrich_mas_pred_with_posthoc_uncertainty(pred: Dict, prompt: str) -> Dict:
    agents = pred.get("agents", [])
    if not agents:
        return pred

    formula_uncertainty_states: Dict[str, float] = {}
    formula_values: List[Optional[float]] = []
    hazard_num_steps = _hazard_num_time_steps(prompt, agents)
    hazard_step_formula_uncertainties: List[List[Optional[float]]] = [[] for _ in range(hazard_num_steps)]
    agent_hazard_step_indices: List[int] = []

    for agent in agents:
        role = str(agent.get("role", "")).lower()
        name = agent.get("name", "Agent")
        self_uncertainty = parse_uncertainty_value(agent.get("self_uncertainty"))
        if self_uncertainty is None:
            self_uncertainty = extract_agent_uncertainty(agent.get("output", ""))
        message_adoption = agent.get("message_adoption", {}) or {}
        hazard_step_idx = _hazard_time_step_index(role, prompt)
        formula_uncertainty = _compute_formula_uncertainty(
            self_uncertainty=self_uncertainty,
            message_adoption=message_adoption,
            previous_formula_uncertainties=formula_uncertainty_states,
            incoming_agent_names=_incoming_agent_names(role, prompt, hazard_step_idx),
        )
        agent["self_uncertainty_num"] = _clamp01(self_uncertainty)
        agent["formula_uncertainty"] = formula_uncertainty
        formula_values.append(formula_uncertainty)
        hazard_step_formula_uncertainties[hazard_step_idx].append(formula_uncertainty)
        agent_hazard_step_indices.append(hazard_step_idx)
        for alias in _agent_aliases(name, role, prompt):
            if formula_uncertainty is not None:
                formula_uncertainty_states[_normalize_agent_label(alias)] = formula_uncertainty

    hazard_values = [
        _compute_hazard_ht(step_formula_uncertainties)
        for step_formula_uncertainties in hazard_step_formula_uncertainties
    ]
    for agent, step_idx in zip(agents, agent_hazard_step_indices):
        agent["hazard_ht"] = hazard_values[step_idx]

    pred["uncertainty_agent_list"] = [
        _clamp01(parse_uncertainty_value(agent.get("self_uncertainty")))
        for agent in agents
        if _clamp01(parse_uncertainty_value(agent.get("self_uncertainty"))) is not None
    ]
    pred["uncertainty_final"] = pred["uncertainty_agent_list"][-1] if pred["uncertainty_agent_list"] else None
    pred["formula_uncertainty_agent_list"] = [v for v in formula_values if v is not None]
    pred["formula_uncertainty_final"] = pred["formula_uncertainty_agent_list"][-1] if pred["formula_uncertainty_agent_list"] else None
    pred["hazard_ht_list"] = hazard_values
    pred["hazard_ht_final"] = hazard_values[-1] if hazard_values else None
    pred["system_step_uncertainty_list"] = hazard_values
    pred["system_uncertainty_final"] = _aggregate_temporal_uncertainty(hazard_values)
    return pred


def apply_posthoc_uncertainty(preds: List[Dict], method: str, prompt: str):
    if method != "mas":
        return preds
    for pred in preds:
        enrich_mas_pred_with_posthoc_uncertainty(pred, prompt)
    return preds


def _get_uncertainty_field_names(source: str = "self"):
    if source == "formula":
        return "formula_uncertainty_final", "formula_uncertainty_agent_list", "formula_uncertainty"
    if source == "hazard":
        return "hazard_ht_final", "hazard_ht_list", "hazard_ht"
    if source == "system":
        return "system_uncertainty_final", "system_step_uncertainty_list", None
    return "uncertainty_final", "uncertainty_agent_list", "self_uncertainty"


def _final_uncertainty_from_pred(pred: Dict, source: str = "self"):
    final_key, _, agent_key = _get_uncertainty_field_names(source)
    if pred.get(final_key) is not None:
        return _clamp01(parse_uncertainty_value(pred[final_key]))
    if source == "self" and pred.get("uncertainty") is not None:
        return _clamp01(parse_uncertainty_value(pred["uncertainty"]))
    agents = pred.get("agents", [])
    if agents:
        if agent_key is not None and agents[-1].get(agent_key) is not None:
            return _clamp01(parse_uncertainty_value(agents[-1].get(agent_key)))
        if source == "self":
            u = extract_agent_uncertainty(agents[-1].get("output", ""))
            if u is not None:
                return _clamp01(u)
    if source == "self":
        u = extract_agent_uncertainty(pred.get("raw_prediction", ""))
        if u is not None:
            return _clamp01(u)
    return None


def _all_agent_uncertainties_from_pred(pred: Dict, source: str = "self"):
    _, list_key, agent_key = _get_uncertainty_field_names(source)
    if isinstance(pred.get(list_key), list):
        vals = []
        for u in pred[list_key]:
            if u is None:
                continue
            parsed = _clamp01(parse_uncertainty_value(u))
            if parsed is not None:
                vals.append(parsed)
        return vals
    if agent_key is None:
        return []
    if source == "self" and pred.get("uncertainty") is not None:
        parsed = _clamp01(parse_uncertainty_value(pred["uncertainty"]))
        return [parsed] if parsed is not None else []
    vals = []
    for agent in pred.get("agents", []):
        if agent.get(agent_key) is not None:
            parsed = _clamp01(parse_uncertainty_value(agent.get(agent_key)))
            if parsed is not None:
                vals.append(parsed)
            continue
        if source == "self":
            u = extract_agent_uncertainty(agent.get("output", ""))
            if u is not None:
                vals.append(_clamp01(u))
    return vals


def _summarize_uncertainties(values: List[float]):
    if not values:
        return {"final": None, "mean": None, "max": None}
    return {
        "final": values[-1],
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _auroc(labels: List[int], scores: List[float]):
    n = len(labels)
    if n == 0 or n != len(scores):
        return None
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    rank = 1
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum_pos += avg_rank
        rank += (j - i)
        i = j

    u_stat = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    return u_stat / (n_pos * n_neg)


def _ece(labels: List[int], probs: List[float], n_bins: int = 10):
    n = len(labels)
    if n == 0 or n != len(probs):
        return None
    total = float(n)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idxs = []
        for i, p in enumerate(probs):
            if b < n_bins - 1:
                if lo <= p < hi:
                    idxs.append(i)
            else:
                if lo <= p <= hi:
                    idxs.append(i)
        if not idxs:
            continue
        acc = sum(labels[i] for i in idxs) / len(idxs)
        conf = sum(probs[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / total) * abs(acc - conf)
    return ece


def _brier(labels: List[int], probs: List[float]):
    n = len(labels)
    if n == 0 or n != len(probs):
        return None
    return sum((probs[i] - labels[i]) ** 2 for i in range(n)) / n


def evaluate_uncertainty_metrics(preds: List[Dict], mode: str = "final", source: str = "self"):
    labels: List[int] = []
    probs_correct: List[float] = []
    ranking_scores: List[float] = []
    for pred in preds:
        if mode == "final":
            u = _final_uncertainty_from_pred(pred, source=source)
        else:
            all_u = _all_agent_uncertainties_from_pred(pred, source=source)
            if not all_u:
                u = None
            elif mode == "mean":
                u = sum(all_u) / len(all_u)
            elif mode == "max":
                u = max(all_u)
            else:
                raise ValueError(f"Unknown uncertainty mode: {mode}")
        if u is None:
            continue
        labels.append(1 if pred.get("correct", False) else 0)
        score = 1.0 - u
        probs_correct.append(score)
        ranking_scores.append(score)

    return {
        "n_with_uncertainty": len(labels),
        "auroc": _auroc(labels, ranking_scores),
        "ece": _ece(labels, probs_correct),
        "brier": _brier(labels, probs_correct),
    }


def round4_or_none(x):
    return None if x is None else round(x, 4)


def print_posthoc_sample_reports(preds: List[Dict], args):
    hierarchical_name_map = {
        "planner": "Math Agent",
        "critic": "Science Agent",
        "refiner": "Code Agent",
        "judger": "Task Summarizer",
    }

    def _fmt_unc(x):
        parsed = _clamp01(parse_uncertainty_value(x))
        return "N/A" if parsed is None else f"{parsed:.4f}"

    print("\n==================== Posthoc Uncertainty Replay ====================")
    for idx, pred in enumerate(preds, start=1):
        agents = pred.get("agents", [])
        self_values = _all_agent_uncertainties_from_pred(pred, source="self")
        formula_values = _all_agent_uncertainties_from_pred(pred, source="formula")
        hazard_values = _all_agent_uncertainties_from_pred(pred, source="hazard")
        system_step_values = _all_agent_uncertainties_from_pred(pred, source="system")
        self_summary = _summarize_uncertainties(self_values)
        formula_summary = _summarize_uncertainties(formula_values)
        hazard_summary = _summarize_uncertainties(hazard_values)
        system_final = _final_uncertainty_from_pred(pred, source="system")

        print(f"Problem #{idx} Posthoc:")
        if agents:
            print("Agent Uncertainties:")
            for agent in agents:
                name = agent.get("name", "Agent")
                role = agent.get("role", "")
                display_name = name
                if args.method == "mas" and args.prompt == "hierarchical":
                    display_name = hierarchical_name_map.get(str(role).lower(), name)
                print(
                    f"  {display_name}: "
                    f"Self={_fmt_unc(agent.get('self_uncertainty'))} | "
                    f"Formula={_fmt_unc(agent.get('formula_uncertainty'))} | "
                    f"h_t={_fmt_unc(agent.get('hazard_ht'))}"
                )
        if hazard_values:
            print(
                "Hazard-h Steps: "
                + " | ".join(f"t{i + 1}={_fmt_unc(v)}" for i, v in enumerate(hazard_values))
            )
        if system_step_values:
            print(
                "System-Step-Uncertainty: "
                + " | ".join(f"t{i + 1}={_fmt_unc(v)}" for i, v in enumerate(system_step_values))
            )
        print(
            f"Uncertainty: Final={_fmt_unc(self_summary['final'])} | "
            f"Mean={_fmt_unc(self_summary['mean'])} | Max={_fmt_unc(self_summary['max'])}"
        )
        print(
            "Formula-Uncertainty: "
            f"Final={_fmt_unc(formula_summary['final'])} | "
            f"Mean={_fmt_unc(formula_summary['mean'])} | "
            f"Max={_fmt_unc(formula_summary['max'])}"
        )
        print(
            "Hazard-h: "
            f"Final={_fmt_unc(hazard_summary['final'])} | "
            f"Mean={_fmt_unc(hazard_summary['mean'])} | "
            f"Max={_fmt_unc(hazard_summary['max'])}"
        )
        print(f"System-Uncertainty-Final: {_fmt_unc(system_final)}")


def export_preds_jsonl(preds: List[Dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pred in preds:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


def append_preds_jsonl(preds: List[Dict], output_path: Path):
    if not preds:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for pred in preds:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


def load_preds_jsonl(input_path: Path) -> List[Dict]:
    preds: List[Dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    return preds


def build_uncertainty_metrics_summary(preds: List[Dict]) -> Dict:
    unc_final = evaluate_uncertainty_metrics(preds, mode="final", source="self")
    unc_mean = evaluate_uncertainty_metrics(preds, mode="mean", source="self")
    unc_max = evaluate_uncertainty_metrics(preds, mode="max", source="self")
    formula_unc_final = evaluate_uncertainty_metrics(preds, mode="final", source="formula")
    formula_unc_mean = evaluate_uncertainty_metrics(preds, mode="mean", source="formula")
    formula_unc_max = evaluate_uncertainty_metrics(preds, mode="max", source="formula")
    hazard_ht_final = evaluate_uncertainty_metrics(preds, mode="final", source="hazard")
    hazard_ht_mean = evaluate_uncertainty_metrics(preds, mode="mean", source="hazard")
    hazard_ht_max = evaluate_uncertainty_metrics(preds, mode="max", source="hazard")
    system_unc_final = evaluate_uncertainty_metrics(preds, mode="final", source="system")
    return {
        "Uncertainty-Final": unc_final,
        "Uncertainty-Mean": unc_mean,
        "Uncertainty-Max": unc_max,
        "Formula-Uncertainty-Final": formula_unc_final,
        "Formula-Uncertainty-Mean": formula_unc_mean,
        "Formula-Uncertainty-Max": formula_unc_max,
        "Hazard-h-Final": hazard_ht_final,
        "Hazard-h-Mean": hazard_ht_mean,
        "Hazard-h-Max": hazard_ht_max,
        "System-Uncertainty-Final": system_unc_final,
    }


def print_uncertainty_metrics_summary(metrics: Dict):
    for label, values in metrics.items():
        print(
            f"{label}: "
            f"AUROC={round4_or_none(values['auroc'])} | "
            f"ECE={round4_or_none(values['ece'])} | "
            f"Brier={round4_or_none(values['brier'])}"
        )


def export_metrics_json(metrics: Dict, output_path: Path, metadata: Optional[Dict] = None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics}
    if metadata:
        payload["metadata"] = metadata
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
