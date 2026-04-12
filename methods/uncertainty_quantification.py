import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

def parse_uncertainty_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
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


def _clamp01(x):
    if x is None:
        return None
    return max(0.0, min(1.0, float(x)))


def _normalize_agent_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _is_divergent_star(pred: Dict) -> bool:
    graph = pred.get("mas_graph") or {}
    return graph.get("topology") == "star_divergent"


def _agent_aliases(name: str, role: str) -> List[str]:
    aliases = {name, role}
    agent_num_match = re.search(r"(\d+)", str(name))
    if agent_num_match:
        aliases.add(f"Agent {agent_num_match.group(1)}")
    return [alias for alias in aliases if alias]


def _graph_incoming_agent_names(agent: Dict, pred: Dict) -> List[str] | None:
    if isinstance(agent.get("incoming_agents"), list):
        return [str(name) for name in agent.get("incoming_agents", []) if str(name).strip()]
    graph = pred.get("mas_graph") or {}
    incoming = graph.get("incoming", {})
    name = str(agent.get("name", ""))
    if isinstance(incoming, dict) and name in incoming:
        return [str(label) for label in incoming.get(name, []) if str(label).strip()]
    return None


def _graph_hazard_time_step_index(agent: Dict, pred: Dict, fallback_idx: int) -> int:
    if isinstance(pred.get("mas_graph"), dict):
        graph = pred["mas_graph"]
        levels = graph.get("levels", {})
        name = str(agent.get("name", ""))
        level = levels.get(name)
        if isinstance(level, int):
            return level
    return fallback_idx


def _graph_hazard_num_time_steps(pred: Dict, agents: List[Dict]) -> int:
    graph = pred.get("mas_graph") or {}
    levels = graph.get("levels", {})
    if isinstance(levels, dict) and levels:
        max_level = max(int(level) for level in levels.values())
        return max_level + 1
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


def _normalized_structural_uncertainty(value: Optional[float], uq_method: str) -> Optional[float]:
    parsed = parse_uncertainty_value(value)
    if parsed is None:
        return None
    if uq_method in {"ASK4CONF", "MSP"}:
        return _clamp01(parsed)
    if uq_method == "NLL":
        return _clamp01(1.0 - math.exp(-parsed))
    return _clamp01(parsed)


def _pred_available_uq_methods(pred: Dict) -> List[str]:
    methods: List[str] = []
    for agent in pred.get("agents", []):
        by_method = agent.get("self_uncertainty_by_method")
        if isinstance(by_method, dict):
            for method_name in by_method:
                if method_name not in methods:
                    methods.append(method_name)
    if not methods:
        methods.append("ASK4CONF")
    return methods


def _agent_self_uncertainty_for_method(agent: Dict, uq_method: str) -> Optional[float]:
    by_method = agent.get("self_uncertainty_by_method")
    if isinstance(by_method, dict) and uq_method in by_method:
        return parse_uncertainty_value(by_method.get(uq_method))
    if uq_method == "ASK4CONF":
        value = parse_uncertainty_value(agent.get("self_uncertainty"))
        if value is not None:
            return value
        return extract_agent_uncertainty(agent.get("output", ""))
    return None


def _agent_message_adoption_for_method(agent: Dict, uq_method: str) -> Dict[str, float]:
    by_method = agent.get("message_adoption_by_method")
    if isinstance(by_method, dict):
        method_value = by_method.get(uq_method)
        if isinstance(method_value, dict):
            return method_value
    if uq_method == "ASK4CONF":
        value = agent.get("message_adoption", {})
        return value if isinstance(value, dict) else {}
    return {}


def _resolved_message_adoption_for_method(
    agent: Dict,
    uq_method: str,
    incoming_agent_names: List[str],
    adoption_mode: str,
) -> Dict[str, float]:
    if adoption_mode == "all_one":
        return {str(name): 1.0 for name in incoming_agent_names if str(name).strip()}
    return _agent_message_adoption_for_method(agent, uq_method)


def _agent_method_field(agent: Dict, field_name: str, uq_method: str):
    by_method = agent.get(f"{field_name}_by_method")
    if isinstance(by_method, dict) and uq_method in by_method:
        return by_method.get(uq_method)
    if uq_method == "ASK4CONF":
        return agent.get(field_name)
    return None


def _pred_method_field(pred: Dict, field_name: str, uq_method: str):
    by_method = pred.get(f"{field_name}_by_method")
    if isinstance(by_method, dict) and uq_method in by_method:
        return by_method.get(uq_method)
    if uq_method == "ASK4CONF":
        return pred.get(field_name)
    return None


def enrich_mas_pred_with_posthoc_uncertainty(pred: Dict, adoption_mode: str = "original") -> Dict:
    agents = pred.get("agents", [])
    if not agents:
        return pred

    uq_methods = _pred_available_uq_methods(pred)
    pred["uncertainty_agent_list_by_method"] = {}
    pred["uncertainty_final_by_method"] = {}
    pred["formula_uncertainty_agent_list_by_method"] = {}
    pred["formula_uncertainty_final_by_method"] = {}
    pred["hazard_ht_list_by_method"] = {}
    pred["hazard_ht_final_by_method"] = {}
    pred["system_step_uncertainty_list_by_method"] = {}
    pred["system_uncertainty_final_by_method"] = {}

    for uq_method in uq_methods:
        formula_uncertainty_states: Dict[str, float] = {}
        formula_values: List[Optional[float]] = []
        hazard_num_steps = _graph_hazard_num_time_steps(pred, agents)
        hazard_step_formula_uncertainties: List[List[Optional[float]]] = [[] for _ in range(hazard_num_steps)]
        agent_hazard_step_indices: List[int] = []
        self_values: List[float] = []

        for fallback_idx, agent in enumerate(agents):
            role = str(agent.get("role", "")).lower()
            name = agent.get("name", "Agent")
            self_uncertainty = _agent_self_uncertainty_for_method(agent, uq_method)
            structural_self_uncertainty = _normalized_structural_uncertainty(self_uncertainty, uq_method)
            hazard_step_idx = _graph_hazard_time_step_index(agent, pred, fallback_idx)
            incoming_agent_names = _graph_incoming_agent_names(agent, pred) or []
            message_adoption = _resolved_message_adoption_for_method(
                agent,
                uq_method,
                incoming_agent_names,
                adoption_mode,
            )
            formula_uncertainty = _compute_formula_uncertainty(
                self_uncertainty=structural_self_uncertainty,
                message_adoption=message_adoption,
                previous_formula_uncertainties=formula_uncertainty_states,
                incoming_agent_names=incoming_agent_names,
            )

            agent.setdefault("self_uncertainty_num_by_method", {})[uq_method] = parse_uncertainty_value(self_uncertainty)
            agent.setdefault("formula_uncertainty_by_method", {})[uq_method] = formula_uncertainty
            formula_values.append(formula_uncertainty)
            hazard_step_formula_uncertainties[hazard_step_idx].append(formula_uncertainty)
            agent_hazard_step_indices.append(hazard_step_idx)
            parsed_self = parse_uncertainty_value(self_uncertainty)
            if parsed_self is not None:
                self_values.append(parsed_self)
            for alias in _agent_aliases(name, role):
                if formula_uncertainty is not None:
                    formula_uncertainty_states[_normalize_agent_label(alias)] = formula_uncertainty

        hazard_values = [
            _compute_hazard_ht(step_formula_uncertainties)
            for step_formula_uncertainties in hazard_step_formula_uncertainties
        ]
        for agent, step_idx in zip(agents, agent_hazard_step_indices):
            agent.setdefault("hazard_ht_by_method", {})[uq_method] = hazard_values[step_idx]

        pred["uncertainty_agent_list_by_method"][uq_method] = self_values
        pred["uncertainty_final_by_method"][uq_method] = self_values[-1] if self_values else None
        pred["formula_uncertainty_agent_list_by_method"][uq_method] = [v for v in formula_values if v is not None]
        pred["formula_uncertainty_final_by_method"][uq_method] = (
            pred["formula_uncertainty_agent_list_by_method"][uq_method][-1]
            if pred["formula_uncertainty_agent_list_by_method"][uq_method]
            else None
        )
        pred["hazard_ht_list_by_method"][uq_method] = hazard_values
        pred["hazard_ht_final_by_method"][uq_method] = hazard_values[-1] if hazard_values else None
        pred["system_step_uncertainty_list_by_method"][uq_method] = hazard_values
        pred["system_uncertainty_final_by_method"][uq_method] = _aggregate_temporal_uncertainty(hazard_values)
        if _is_divergent_star(pred):
            pred["uncertainty_final_by_method"][uq_method] = None
            pred["formula_uncertainty_final_by_method"][uq_method] = None

    if "ASK4CONF" in uq_methods:
        pred["uncertainty_agent_list"] = pred["uncertainty_agent_list_by_method"].get("ASK4CONF", [])
        pred["uncertainty_final"] = pred["uncertainty_final_by_method"].get("ASK4CONF")
        pred["formula_uncertainty_agent_list"] = pred["formula_uncertainty_agent_list_by_method"].get("ASK4CONF", [])
        pred["formula_uncertainty_final"] = pred["formula_uncertainty_final_by_method"].get("ASK4CONF")
        pred["hazard_ht_list"] = pred["hazard_ht_list_by_method"].get("ASK4CONF", [])
        pred["hazard_ht_final"] = pred["hazard_ht_final_by_method"].get("ASK4CONF")
        pred["system_step_uncertainty_list"] = pred["system_step_uncertainty_list_by_method"].get("ASK4CONF", [])
        pred["system_uncertainty_final"] = pred["system_uncertainty_final_by_method"].get("ASK4CONF")
        for agent in agents:
            if isinstance(agent.get("self_uncertainty_num_by_method"), dict):
                agent["self_uncertainty_num"] = agent["self_uncertainty_num_by_method"].get("ASK4CONF")
            if isinstance(agent.get("formula_uncertainty_by_method"), dict):
                agent["formula_uncertainty"] = agent["formula_uncertainty_by_method"].get("ASK4CONF")
            if isinstance(agent.get("hazard_ht_by_method"), dict):
                agent["hazard_ht"] = agent["hazard_ht_by_method"].get("ASK4CONF")
    return pred


def apply_posthoc_uncertainty(preds: List[Dict], method: str, adoption_mode: str = "original"):
    if method != "mas":
        return preds
    for pred in preds:
        enrich_mas_pred_with_posthoc_uncertainty(pred, adoption_mode=adoption_mode)
    return preds


def _final_uncertainty_from_pred(pred: Dict, uq_method: str = "ASK4CONF", source: str = "self"):
    if _is_divergent_star(pred) and source not in {"hazard", "system"}:
        return None
    if source == "formula":
        value = _pred_method_field(pred, "formula_uncertainty_final", uq_method)
        return parse_uncertainty_value(value)
    if source == "hazard":
        value = _pred_method_field(pred, "hazard_ht_final", uq_method)
        return parse_uncertainty_value(value)
    if source == "system":
        value = _pred_method_field(pred, "system_uncertainty_final", uq_method)
        return parse_uncertainty_value(value)

    value = _pred_method_field(pred, "uncertainty_final", uq_method)
    if value is not None:
        return parse_uncertainty_value(value)
    if uq_method == "ASK4CONF" and pred.get("uncertainty") is not None:
        return parse_uncertainty_value(pred["uncertainty"])
    agents = pred.get("agents", [])
    if agents:
        value = _agent_self_uncertainty_for_method(agents[-1], uq_method)
        if value is not None:
            return value
    if uq_method == "ASK4CONF":
        return extract_agent_uncertainty(pred.get("raw_prediction", ""))
    return None


def _all_agent_uncertainties_from_pred(pred: Dict, uq_method: str = "ASK4CONF", source: str = "self"):
    if source == "formula":
        value = _pred_method_field(pred, "formula_uncertainty_agent_list", uq_method)
    elif source == "hazard":
        value = _pred_method_field(pred, "hazard_ht_list", uq_method)
    elif source == "system":
        value = _pred_method_field(pred, "system_step_uncertainty_list", uq_method)
    else:
        value = _pred_method_field(pred, "uncertainty_agent_list", uq_method)
    if isinstance(value, list):
        vals = []
        for item in value:
            parsed = parse_uncertainty_value(item)
            if parsed is not None:
                vals.append(parsed)
        return vals

    vals = []
    if source == "self" and uq_method == "ASK4CONF" and pred.get("uncertainty") is not None:
        parsed = parse_uncertainty_value(pred["uncertainty"])
        return [parsed] if parsed is not None else []
    if source == "system":
        return vals
    agent_field = "formula_uncertainty" if source == "formula" else "hazard_ht" if source == "hazard" else "self_uncertainty"
    for agent in pred.get("agents", []):
        if source == "self":
            parsed = _agent_self_uncertainty_for_method(agent, uq_method)
        else:
            parsed = parse_uncertainty_value(_agent_method_field(agent, agent_field, uq_method))
        if parsed is not None:
            vals.append(parsed)
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


def _uncertainty_to_confidence(
    u: Optional[float],
    uq_method: str,
    source: str = "self",
) -> Optional[float]:
    value = parse_uncertainty_value(u)
    if value is None:
        return None
    if uq_method == "ASK4CONF":
        return _clamp01(1.0 - value)
    if uq_method == "MSP":
        return _clamp01(1.0 - value)
    if uq_method == "NLL":
        if source == "self":
            return _clamp01(math.exp(-value))
        return _clamp01(1.0 - value)
    return None


def evaluate_uncertainty_metrics(preds: List[Dict], mode: str = "final", source: str = "self", uq_method: str = "ASK4CONF"):
    labels: List[int] = []
    probs_correct: List[float] = []
    ranking_scores: List[float] = []
    for pred in preds:
        if mode == "final":
            u = _final_uncertainty_from_pred(pred, uq_method=uq_method, source=source)
        else:
            all_u = _all_agent_uncertainties_from_pred(pred, uq_method=uq_method, source=source)
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
        confidence = _uncertainty_to_confidence(u, uq_method, source=source)
        if confidence is None:
            continue
        labels.append(1 if pred.get("correct", False) else 0)
        probs_correct.append(confidence)
        ranking_scores.append(confidence)

    return {
        "n_with_uncertainty": len(labels),
        "auroc": _auroc(labels, ranking_scores),
        "ece": _ece(labels, probs_correct),
        "brier": _brier(labels, probs_correct),
    }


def _pred_has_metric_uncertainty(pred: Dict, mode: str, source: str, uq_method: str) -> bool:
    if mode == "final":
        return _final_uncertainty_from_pred(pred, uq_method=uq_method, source=source) is not None

    all_u = _all_agent_uncertainties_from_pred(pred, uq_method=uq_method, source=source)
    return len(all_u) > 0


def _required_specs_for_complete_case(pred: Dict) -> List[tuple[str, str]]:
    if _is_divergent_star(pred):
        return [
            ("mean", "self"),
            ("max", "self"),
            ("mean", "formula"),
            ("max", "formula"),
            ("final", "hazard"),
            ("mean", "hazard"),
            ("max", "hazard"),
        ]

    return [
        ("final", "self"),
        ("mean", "self"),
        ("max", "self"),
        ("final", "formula"),
        ("mean", "formula"),
        ("max", "formula"),
        ("final", "hazard"),
        ("mean", "hazard"),
        ("max", "hazard"),
        ("final", "system"),
    ]


def _filter_complete_case_preds(preds: List[Dict], uq_method: str) -> List[Dict]:

    filtered: List[Dict] = []
    for pred in preds:
        required_specs = _required_specs_for_complete_case(pred)
        if all(_pred_has_metric_uncertainty(pred, mode, source, uq_method) for mode, source in required_specs):
            filtered.append(pred)
    return filtered


def round4_or_none(x):
    return None if x is None else round(x, 4)


def print_posthoc_sample_reports(preds: List[Dict], args):
    def _fmt_unc(x):
        parsed = parse_uncertainty_value(x)
        return "N/A" if parsed is None else f"{parsed:.4f}"

    print("\n==================== Posthoc Uncertainty Replay ====================")
    for idx, pred in enumerate(preds, start=1):
        agents = pred.get("agents", [])
        print(f"Problem #{idx} Posthoc:")
        for uq_method in _pred_available_uq_methods(pred):
            self_values = _all_agent_uncertainties_from_pred(pred, uq_method=uq_method, source="self")
            formula_values = _all_agent_uncertainties_from_pred(pred, uq_method=uq_method, source="formula")
            hazard_values = _all_agent_uncertainties_from_pred(pred, uq_method=uq_method, source="hazard")
            system_step_values = _all_agent_uncertainties_from_pred(pred, uq_method=uq_method, source="system")
            self_summary = _summarize_uncertainties(self_values)
            formula_summary = _summarize_uncertainties(formula_values)
            hazard_summary = _summarize_uncertainties(hazard_values)
            system_final = _final_uncertainty_from_pred(pred, uq_method=uq_method, source="system")

            print(f"  UQ Method: {uq_method}")
            if agents:
                print("  Agent Uncertainties:")
                for agent in agents:
                    print(
                        f"    {agent.get('name', 'Agent')}: "
                        f"Self={_fmt_unc(_agent_self_uncertainty_for_method(agent, uq_method))} | "
                        f"Formula={_fmt_unc(_agent_method_field(agent, 'formula_uncertainty', uq_method))} | "
                        f"h_t={_fmt_unc(_agent_method_field(agent, 'hazard_ht', uq_method))}"
                    )
            if hazard_values:
                print(
                    "  Hazard-h Steps: "
                    + " | ".join(f"t{i + 1}={_fmt_unc(v)}" for i, v in enumerate(hazard_values))
                )
            if system_step_values:
                print(
                    "  System-Step-Uncertainty: "
                    + " | ".join(f"t{i + 1}={_fmt_unc(v)}" for i, v in enumerate(system_step_values))
                )
            print(
                f"  Uncertainty: Final={_fmt_unc(self_summary['final'])} | "
                f"Mean={_fmt_unc(self_summary['mean'])} | Max={_fmt_unc(self_summary['max'])}"
            )
            print(
                "  Formula-Uncertainty: "
                f"Final={_fmt_unc(formula_summary['final'])} | "
                f"Mean={_fmt_unc(formula_summary['mean'])} | "
                f"Max={_fmt_unc(formula_summary['max'])}"
            )
            print(
                "  Hazard-h: "
                f"Final={_fmt_unc(hazard_summary['final'])} | "
                f"Mean={_fmt_unc(hazard_summary['mean'])} | "
                f"Max={_fmt_unc(hazard_summary['max'])}"
            )
            print(f"  System-Uncertainty-Final: {_fmt_unc(system_final)}")


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


def _build_metrics_summary_for_method(preds: List[Dict], uq_method: str) -> Dict:
    complete_case_preds = _filter_complete_case_preds(preds, uq_method)
    unc_final = evaluate_uncertainty_metrics(complete_case_preds, mode="final", source="self", uq_method=uq_method)
    unc_mean = evaluate_uncertainty_metrics(complete_case_preds, mode="mean", source="self", uq_method=uq_method)
    unc_max = evaluate_uncertainty_metrics(complete_case_preds, mode="max", source="self", uq_method=uq_method)
    formula_unc_final = evaluate_uncertainty_metrics(complete_case_preds, mode="final", source="formula", uq_method=uq_method)
    formula_unc_mean = evaluate_uncertainty_metrics(complete_case_preds, mode="mean", source="formula", uq_method=uq_method)
    formula_unc_max = evaluate_uncertainty_metrics(complete_case_preds, mode="max", source="formula", uq_method=uq_method)
    hazard_ht_final = evaluate_uncertainty_metrics(complete_case_preds, mode="final", source="hazard", uq_method=uq_method)
    hazard_ht_mean = evaluate_uncertainty_metrics(complete_case_preds, mode="mean", source="hazard", uq_method=uq_method)
    hazard_ht_max = evaluate_uncertainty_metrics(complete_case_preds, mode="max", source="hazard", uq_method=uq_method)
    system_unc_final = evaluate_uncertainty_metrics(complete_case_preds, mode="final", source="system", uq_method=uq_method)
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


def build_uncertainty_metrics_summary(preds: List[Dict]) -> Dict:
    methods: List[str] = []
    for pred in preds:
        for uq_method in _pred_available_uq_methods(pred):
            if uq_method not in methods:
                methods.append(uq_method)
    return {
        "metrics_by_uq_method": {
            uq_method: _build_metrics_summary_for_method(preds, uq_method)
            for uq_method in methods
        }
    }


def print_uncertainty_metrics_summary(metrics: Dict):
    metrics_by_method = metrics.get("metrics_by_uq_method", {})
    for uq_method, method_metrics in metrics_by_method.items():
        print(f"== {uq_method} ==")
        for label, values in method_metrics.items():
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
