import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

PRR_MAX_REJECTION_RATE = 0.5


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


def extract_local_uncertainty(text: str):
    if not text:
        return None

    match = re.search(
        r'<local_uncertainty\b[^>]*\bscore="([01](?:\.\d+)?|0?\.\d+)"[^>]*/?>',
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1))

    match = re.search(
        r"(?im)^\s*local[_\s-]*uncertainty\s*[:=]\s*([01](?:\.\d+)?|0?\.\d+)\s*$",
        text,
    )
    if match:
        return float(match.group(1))

    return None


def _clamp01(x):
    if x is None:
        return None
    return max(0.0, min(1.0, float(x)))


def _normalize_agent_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def extract_alpha_scores(text: str) -> Dict[str, float]:
    if not text:
        return {}

    parsed: Dict[str, float] = {}
    for match in re.finditer(r"<alpha\b([^>]*)/?>", text, flags=re.IGNORECASE):
        attrs = match.group(1)
        agent_match = re.search(r'\bagent="([^"]+)"', attrs, flags=re.IGNORECASE)
        if not agent_match:
            continue
        score_match = re.search(
            r'\bscore="([01](?:\.\d+)?|0?\.\d+)"',
            attrs,
            flags=re.IGNORECASE,
        )
        if not score_match:
            continue
        parsed[agent_match.group(1).strip()] = _clamp01(float(score_match.group(1)))
    return {
        agent_name: score
        for agent_name, score in parsed.items()
        if score is not None
    }


def _legacy_prompt_label_aliases(name: str, role: str, pred: Dict) -> List[str]:
    graph = pred.get("mas_graph") or {}
    topology = str(graph.get("topology", "") or "").strip().lower()
    role_name = str(role or "").strip().lower()
    canonical_name = str(name or "").strip()

    if topology == "sequential":
        label_map = {
            "planner": "Planner Agent",
            "critic": "Critic Agent",
            "refiner": "Refiner Agent",
            "judger": "Judger Agent",
        }
        mapped = label_map.get(role_name)
        return [mapped] if mapped else []

    if topology == "hierarchical":
        label_map = {
            "planner": "Math Agent",
            "critic": "Science Agent",
            "refiner": "Code Agent",
            "judger": "Task Summarizer",
        }
        mapped = label_map.get(role_name)
        return [mapped] if mapped else []

    if canonical_name.endswith("Agent"):
        return [canonical_name]
    return []


def _agent_aliases(name: str, role: str, pred: Dict) -> List[str]:
    aliases = {name, role}
    agent_num_match = re.search(r"(\d+)", str(name))
    if agent_num_match:
        aliases.add(f"Agent {agent_num_match.group(1)}")
    aliases.update(_legacy_prompt_label_aliases(name, role, pred))
    return [alias for alias in aliases if alias]


def _graph_incoming_agent_names(agent: Dict, pred: Dict) -> List[str] | None:
    if isinstance(agent.get("incoming_agents"), list):
        return [str(name) for name in agent.get("incoming_agents", []) if str(name).strip()]
    graph = pred.get("mas_graph") or {}
    incoming = graph.get("incoming", {})
    name = str(agent.get("name", ""))
    if isinstance(incoming, dict):
        return [str(label) for label in incoming.get(name, []) if str(label).strip()]
    return None


def _jsonl_alpha_lookup(alpha: Dict[str, float]) -> Dict[str, float]:
    return {
        _normalize_agent_label(agent_name): _clamp01(parse_uncertainty_value(score))
        for agent_name, score in alpha.items()
        if _clamp01(parse_uncertainty_value(score)) is not None
    }


def _alpha_lookup(
    alpha: Dict[str, float],
    incoming_agent_names: List[str],
) -> Dict[str, float]:
    raw_lookup = _jsonl_alpha_lookup(alpha)
    incoming_alpha: Dict[str, float] = {}
    total_alpha = 0.0
    for incoming_name in incoming_agent_names:
        normalized_name = _normalize_agent_label(incoming_name)
        alpha_value = raw_lookup.get(normalized_name)
        if alpha_value is None or alpha_value <= 0.0:
            continue
        incoming_alpha[normalized_name] = alpha_value
        total_alpha += alpha_value

    if total_alpha <= 0.0:
        return {}

    return {
        normalized_name: alpha_value / total_alpha
        for normalized_name, alpha_value in incoming_alpha.items()
    }


def _compute_prop_uncertainty(
    *,
    local_uncertainty: Optional[float],
    alpha: Dict[str, float],
    previous_prop_uncertainties: Dict[str, float],
    incoming_agent_names: List[str],
) -> Optional[float]:
    base_uncertainty = _clamp01(local_uncertainty)
    factors = [1.0 - base_uncertainty] if base_uncertainty is not None else []
    has_signal = base_uncertainty is not None
    alpha_lookup = _alpha_lookup(alpha, incoming_agent_names)

    for incoming_name in incoming_agent_names:
        alpha_value = alpha_lookup.get(_normalize_agent_label(incoming_name))
        source_pi = previous_prop_uncertainties.get(_normalize_agent_label(incoming_name))
        if alpha_value is None or source_pi is None:
            continue
        has_signal = True
        factors.append(1.0 - alpha_value * source_pi)

    if not has_signal:
        return None

    prod = 1.0
    for factor in factors:
        prod *= factor
    return _clamp01(1.0 - prod)


def _normalized_structural_uncertainty(value: Optional[float], uq_method: str) -> Optional[float]:
    parsed = parse_uncertainty_value(value)
    if parsed is None:
        return None
    if uq_method in {"Verb", "MSP"}:
        return _clamp01(parsed)
    return _clamp01(parsed)


def _pred_available_uq_methods(pred: Dict) -> List[str]:
    methods: List[str] = []
    for agent in pred.get("agents", []):
        by_method = agent.get("local_uncertainty_by_method")
        if isinstance(by_method, dict):
            for method_name in by_method:
                if method_name not in methods:
                    methods.append(method_name)
    if not methods:
        methods.append("Verb")
    return methods


def _agent_local_uncertainty_for_method(agent: Dict, uq_method: str) -> Optional[float]:
    by_method = agent.get("local_uncertainty_by_method")
    if isinstance(by_method, dict) and uq_method in by_method:
        return parse_uncertainty_value(by_method.get(uq_method))
    if uq_method == "Verb":
        value = parse_uncertainty_value(agent.get("local_uncertainty"))
        if value is not None:
            return value
        return extract_local_uncertainty(agent.get("output", ""))
    return None


def _agent_alpha_for_method(agent: Dict, uq_method: str) -> Dict[str, float]:
    by_method = agent.get("alpha_by_method")
    if isinstance(by_method, dict):
        method_value = by_method.get(uq_method)
        if isinstance(method_value, dict):
            return method_value
    value = agent.get("alpha", {})
    if isinstance(value, dict) and value:
        return value
    parsed_from_output = extract_alpha_scores(str(agent.get("output", "")))
    if parsed_from_output:
        return parsed_from_output
    if isinstance(value, dict):
        return value
    return {}


def _agent_method_field(agent: Dict, field_name: str, uq_method: str):
    by_method = agent.get(f"{field_name}_by_method")
    if isinstance(by_method, dict) and uq_method in by_method:
        return by_method.get(uq_method)
    if uq_method == "Verb":
        return agent.get(field_name)
    return None


def _pred_method_field(pred: Dict, field_name: str, uq_method: str):
    by_method = pred.get(f"{field_name}_by_method")
    if isinstance(by_method, dict) and uq_method in by_method:
        return by_method.get(uq_method)
    if uq_method == "Verb":
        return pred.get(field_name)
    return None


def enrich_mas_pred_with_posthoc_uncertainty(pred: Dict) -> Dict:
    agents = pred.get("agents", [])
    if not agents:
        return pred

    uq_methods = _pred_available_uq_methods(pred)
    pred["local_uncertainty_agent_list_by_method"] = {}
    pred["local_uncertainty_final_by_method"] = {}
    pred["prop_uncertainty_agent_list_by_method"] = {}
    pred["prop_uncertainty_final_by_method"] = {}

    for uq_method in uq_methods:
        prop_uncertainty_states: Dict[str, float] = {}
        prop_values: List[Optional[float]] = []
        local_values: List[float] = []

        for agent in agents:
            role = str(agent.get("role", "")).lower()
            name = agent.get("name", "Agent")
            local_uncertainty = _agent_local_uncertainty_for_method(agent, uq_method)
            structural_local_uncertainty = _normalized_structural_uncertainty(local_uncertainty, uq_method)
            incoming_agent_names = _graph_incoming_agent_names(agent, pred) or []
            alpha = _agent_alpha_for_method(agent, uq_method)
            prop_uncertainty = _compute_prop_uncertainty(
                local_uncertainty=structural_local_uncertainty,
                alpha=alpha,
                previous_prop_uncertainties=prop_uncertainty_states,
                incoming_agent_names=incoming_agent_names,
            )

            agent.setdefault("local_uncertainty_num_by_method", {})[uq_method] = parse_uncertainty_value(local_uncertainty)
            agent.setdefault("prop_uncertainty_by_method", {})[uq_method] = prop_uncertainty
            prop_values.append(prop_uncertainty)
            parsed_local = parse_uncertainty_value(local_uncertainty)
            if parsed_local is not None:
                local_values.append(parsed_local)
            for alias in _agent_aliases(name, role, pred):
                if prop_uncertainty is not None:
                    prop_uncertainty_states[_normalize_agent_label(alias)] = prop_uncertainty

        pred["local_uncertainty_agent_list_by_method"][uq_method] = local_values
        pred["local_uncertainty_final_by_method"][uq_method] = local_values[-1] if local_values else None
        pred["prop_uncertainty_agent_list_by_method"][uq_method] = [v for v in prop_values if v is not None]
        pred["prop_uncertainty_final_by_method"][uq_method] = (
            pred["prop_uncertainty_agent_list_by_method"][uq_method][-1]
            if pred["prop_uncertainty_agent_list_by_method"][uq_method]
            else None
        )
    if "Verb" in uq_methods:
        pred["local_uncertainty_agent_list"] = pred["local_uncertainty_agent_list_by_method"].get("Verb", [])
        pred["local_uncertainty_final"] = pred["local_uncertainty_final_by_method"].get("Verb")
        pred["prop_uncertainty_agent_list"] = pred["prop_uncertainty_agent_list_by_method"].get("Verb", [])
        pred["prop_uncertainty_final"] = pred["prop_uncertainty_final_by_method"].get("Verb")
        for agent in agents:
            if isinstance(agent.get("local_uncertainty_num_by_method"), dict):
                agent["local_uncertainty_num"] = agent["local_uncertainty_num_by_method"].get("Verb")
            if isinstance(agent.get("prop_uncertainty_by_method"), dict):
                agent["prop_uncertainty"] = agent["prop_uncertainty_by_method"].get("Verb")
    return pred


def apply_posthoc_uncertainty(preds: List[Dict], method: str):
    if method != "mas":
        return preds
    for pred in preds:
        enrich_mas_pred_with_posthoc_uncertainty(pred)
    return preds


def _final_uncertainty_from_pred(pred: Dict, uq_method: str = "Verb", source: str = "local"):
    if source == "prop":
        value = _pred_method_field(pred, "prop_uncertainty_final", uq_method)
        return parse_uncertainty_value(value)

    value = _pred_method_field(pred, "local_uncertainty_final", uq_method)
    if value is not None:
        return parse_uncertainty_value(value)
    agents = pred.get("agents", [])
    if agents:
        value = _agent_local_uncertainty_for_method(agents[-1], uq_method)
        if value is not None:
            return value
    if uq_method == "Verb":
        return extract_local_uncertainty(pred.get("raw_prediction", ""))
    return None


def _all_agent_uncertainties_from_pred(pred: Dict, uq_method: str = "Verb", source: str = "local"):
    if source == "prop":
        value = _pred_method_field(pred, "prop_uncertainty_agent_list", uq_method)
    else:
        value = _pred_method_field(pred, "local_uncertainty_agent_list", uq_method)
    if isinstance(value, list):
        vals = []
        for item in value:
            parsed = parse_uncertainty_value(item)
            if parsed is not None:
                vals.append(parsed)
        return vals

    vals = []
    agent_field = "prop_uncertainty" if source == "prop" else "local_uncertainty"
    for agent in pred.get("agents", []):
        if source == "local":
            parsed = _agent_local_uncertainty_for_method(agent, uq_method)
        else:
            parsed = parse_uncertainty_value(_agent_method_field(agent, agent_field, uq_method))
        if parsed is not None:
            vals.append(parsed)
    return vals


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


def _rejection_curve_auc_for_order(
    labels: List[int],
    rejection_order: List[int],
    max_rejections: int,
) -> float | None:
    n = len(labels)
    if n == 0 or max_rejections <= 0:
        return None

    kept = n
    kept_correct = sum(labels)
    rejection_rates = [0.0]
    retained_accuracies = [kept_correct / kept]

    for rejected_so_far, idx in enumerate(rejection_order[:max_rejections], start=1):
        kept -= 1
        kept_correct -= labels[idx]
        if kept <= 0:
            break
        rejection_rates.append(rejected_so_far / n)
        retained_accuracies.append(kept_correct / kept)

    if len(rejection_rates) < 2:
        return None

    auc = 0.0
    for idx in range(1, len(rejection_rates)):
        width = rejection_rates[idx] - rejection_rates[idx - 1]
        auc += width * (retained_accuracies[idx] + retained_accuracies[idx - 1]) / 2.0
    return auc


def _prr(
    labels: List[int],
    confidences: List[float],
    *,
    max_rejection_rate: float = PRR_MAX_REJECTION_RATE,
):
    n = len(labels)
    if n == 0 or n != len(confidences):
        return None

    max_rejections = min(n - 1, int(math.floor(max_rejection_rate * n)))
    if max_rejections <= 0:
        return None

    uncertainty_order = sorted(
        range(n),
        key=lambda idx: (confidences[idx], idx),
    )
    oracle_order = sorted(
        range(n),
        key=lambda idx: (labels[idx], confidences[idx], idx),
    )

    uncertainty_auc = _rejection_curve_auc_for_order(labels, uncertainty_order, max_rejections)
    oracle_auc = _rejection_curve_auc_for_order(labels, oracle_order, max_rejections)
    if uncertainty_auc is None or oracle_auc is None:
        return None

    random_auc = (sum(labels) / n) * (max_rejections / n)
    denom = oracle_auc - random_auc
    if abs(denom) <= 1e-12:
        return None
    return (uncertainty_auc - random_auc) / denom


def _uncertainty_to_confidence(
    u: Optional[float],
    uq_method: str,
    source: str = "local",
) -> Optional[float]:
    value = parse_uncertainty_value(u)
    if value is None:
        return None
    if uq_method == "Verb":
        return _clamp01(1.0 - value)
    if uq_method == "MSP":
        return _clamp01(1.0 - value)
    return None


def evaluate_uncertainty_metrics(preds: List[Dict], source: str = "local", uq_method: str = "Verb"):
    labels: List[int] = []
    probs_correct: List[float] = []
    ranking_scores: List[float] = []
    for pred in preds:
        u = _final_uncertainty_from_pred(pred, uq_method=uq_method, source=source)
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
        "prr": _prr(labels, probs_correct),
    }


def _filter_complete_case_preds(preds: List[Dict], uq_method: str) -> List[Dict]:
    filtered: List[Dict] = []
    for pred in preds:
        has_local = _final_uncertainty_from_pred(pred, uq_method=uq_method, source="local") is not None
        has_prop = _final_uncertainty_from_pred(pred, uq_method=uq_method, source="prop") is not None
        if has_local and has_prop:
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
            local_final = _final_uncertainty_from_pred(pred, uq_method=uq_method, source="local")
            prop_final = _final_uncertainty_from_pred(pred, uq_method=uq_method, source="prop")

            print(f"  UQ Method: {uq_method}")
            if agents:
                print("  Agent Uncertainties:")
                for agent in agents:
                    print(
                        f"    {agent.get('name', 'Agent')}: "
                        f"Local={_fmt_unc(_agent_local_uncertainty_for_method(agent, uq_method))} | "
                        f"Prop={_fmt_unc(_agent_method_field(agent, 'prop_uncertainty', uq_method))}"
                    )
            print(
                f"  Local-Uncertainty-Final: {_fmt_unc(local_final)}"
            )
            print(
                f"  Prop-Uncertainty-Final: {_fmt_unc(prop_final)}"
            )


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
    return {
        "Local-Uncertainty-Final": evaluate_uncertainty_metrics(
            complete_case_preds, source="local", uq_method=uq_method
        ),
        "Prop-Uncertainty-Final": evaluate_uncertainty_metrics(
            complete_case_preds, source="prop", uq_method=uq_method
        ),
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
                f"PRR={round4_or_none(values['prr'])}"
            )


def export_metrics_json(metrics: Dict, output_path: Path, metadata: Optional[Dict] = None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics}
    if metadata:
        payload["metadata"] = metadata
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
