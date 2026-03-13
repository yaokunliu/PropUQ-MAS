#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires numpy and torch in the active Python environment."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.uncertainty_quantification import (  # noqa: E402
    _agent_aliases,
    _auroc,
    _brier,
    _compute_formula_uncertainty,
    _ece,
    _hazard_num_time_steps,
    _hazard_time_step_index,
    _incoming_agent_names,
    _normalize_agent_label,
    extract_agent_uncertainty,
    parse_uncertainty_value,
)


FILENAME_RE = re.compile(
    r"^uq_preds_mas_(?P<model>.+)_(?P<task>[^_]+)_(?P<prompt>[^_]+)_(?P<mode>anchor|continuous)_test_seed(?P<seed>\d+)_n(?P<n>-?\d+)\.jsonl$"
)
EPS = 1e-6


@dataclass
class HazardExample:
    example_id: int
    correct: int
    step_features: list[list[float]]
    baseline_system_uncertainty: float | None


@dataclass
class Metrics:
    n: int
    accuracy: float
    auroc: float | None
    ece: float | None
    brier: float | None
    log_loss: float | None
    avg_confidence: float | None
    avg_uncertainty: float | None


class LogisticHazardHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).squeeze(-1)


class TinyMLPHazardHead(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a learned hazard head on top of existing anchor/continuous UQ predictions.",
    )
    parser.add_argument("--uq-dir", type=Path, default=ROOT / "preds" / "uq")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "hazard_head")
    parser.add_argument("--task", default="mbppplus")
    parser.add_argument(
        "--uncertainty-modes",
        nargs="+",
        choices=["anchor", "continuous"],
        default=["anchor", "continuous"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--head-type", choices=["logistic", "mlp"], default="logistic")
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument(
        "--selection-metric",
        choices=["bce", "auroc"],
        default="bce",
        help="Validation metric used to select the restart checkpoint.",
    )
    return parser


def parse_filename(path: Path) -> dict[str, str] | None:
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None
    parsed = match.groupdict()
    parsed["model"] = parsed["model"].replace("_", "/")
    return parsed


def clamp01(x: float | None) -> float | None:
    if x is None:
        return None
    return max(0.0, min(1.0, float(x)))


def active_edge_adoption_values(message_adoption: dict[str, object], incoming_agent_names: Sequence[str]) -> list[float]:
    normalized_adoption = {
        _normalize_agent_label(agent_name): clamp01(parse_uncertainty_value(score))
        for agent_name, score in message_adoption.items()
        if clamp01(parse_uncertainty_value(score)) is not None
    }
    values: list[float] = []
    for incoming_name in incoming_agent_names:
        alpha = normalized_adoption.get(_normalize_agent_label(incoming_name))
        if alpha is not None:
            values.append(alpha)
    return values


def compute_hazard_features_from_pred(pred: dict[str, object], prompt: str) -> tuple[list[list[float]], float | None]:
    agents = pred.get("agents", []) or []
    if not agents:
        return [], pred.get("system_uncertainty_final")

    formula_states: dict[str, float] = {}
    hazard_num_steps = _hazard_num_time_steps(prompt, agents)
    hazard_step_formula_uncertainties: list[list[float | None]] = [[] for _ in range(hazard_num_steps)]
    hazard_step_adoption_values: list[list[float]] = [[] for _ in range(hazard_num_steps)]
    hazard_values: list[float | None] = []

    for agent in agents:
        role = str(agent.get("role", "")).lower()
        name = agent.get("name", "Agent")
        self_uncertainty = parse_uncertainty_value(agent.get("self_uncertainty"))
        if self_uncertainty is None:
            self_uncertainty = extract_agent_uncertainty(agent.get("output", ""))
        message_adoption = agent.get("message_adoption", {}) or {}
        hazard_step_idx = _hazard_time_step_index(role, prompt)
        incoming_agent_names = _incoming_agent_names(role, prompt, hazard_step_idx)
        formula_uncertainty = _compute_formula_uncertainty(
            self_uncertainty=self_uncertainty,
            message_adoption=message_adoption,
            previous_formula_uncertainties=formula_states,
            incoming_agent_names=incoming_agent_names,
        )
        hazard_step_formula_uncertainties[hazard_step_idx].append(formula_uncertainty)
        hazard_step_adoption_values[hazard_step_idx].extend(
            active_edge_adoption_values(message_adoption, incoming_agent_names)
        )
        for alias in _agent_aliases(name, role, prompt):
            if formula_uncertainty is not None:
                formula_states[_normalize_agent_label(alias)] = formula_uncertainty

    feature_rows: list[list[float]] = []
    for step_formula_uncertainties, step_adoption_values in zip(
        hazard_step_formula_uncertainties,
        hazard_step_adoption_values,
    ):
        pi_vals = [
            clamp01(value)
            for value in step_formula_uncertainties
            if clamp01(value) is not None
        ]
        if not pi_vals:
            hazard_values.append(None)
            continue
        mean_pi = float(sum(pi_vals) / len(pi_vals))
        max_pi = float(max(pi_vals))
        mean_alpha = float(sum(step_adoption_values) / len(step_adoption_values)) if step_adoption_values else 0.0
        feature_rows.append([mean_pi, max_pi, mean_alpha])
        hazard_values.append((mean_pi + max_pi) / 2.0)

    if hazard_values:
        prod = 1.0
        valid_count = 0
        for score in hazard_values:
            score = clamp01(score)
            if score is None:
                continue
            prod *= (1.0 - score)
            valid_count += 1
        baseline_system_uncertainty = None if valid_count == 0 else 1.0 - (prod ** (1.0 / valid_count))
    else:
        baseline_system_uncertainty = pred.get("system_uncertainty_final")

    return feature_rows, clamp01(baseline_system_uncertainty)


def load_examples(path: Path, prompt: str) -> list[HazardExample]:
    examples: list[HazardExample] = []
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            payload = json.loads(line)
            feature_rows, baseline_system_uncertainty = compute_hazard_features_from_pred(payload, prompt)
            examples.append(
                HazardExample(
                    example_id=idx,
                    correct=1 if payload.get("correct", False) else 0,
                    step_features=feature_rows,
                    baseline_system_uncertainty=baseline_system_uncertainty,
                )
            )
    return examples


def load_datasets(uq_dir: Path, task: str, uncertainty_modes: Sequence[str]) -> dict[tuple[str, str, str], list[HazardExample]]:
    datasets: dict[tuple[str, str, str], list[HazardExample]] = {}
    allowed = set(uncertainty_modes)
    for path in sorted(uq_dir.glob("uq_preds_*.jsonl")):
        parsed = parse_filename(path)
        if parsed is None or parsed["task"] != task or parsed["mode"] not in allowed:
            continue
        datasets[(parsed["model"], parsed["prompt"], parsed["mode"])] = load_examples(path, parsed["prompt"])
    return datasets


def stratified_kfold_indices(labels: Sequence[int], folds: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    pos = [idx for idx, label in enumerate(labels) if label == 1]
    neg = [idx for idx, label in enumerate(labels) if label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    split_folds = [[] for _ in range(folds)]
    for seq in (pos, neg):
        for idx, item in enumerate(seq):
            split_folds[idx % folds].append(item)
    for fold in split_folds:
        fold.sort()
    return split_folds


def stratified_val_split(indices: Sequence[int], labels: Sequence[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    pos = [idx for idx in indices if labels[idx] == 1]
    neg = [idx for idx in indices if labels[idx] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def _split(seq: list[int]) -> tuple[list[int], list[int]]:
        if not seq:
            return [], []
        n_val = max(1, int(round(len(seq) * val_ratio))) if len(seq) > 2 else 1
        n_val = min(n_val, len(seq) - 1) if len(seq) > 1 else 1
        return seq[n_val:], seq[:n_val]

    train_pos, val_pos = _split(pos)
    train_neg, val_neg = _split(neg)
    return sorted(train_pos + train_neg), sorted(val_pos + val_neg)


def subset_examples(examples: Sequence[HazardExample], indices: Iterable[int]) -> list[HazardExample]:
    return [examples[idx] for idx in indices]


def make_head(head_type: str, hidden_dim: int) -> torch.nn.Module:
    if head_type == "logistic":
        return LogisticHazardHead()
    return TinyMLPHazardHead(hidden_dim=hidden_dim)


def aggregate_system_uncertainty_torch(step_scores: torch.Tensor) -> torch.Tensor:
    step_scores = torch.clamp(step_scores, EPS, 1.0 - EPS)
    prod = torch.prod(1.0 - step_scores)
    return 1.0 - torch.pow(prod, 1.0 / step_scores.shape[0])


def model_prob_correct(example: HazardExample, head: torch.nn.Module) -> torch.Tensor | None:
    if not example.step_features:
        return None
    x = torch.tensor(example.step_features, dtype=torch.float32)
    step_h = head(x)
    system_unc = aggregate_system_uncertainty_torch(step_h)
    return torch.clamp(1.0 - system_unc, EPS, 1.0 - EPS)


def evaluate_baseline(examples: Sequence[HazardExample]) -> Metrics:
    labels: list[int] = []
    probs_correct: list[float] = []
    for example in examples:
        if example.baseline_system_uncertainty is None:
            continue
        prob_correct = 1.0 - float(example.baseline_system_uncertainty)
        prob_correct = min(max(prob_correct, EPS), 1.0 - EPS)
        labels.append(example.correct)
        probs_correct.append(prob_correct)
    return compute_metrics(labels, probs_correct)


def evaluate_head(examples: Sequence[HazardExample], head: torch.nn.Module) -> Metrics:
    labels: list[int] = []
    probs_correct: list[float] = []
    for example in examples:
        prob_correct_t = model_prob_correct(example, head)
        if prob_correct_t is None:
            continue
        labels.append(example.correct)
        probs_correct.append(float(prob_correct_t.detach().cpu().item()))
    return compute_metrics(labels, probs_correct)


def compute_metrics(labels: list[int], probs_correct: list[float]) -> Metrics:
    accuracy = float(np.mean(labels)) if labels else 0.0
    avg_conf = float(np.mean(probs_correct)) if probs_correct else None
    avg_unc = None if avg_conf is None else 1.0 - avg_conf
    log_loss = None
    if labels:
        log_loss = float(
            -np.mean(
                [
                    label * np.log(min(max(prob, EPS), 1.0 - EPS))
                    + (1 - label) * np.log(min(max(1.0 - prob, EPS), 1.0 - EPS))
                    for label, prob in zip(labels, probs_correct)
                ]
            )
        )
    return Metrics(
        n=len(labels),
        accuracy=accuracy,
        auroc=_auroc(labels, probs_correct) if labels else None,
        ece=_ece(labels, probs_correct) if labels else None,
        brier=_brier(labels, probs_correct) if labels else None,
        log_loss=log_loss,
        avg_confidence=avg_conf,
        avg_uncertainty=avg_unc,
    )


def validation_score(metrics: Metrics, selection_metric: str) -> float:
    if metrics.n == 0:
        return float("-inf")
    if selection_metric == "auroc":
        return metrics.auroc if metrics.auroc is not None else float("-inf")
    log_loss = metrics.log_loss if metrics.log_loss is not None else 1e6
    return -log_loss


def train_loss(examples: Sequence[HazardExample], head: torch.nn.Module) -> torch.Tensor:
    losses = []
    for example in examples:
        prob_correct = model_prob_correct(example, head)
        if prob_correct is None:
            continue
        target = torch.tensor(float(example.correct), dtype=torch.float32)
        losses.append(F.binary_cross_entropy(prob_correct, target))
    if not losses:
        return torch.tensor(1e6, dtype=torch.float32)
    return torch.stack(losses).mean()


def optimize_head(
    train_examples: Sequence[HazardExample],
    val_examples: Sequence[HazardExample],
    *,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    restarts: int,
    patience: int,
    head_type: str,
    hidden_dim: int,
    selection_metric: str,
) -> tuple[torch.nn.Module, list[dict[str, object]]]:
    reports: list[dict[str, object]] = []
    best_model = make_head(head_type, hidden_dim)
    best_score = float("-inf")

    for restart_idx in range(restarts):
        torch.manual_seed(seed + restart_idx)
        np.random.seed(seed + restart_idx)
        random.seed(seed + restart_idx)

        head = make_head(head_type, hidden_dim)
        optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
        best_state = None
        best_restart_score = float("-inf")
        stale = 0

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = train_loss(train_examples, head)
            loss.backward()
            optimizer.step()

            val_metrics = evaluate_head(val_examples, head)
            score = validation_score(val_metrics, selection_metric)
            if score > best_restart_score:
                best_restart_score = score
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        if best_state is not None:
            head.load_state_dict(best_state)

        train_metrics = evaluate_head(train_examples, head)
        val_metrics = evaluate_head(val_examples, head)
        score = validation_score(val_metrics, selection_metric)
        report = {
            "restart": restart_idx,
            "train_metrics": asdict(train_metrics),
            "val_metrics": asdict(val_metrics),
            "validation_score": score,
            "state_dict": {k: v.detach().cpu().tolist() for k, v in head.state_dict().items()},
        }
        reports.append(report)
        if score > best_score:
            best_score = score
            best_model = make_head(head_type, hidden_dim)
            best_model.load_state_dict(head.state_dict())

    reports.sort(key=lambda item: item["validation_score"], reverse=True)
    return best_model, reports


def state_dict_mean(state_dicts: Sequence[dict[str, object]]) -> dict[str, object]:
    mean_state: dict[str, object] = {}
    for key in state_dicts[0]:
        arrays = [np.asarray(state[key], dtype=float) for state in state_dicts]
        mean_state[key] = np.mean(arrays, axis=0).tolist()
    return mean_state


def summarize_combo(
    model: str,
    prompt: str,
    mode: str,
    task: str,
    examples: Sequence[HazardExample],
    *,
    folds: int,
    val_ratio: float,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    restarts: int,
    patience: int,
    head_type: str,
    hidden_dim: int,
    selection_metric: str,
) -> dict[str, object]:
    labels = [example.correct for example in examples]
    test_folds = stratified_kfold_indices(labels, folds, seed)
    fold_reports = []
    selected_states = []

    for fold_idx, test_indices in enumerate(test_folds):
        remaining = sorted(set(range(len(examples))) - set(test_indices))
        train_indices, val_indices = stratified_val_split(remaining, labels, val_ratio, seed + 1000 + fold_idx)
        train_examples = subset_examples(examples, train_indices)
        val_examples = subset_examples(examples, val_indices)
        test_examples = subset_examples(examples, test_indices)

        best_head, restart_reports = optimize_head(
            train_examples,
            val_examples,
            seed=seed + fold_idx,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            restarts=restarts,
            patience=patience,
            head_type=head_type,
            hidden_dim=hidden_dim,
            selection_metric=selection_metric,
        )
        selected_states.append({k: v.detach().cpu().tolist() for k, v in best_head.state_dict().items()})
        fold_reports.append(
            {
                "fold": fold_idx,
                "sizes": {"train": len(train_examples), "val": len(val_examples), "test": len(test_examples)},
                "baseline": {
                    "train": asdict(evaluate_baseline(train_examples)),
                    "val": asdict(evaluate_baseline(val_examples)),
                    "test": asdict(evaluate_baseline(test_examples)),
                },
                "learned": {
                    "train": asdict(evaluate_head(train_examples, best_head)),
                    "val": asdict(evaluate_head(val_examples, best_head)),
                    "test": asdict(evaluate_head(test_examples, best_head)),
                },
                "selected_state_dict": selected_states[-1],
                "restart_reports": restart_reports,
            }
        )

    recommended_state = state_dict_mean(selected_states)
    recommended_head = make_head(head_type, hidden_dim)
    recommended_head.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in recommended_state.items()})

    def _aggregate(section: str) -> dict[str, float | None]:
        base_entries = [fold["baseline"][section] for fold in fold_reports]
        learned_entries = [fold["learned"][section] for fold in fold_reports]
        return {
            "baseline_auroc_mean": mean_metric("auroc", base_entries),
            "learned_auroc_mean": mean_metric("auroc", learned_entries),
            "baseline_ece_mean": mean_metric("ece", base_entries),
            "learned_ece_mean": mean_metric("ece", learned_entries),
            "baseline_brier_mean": mean_metric("brier", base_entries),
            "learned_brier_mean": mean_metric("brier", learned_entries),
            "baseline_accuracy_mean": mean_metric("accuracy", base_entries),
            "learned_accuracy_mean": mean_metric("accuracy", learned_entries),
            "baseline_avg_confidence_mean": mean_metric("avg_confidence", base_entries),
            "learned_avg_confidence_mean": mean_metric("avg_confidence", learned_entries),
        }

    return {
        "model": model,
        "prompt": prompt,
        "task": task,
        "uncertainty_mode": mode,
        "head_type": head_type,
        "selection_metric": selection_metric,
        "n_examples": len(examples),
        "folds": fold_reports,
        "cv_summary": {"train": _aggregate("train"), "val": _aggregate("val"), "test": _aggregate("test")},
        "recommended_state_dict": recommended_state,
        "full_data_baseline_metrics": asdict(evaluate_baseline(examples)),
        "full_data_recommended_metrics": asdict(evaluate_head(examples, recommended_head)),
    }


def mean_metric(metric_name: str, entries: Sequence[dict[str, object]]) -> float | None:
    values = [entry[metric_name] for entry in entries if entry[metric_name] is not None]
    return float(np.mean(values)) if values else None


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_markdown(summary_rows: Sequence[dict[str, object]]) -> str:
    lines = [
        "# Hazard Head Summary",
        "",
        "| mode | model | prompt | n | baseline_test_auroc | learned_test_auroc | baseline_test_ece | learned_test_ece | baseline_test_brier | learned_test_brier | head_type | selection_metric |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary_rows:
        cv_test = row["cv_test"]
        lines.append(
            f"| {row['uncertainty_mode']} | {row['model']} | {row['prompt']} | {row['n_examples']} | "
            f"{fmt(cv_test['baseline_auroc_mean'])} | {fmt(cv_test['learned_auroc_mean'])} | "
            f"{fmt(cv_test['baseline_ece_mean'])} | {fmt(cv_test['learned_ece_mean'])} | "
            f"{fmt(cv_test['baseline_brier_mean'])} | {fmt(cv_test['learned_brier_mean'])} | "
            f"{row['head_type']} | {row['selection_metric']} |"
        )
    return "\n".join(lines) + "\n"


def write_summary_artifacts(output_dir: Path, combo_reports: Sequence[dict[str, object]]) -> None:
    summary_rows = []
    for report in combo_reports:
        summary_rows.append(
            {
                "uncertainty_mode": report["uncertainty_mode"],
                "model": report["model"],
                "prompt": report["prompt"],
                "n_examples": report["n_examples"],
                "head_type": report["head_type"],
                "selection_metric": report["selection_metric"],
                "cv_test": report["cv_summary"]["test"],
            }
        )
    write_text(output_dir / "hazard_head_summary.json", json.dumps(combo_reports, indent=2, ensure_ascii=False) + "\n")
    write_text(output_dir / "hazard_head_summary.md", render_markdown(summary_rows))


def main() -> None:
    args = build_cli().parse_args()
    datasets = load_datasets(args.uq_dir, args.task, args.uncertainty_modes)
    if not datasets:
        raise SystemExit(
            f"No uq_preds files found for task='{args.task}' and modes={args.uncertainty_modes} in {args.uq_dir}"
        )

    combo_reports = []
    for (model, prompt, mode), examples in sorted(datasets.items()):
        combo_reports.append(
            summarize_combo(
                model,
                prompt,
                mode,
                args.task,
                examples,
                folds=args.folds,
                val_ratio=args.val_ratio,
                seed=args.seed,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                restarts=args.restarts,
                patience=args.patience,
                head_type=args.head_type,
                hidden_dim=args.hidden_dim,
                selection_metric=args.selection_metric,
            )
        )

    write_summary_artifacts(args.output_dir, combo_reports)
    print(f"Wrote hazard-head artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()

# conda run -n mas python /u/yliu105/MAS_UQ/results/train_hazard_head.py --task mbppplus --uncertainty-modes continuous --head-type logistic
# conda run -n mas python /u/yliu105/MAS_UQ/results/train_hazard_head.py --task mbppplus --head-type mlp --hidden-dim 8
# conda run -n mas python /u/yliu105/MAS_UQ/results/train_hazard_head.py \
#   --task mbppplus \
#   --uncertainty-modes continuous \
#   --head-type logistic \
#   --selection-metric bce \
#   --output-dir /u/yliu105/MAS_UQ/results/hazard_head_mbppplus_continuous_logistic_bce


