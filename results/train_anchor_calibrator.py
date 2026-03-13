#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
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
        "This script now requires both numpy and torch in the active Python environment."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.uncertainty_quantification import (  # noqa: E402
    _agent_aliases,
    _aggregate_temporal_uncertainty,
    _auroc,
    _brier,
    _compute_formula_uncertainty,
    _compute_hazard_ht,
    _ece,
    _hazard_num_time_steps,
    _hazard_time_step_index,
    _incoming_agent_names,
    _normalize_agent_label,
)


LEVELS = ["VL", "L", "M", "H", "VH"]
LEVEL_TO_INDEX = {level: idx for idx, level in enumerate(LEVELS)}
DEFAULT_ANCHORS = {"VL": 0.0, "L": 0.25, "M": 0.5, "H": 0.75, "VH": 1.0}
FILENAME_PREFIX = "uq_preds_mas_"
EPS = 1e-6


@dataclass
class AgentExample:
    name: str
    role: str
    self_level: str | None
    message_levels: dict[str, str]


@dataclass
class Example:
    example_id: int
    correct: int
    agents: list[AgentExample]


@dataclass
class Candidate:
    self_anchors: dict[str, float]
    adoption_anchors: dict[str, float]
    calibrator_bias: float
    calibrator_scale: float


@dataclass
class CandidateMetrics:
    n: int
    accuracy: float
    auroc: float | None
    ece: float | None
    brier: float | None
    avg_confidence: float | None
    avg_uncertainty: float | None


class MonotonicAnchorModel(torch.nn.Module):
    def __init__(self, self_init: Sequence[float], adoption_init: Sequence[float]):
        super().__init__()
        self.self_base = torch.nn.Parameter(torch.tensor(logit_safe(self_init[0]), dtype=torch.float32))
        self.self_deltas = torch.nn.Parameter(
            torch.tensor(inverse_softplus_deltas(self_init), dtype=torch.float32)
        )
        self.adopt_base = torch.nn.Parameter(torch.tensor(logit_safe(adoption_init[0]), dtype=torch.float32))
        self.adopt_deltas = torch.nn.Parameter(
            torch.tensor(inverse_softplus_deltas(adoption_init), dtype=torch.float32)
        )
        self.calibrator_bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.calibrator_log_scale = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def monotonic_anchors(self, base_param: torch.Tensor, delta_params: torch.Tensor) -> torch.Tensor:
        base = torch.sigmoid(base_param).view(1)
        deltas = F.softplus(delta_params) + 1e-3
        csum = torch.cumsum(deltas, dim=0)
        raw = torch.cat([base, base + csum], dim=0)
        return project_monotonic_tensor(raw)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self_anchors = self.monotonic_anchors(self.self_base, self.self_deltas)
        adoption_anchors = self.monotonic_anchors(self.adopt_base, self.adopt_deltas)
        calibrator_scale = F.softplus(self.calibrator_log_scale) + 1e-3
        return self_anchors, adoption_anchors, self.calibrator_bias, calibrator_scale


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jointly calibrate anchor-based self_uncertainty and message_adoption with torch/numpy.",
    )
    parser.add_argument("--uq-dir", type=Path, default=ROOT / "preds" / "uq")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "anchor_calibration")
    parser.add_argument("--task", default="mbppplus")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument(
        "--selection-mode",
        choices=["auroc_first", "calibration_first", "balanced"],
        default="balanced",
        help="How to rank candidates on the validation split.",
    )
    return parser


def parse_filename(path: Path) -> dict[str, str] | None:
    name = path.name
    if not name.startswith(FILENAME_PREFIX) or not name.endswith(".jsonl"):
        return None
    stem = name[len(FILENAME_PREFIX):-len(".jsonl")]
    suffix = "_test_seed42_n378"
    if not stem.endswith(suffix):
        return None
    body = stem[: -len(suffix)]
    parts = body.split("_")
    if len(parts) < 4:
        return None
    mode = parts[-1]
    prompt = parts[-2]
    task = parts[-3]
    model = "_".join(parts[:-3]).replace("_", "/")
    return {"model": model, "task": task, "prompt": prompt, "mode": mode}


def load_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            payload = json.loads(line)
            agents = [
                AgentExample(
                    name=str(agent.get("name", "Agent")),
                    role=str(agent.get("role", "")).lower(),
                    self_level=agent.get("self_uncertainty"),
                    message_levels={str(k): str(v) for k, v in (agent.get("message_adoption") or {}).items()},
                )
                for agent in payload.get("agents", [])
            ]
            examples.append(Example(example_id=idx, correct=1 if payload.get("correct", False) else 0, agents=agents))
    return examples


def load_datasets(uq_dir: Path, task: str) -> dict[tuple[str, str], list[Example]]:
    datasets: dict[tuple[str, str], list[Example]] = {}
    for path in sorted(uq_dir.glob("uq_preds_*.jsonl")):
        parsed = parse_filename(path)
        if parsed is None or parsed["mode"] != "anchor" or parsed["task"] != task:
            continue
        datasets[(parsed["model"], parsed["prompt"])] = load_examples(path)
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


def subset_examples(examples: Sequence[Example], indices: Iterable[int]) -> list[Example]:
    return [examples[idx] for idx in indices]


def empirical_initialization(train_examples: Sequence[Example], source: str) -> list[float]:
    error_rates: dict[str, list[int]] = {level: [] for level in LEVELS}
    for example in train_examples:
        error = 1 - example.correct
        for agent in example.agents:
            if source == "self":
                if agent.self_level in error_rates:
                    error_rates[agent.self_level].append(error)
            else:
                for level in agent.message_levels.values():
                    if level in error_rates:
                        error_rates[level].append(error)
    values = [sum(error_rates[level]) / len(error_rates[level]) if error_rates[level] else DEFAULT_ANCHORS[level] for level in LEVELS]
    return make_monotonic(values)


def inverse_softplus(value: float) -> float:
    if value < 1e-6:
        return -20.0
    return math.log(math.expm1(value))


def inverse_softplus_deltas(values: Sequence[float]) -> list[float]:
    mono = make_monotonic(values)
    return [inverse_softplus(max(mono[idx + 1] - mono[idx], 1e-3)) for idx in range(len(mono) - 1)]


def logit_safe(value: float) -> float:
    value = min(max(float(value), EPS), 1.0 - EPS)
    return math.log(value / (1.0 - value))


def project_monotonic_tensor(raw: torch.Tensor) -> torch.Tensor:
    low = 1e-4
    high = 1.0 - 1e-4
    min_gap = 1e-3
    clipped = torch.clamp(raw, low, high)
    values = [clipped[0]]
    for idx in range(1, clipped.shape[0]):
        values.append(torch.maximum(clipped[idx], values[idx - 1] + min_gap))
    stacked = torch.stack(values)
    overflow = torch.relu(stacked[-1] - high)
    stacked = stacked - overflow
    values = [stacked[0]]
    for idx in range(1, stacked.shape[0]):
        values.append(torch.maximum(stacked[idx], values[idx - 1] + min_gap))
    stacked = torch.stack(values)
    return torch.clamp(stacked, low, high)


def make_monotonic(values: Sequence[float]) -> list[float]:
    arr = np.clip(np.sort(np.asarray(values, dtype=float)), 1e-4, 1.0 - 1e-4)
    min_gap = 0.015
    for idx in range(1, len(arr)):
        arr[idx] = max(arr[idx], arr[idx - 1] + min_gap)
    overflow = max(0.0, arr[-1] - (1.0 - 1e-4))
    if overflow > 0:
        arr -= overflow
        arr[0] = max(arr[0], 1e-4)
        for idx in range(1, len(arr)):
            arr[idx] = max(arr[idx], arr[idx - 1] + min_gap)
    return np.clip(arr, 1e-4, 1.0 - 1e-4).tolist()


def build_anchor_maps(candidate_self: Sequence[float], candidate_adopt: Sequence[float]) -> tuple[dict[str, float], dict[str, float]]:
    return (
        {level: float(candidate_self[idx]) for idx, level in enumerate(LEVELS)},
        {level: float(candidate_adopt[idx]) for idx, level in enumerate(LEVELS)},
    )


def compute_system_uncertainty(example: Example, prompt: str, self_map: dict[str, float], adopt_map: dict[str, float]) -> float | None:
    formula_states: dict[str, float] = {}
    hazard_num_steps = _hazard_num_steps(prompt, example.agents)
    hazard_step_formula_uncertainties: list[list[float | None]] = [[] for _ in range(hazard_num_steps)]

    for agent in example.agents:
        self_uncertainty = self_map.get(agent.self_level) if agent.self_level in self_map else None
        hazard_step_idx = _hazard_time_step_index(agent.role, prompt)
        formula_uncertainty = _compute_formula_uncertainty(
            self_uncertainty=self_uncertainty,
            message_adoption={k: adopt_map[v] for k, v in agent.message_levels.items() if v in adopt_map},
            previous_formula_uncertainties=formula_states,
            incoming_agent_names=_incoming_agent_names(agent.role, prompt, hazard_step_idx),
        )
        hazard_step_formula_uncertainties[hazard_step_idx].append(formula_uncertainty)
        for alias in _agent_aliases(agent.name, agent.role, prompt):
            if formula_uncertainty is not None:
                formula_states[_normalize_agent_label(alias)] = formula_uncertainty

    hazard_values = [_compute_hazard_ht(step_values) for step_values in hazard_step_formula_uncertainties]
    return _aggregate_temporal_uncertainty(hazard_values)


def _hazard_num_steps(prompt: str, agents: Sequence[AgentExample]) -> int:
    return _hazard_num_time_steps(prompt, [{"role": agent.role} for agent in agents])


def candidate_from_model(model: MonotonicAnchorModel) -> Candidate:
    with torch.no_grad():
        self_a, adopt_a, bias, scale = model()
    self_map, adopt_map = build_anchor_maps(self_a.tolist(), adopt_a.tolist())
    return Candidate(
        self_anchors=self_map,
        adoption_anchors=adopt_map,
        calibrator_bias=float(bias.item()),
        calibrator_scale=float(scale.item()),
    )


def baseline_candidate() -> Candidate:
    return Candidate(
        self_anchors=dict(DEFAULT_ANCHORS),
        adoption_anchors=dict(DEFAULT_ANCHORS),
        calibrator_bias=0.0,
        calibrator_scale=1.0,
    )


def calibrate_confidence(raw_uncertainty: float | None, candidate: Candidate) -> float | None:
    if raw_uncertainty is None:
        return None
    base_conf = np.clip(1.0 - raw_uncertainty, EPS, 1.0 - EPS)
    score = candidate.calibrator_bias + candidate.calibrator_scale * logit_safe(base_conf)
    return 1.0 / (1.0 + math.exp(-score))


def evaluate_candidate(examples: Sequence[Example], prompt: str, candidate: Candidate) -> CandidateMetrics:
    labels: list[int] = []
    probs: list[float] = []
    uncs: list[float] = []
    for example in examples:
        raw_unc = compute_system_uncertainty(example, prompt, candidate.self_anchors, candidate.adoption_anchors)
        conf = calibrate_confidence(raw_unc, candidate)
        if conf is None:
            continue
        labels.append(example.correct)
        probs.append(conf)
        uncs.append(1.0 - conf)
    accuracy = float(np.mean(labels)) if labels else 0.0
    avg_conf = float(np.mean(probs)) if probs else None
    avg_unc = float(np.mean(uncs)) if uncs else None
    return CandidateMetrics(
        n=len(labels),
        accuracy=accuracy,
        auroc=_auroc(labels, probs) if labels else None,
        ece=_ece(labels, probs) if labels else None,
        brier=_brier(labels, probs) if labels else None,
        avg_confidence=avg_conf,
        avg_uncertainty=avg_unc,
    )


def validation_score(metrics: CandidateMetrics, selection_mode: str) -> float:
    if metrics.n == 0:
        return float("-inf")
    auroc = metrics.auroc if metrics.auroc is not None else 0.0
    ece = metrics.ece if metrics.ece is not None else 1.0
    brier = metrics.brier if metrics.brier is not None else 1.0
    cal_gap = abs((metrics.avg_confidence or 0.0) - metrics.accuracy)
    if selection_mode == "auroc_first":
        return auroc
    if selection_mode == "calibration_first":
        return -(0.60 * ece + 0.35 * brier + 0.20 * cal_gap)
    return auroc - 0.35 * ece - 0.20 * brier - 0.10 * cal_gap


def train_objective(
    model: MonotonicAnchorModel,
    examples: Sequence[Example],
    prompt: str,
    default_self: Sequence[float],
    default_adopt: Sequence[float],
) -> torch.Tensor:
    self_a, adopt_a, bias, scale = model()
    self_map, adopt_map = build_anchor_maps(self_a.tolist(), adopt_a.tolist())
    logits = []
    labels = []
    for example in examples:
        raw_unc = compute_system_uncertainty(example, prompt, self_map, adopt_map)
        if raw_unc is None:
            continue
        base_conf = min(max(1.0 - raw_unc, EPS), 1.0 - EPS)
        logit_val = bias + scale * logit_safe(base_conf)
        logits.append(logit_val)
        labels.append(float(example.correct))
    if not logits:
        return torch.tensor(1e6, dtype=torch.float32)

    logits_t = torch.stack(logits)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    probs_t = torch.sigmoid(logits_t)
    bce = F.binary_cross_entropy(probs_t, labels_t)
    brier = torch.mean((probs_t - labels_t) ** 2)
    avg_conf = probs_t.mean()
    avg_acc = labels_t.mean()
    cal_penalty = torch.abs(avg_conf - avg_acc)
    default_self_t = torch.tensor(default_self, dtype=torch.float32)
    default_adopt_t = torch.tensor(default_adopt, dtype=torch.float32)
    reg = 0.01 * (
        torch.mean((self_a - default_self_t) ** 2)
        + torch.mean((adopt_a - default_adopt_t) ** 2)
        + 0.2 * bias.pow(2)
        + 0.05 * (scale - 1.0).pow(2)
    )
    return bce + 0.5 * brier + 0.2 * cal_penalty + reg


def optimize_candidate(
    train_examples: Sequence[Example],
    val_examples: Sequence[Example],
    prompt: str,
    *,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    restarts: int,
    selection_mode: str,
) -> tuple[Candidate, list[dict[str, object]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    default_self = [DEFAULT_ANCHORS[level] for level in LEVELS]
    default_adopt = [DEFAULT_ANCHORS[level] for level in LEVELS]
    empirical_self = empirical_initialization(train_examples, "self")
    empirical_adopt = empirical_initialization(train_examples, "adoption")

    initializations = [
        (default_self, default_adopt),
        (empirical_self, empirical_adopt),
    ]
    for _ in range(max(0, restarts - len(initializations))):
        initializations.append(
            (
                make_monotonic(np.random.uniform(0.0, 1.0, len(LEVELS))),
                make_monotonic(np.random.uniform(0.0, 1.0, len(LEVELS))),
            )
        )

    reports: list[dict[str, object]] = []
    best_candidate = baseline_candidate()
    best_score = float("-inf")

    for restart_idx, (self_init, adopt_init) in enumerate(initializations):
        model = MonotonicAnchorModel(self_init, adopt_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_state = None
        best_restart_score = float("-inf")
        patience = 60
        stale = 0

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = train_objective(model, train_examples, prompt, default_self, default_adopt)
            loss.backward()
            optimizer.step()

            candidate = candidate_from_model(model)
            val_metrics = evaluate_candidate(val_examples, prompt, candidate)
            score = validation_score(val_metrics, selection_mode)
            if score > best_restart_score:
                best_restart_score = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        candidate = candidate_from_model(model)
        train_metrics = evaluate_candidate(train_examples, prompt, candidate)
        val_metrics = evaluate_candidate(val_examples, prompt, candidate)
        score = validation_score(val_metrics, selection_mode)
        report = {
            "restart": restart_idx,
            "candidate": candidate_to_dict(candidate),
            "train_metrics": asdict(train_metrics),
            "val_metrics": asdict(val_metrics),
            "validation_score": score,
        }
        reports.append(report)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    reports.sort(key=lambda item: item["validation_score"], reverse=True)
    return best_candidate, reports


def candidate_to_dict(candidate: Candidate) -> dict[str, object]:
    return {
        "self_anchors": dict(candidate.self_anchors),
        "adoption_anchors": dict(candidate.adoption_anchors),
        "calibrator_bias": candidate.calibrator_bias,
        "calibrator_scale": candidate.calibrator_scale,
    }


def median_candidate(candidates: Sequence[Candidate]) -> Candidate:
    return Candidate(
        self_anchors={level: float(np.median([candidate.self_anchors[level] for candidate in candidates])) for level in LEVELS},
        adoption_anchors={level: float(np.median([candidate.adoption_anchors[level] for candidate in candidates])) for level in LEVELS},
        calibrator_bias=float(np.median([candidate.calibrator_bias for candidate in candidates])),
        calibrator_scale=float(np.median([candidate.calibrator_scale for candidate in candidates])),
    )


def mean_metric(metric_name: str, entries: Sequence[dict[str, object]]) -> float | None:
    values = [entry[metric_name] for entry in entries if entry[metric_name] is not None]
    return float(np.mean(values)) if values else None


def reliability_bins(examples: Sequence[Example], prompt: str, candidate: Candidate, bins: int) -> list[dict[str, object]]:
    counts = np.zeros(bins, dtype=int)
    sum_conf = np.zeros(bins, dtype=float)
    sum_acc = np.zeros(bins, dtype=float)
    for example in examples:
        raw_unc = compute_system_uncertainty(example, prompt, candidate.self_anchors, candidate.adoption_anchors)
        conf = calibrate_confidence(raw_unc, candidate)
        if conf is None:
            continue
        idx = min(int(conf * bins), bins - 1)
        counts[idx] += 1
        sum_conf[idx] += conf
        sum_acc[idx] += example.correct
    rows = []
    for idx in range(bins):
        rows.append(
            {
                "bin": idx,
                "count": int(counts[idx]),
                "avg_confidence": float(sum_conf[idx] / counts[idx]) if counts[idx] else None,
                "empirical_accuracy": float(sum_acc[idx] / counts[idx]) if counts[idx] else None,
            }
        )
    return rows


def summarize_combo(
    model: str,
    prompt: str,
    examples: Sequence[Example],
    *,
    folds: int,
    val_ratio: float,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    restarts: int,
    ece_bins: int,
    selection_mode: str,
) -> dict[str, object]:
    labels = [example.correct for example in examples]
    test_folds = stratified_kfold_indices(labels, folds, seed)
    fold_reports = []
    selected_candidates = []

    for fold_idx, test_indices in enumerate(test_folds):
        remaining = sorted(set(range(len(examples))) - set(test_indices))
        train_indices, val_indices = stratified_val_split(remaining, labels, val_ratio, seed + 1000 + fold_idx)
        train_examples = subset_examples(examples, train_indices)
        val_examples = subset_examples(examples, val_indices)
        test_examples = subset_examples(examples, test_indices)

        best_candidate, restart_reports = optimize_candidate(
            train_examples,
            val_examples,
            prompt,
            seed=seed + fold_idx,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            restarts=restarts,
            selection_mode=selection_mode,
        )
        baseline = baseline_candidate()
        selected_candidates.append(best_candidate)
        fold_reports.append(
            {
                "fold": fold_idx,
                "sizes": {"train": len(train_examples), "val": len(val_examples), "test": len(test_examples)},
                "baseline": {
                    "train": asdict(evaluate_candidate(train_examples, prompt, baseline)),
                    "val": asdict(evaluate_candidate(val_examples, prompt, baseline)),
                    "test": asdict(evaluate_candidate(test_examples, prompt, baseline)),
                },
                "calibrated": {
                    "train": asdict(evaluate_candidate(train_examples, prompt, best_candidate)),
                    "val": asdict(evaluate_candidate(val_examples, prompt, best_candidate)),
                    "test": asdict(evaluate_candidate(test_examples, prompt, best_candidate)),
                },
                "selected_candidate": candidate_to_dict(best_candidate),
                "restart_reports": restart_reports,
            }
        )

    recommended = median_candidate(selected_candidates)
    baseline = baseline_candidate()

    def _aggregate(section: str) -> dict[str, float | None]:
        base_entries = [fold["baseline"][section] for fold in fold_reports]
        cal_entries = [fold["calibrated"][section] for fold in fold_reports]
        return {
            "baseline_auroc_mean": mean_metric("auroc", base_entries),
            "calibrated_auroc_mean": mean_metric("auroc", cal_entries),
            "baseline_ece_mean": mean_metric("ece", base_entries),
            "calibrated_ece_mean": mean_metric("ece", cal_entries),
            "baseline_brier_mean": mean_metric("brier", base_entries),
            "calibrated_brier_mean": mean_metric("brier", cal_entries),
            "baseline_accuracy_mean": mean_metric("accuracy", base_entries),
            "calibrated_accuracy_mean": mean_metric("accuracy", cal_entries),
            "baseline_avg_confidence_mean": mean_metric("avg_confidence", base_entries),
            "calibrated_avg_confidence_mean": mean_metric("avg_confidence", cal_entries),
        }

    return {
        "model": model,
        "prompt": prompt,
        "task": "mbppplus",
        "selection_mode": selection_mode,
        "n_examples": len(examples),
        "folds": fold_reports,
        "cv_summary": {"train": _aggregate("train"), "val": _aggregate("val"), "test": _aggregate("test")},
        "recommended_candidate": candidate_to_dict(recommended),
        "full_data_baseline_metrics": asdict(evaluate_candidate(examples, prompt, baseline)),
        "full_data_recommended_metrics": asdict(evaluate_candidate(examples, prompt, recommended)),
        "full_data_reliability": {
            "baseline": reliability_bins(examples, prompt, baseline, ece_bins),
            "recommended": reliability_bins(examples, prompt, recommended, ece_bins),
        },
    }


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def render_anchor_dict(anchor_dict: dict[str, float]) -> str:
    return ", ".join(f"{level}={anchor_dict[level]:.3f}" for level in LEVELS)


def render_markdown(summary_rows: Sequence[dict[str, object]]) -> str:
    lines = [
        "# Anchor Calibration Summary",
        "",
        "| selection_mode | model | prompt | n | baseline_test_auroc | calibrated_test_auroc | baseline_test_ece | calibrated_test_ece | baseline_test_brier | calibrated_test_brier | recommended_self_anchors | recommended_adoption_anchors |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['selection_mode']} | {row['model']} | {row['prompt']} | {row['n_examples']} | "
            f"{fmt(row['cv_test']['baseline_auroc_mean'])} | {fmt(row['cv_test']['calibrated_auroc_mean'])} | "
            f"{fmt(row['cv_test']['baseline_ece_mean'])} | {fmt(row['cv_test']['calibrated_ece_mean'])} | "
            f"{fmt(row['cv_test']['baseline_brier_mean'])} | {fmt(row['cv_test']['calibrated_brier_mean'])} | "
            f"{row['recommended_self_anchors']} | {row['recommended_adoption_anchors']} |"
        )
    return "\n".join(lines) + "\n"


def write_summary_artifacts(output_dir: Path, combo_reports: Sequence[dict[str, object]]) -> None:
    summary_rows = []
    for report in combo_reports:
        recommended = report["recommended_candidate"]
        summary_rows.append(
            {
                "selection_mode": report["selection_mode"],
                "model": report["model"],
                "prompt": report["prompt"],
                "n_examples": report["n_examples"],
                "cv_test": report["cv_summary"]["test"],
                "recommended_self_anchors": render_anchor_dict(recommended["self_anchors"]),
                "recommended_adoption_anchors": render_anchor_dict(recommended["adoption_anchors"]),
            }
        )
    write_text(output_dir / "anchor_calibration_summary.json", json.dumps(combo_reports, indent=2, ensure_ascii=False) + "\n")
    write_text(output_dir / "anchor_calibration_summary.md", render_markdown(summary_rows))


def main() -> None:
    args = build_cli().parse_args()
    datasets = load_datasets(args.uq_dir, args.task)
    if not datasets:
        raise SystemExit(f"No anchor-mode uq_preds files found for task='{args.task}' in {args.uq_dir}")

    combo_reports = []
    for (model, prompt), examples in sorted(datasets.items()):
        combo_reports.append(
            summarize_combo(
                model,
                prompt,
                examples,
                folds=args.folds,
                val_ratio=args.val_ratio,
                seed=args.seed,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                restarts=args.restarts,
                ece_bins=args.ece_bins,
                selection_mode=args.selection_mode,
            )
        )

    write_summary_artifacts(args.output_dir, combo_reports)
    print(f"Wrote calibration artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()

# python /u/yliu105/MAS_UQ/results/train_anchor_calibrator.py --selection-mode auroc_first --output-dir /u/yliu105/MAS_UQ/results/anchor_calibration_auroc
# python /u/yliu105/MAS_UQ/results/train_anchor_calibrator.py --selection-mode calibration_first --output-dir /u/yliu105/MAS_UQ/results/anchor_calibration_calibration
# python /u/yliu105/MAS_UQ/results/train_anchor_calibrator.py --selection-mode balanced --output-dir /u/yliu105/MAS_UQ/results/anchor_calibration_balanced
