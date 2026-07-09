#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "outputs" / "MSP"
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_GENERATION_RUNTIME_PATH = DEFAULT_OUTPUT_DIR / "medqa" / "generation_runtime"
TARGET_MODEL = "Qwen/Qwen3-8B"
TASK = "medqa"
TOPOLOGY = "sequential"
MAS_PROMPT = "role"
RAW_TOPOLOGY = "sequential"
NODE_NUM = 4
SEEDS = tuple(range(42, 52))
EXPECTED_REPEATED_TRAJECTORIES = 10
PRR_MAX_REJECTION_RATE = 0.5


@dataclass
class UPropRecord:
    task: str
    topology: str
    mas_prompt: str
    sample_idx: int
    gold: str
    prediction: str
    correct: int
    target_seed: int
    matched_trajectories: int
    num_repeated_trajectories: int
    uprop_uncertainty: float
    intrinsic_uncertainty: float
    extrinsic_uncertainty: float
    mean_step_pe: float


def raw_path_for(raw_dir: Path, seed: int) -> Path:
    model_dir = raw_dir
    for part in TARGET_MODEL.split("/"):
        model_dir /= part
    return (
        model_dir
        / "raw"
        / f"raw_preds_mas_{TASK}_{RAW_TOPOLOGY}_nodes{NODE_NUM}_role_seed{seed}_n-1.jsonl"
    )


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_raw_predictions(raw_dir: Path) -> dict[int, list[dict]]:
    missing: list[Path] = []
    preds_by_seed: dict[int, list[dict]] = {}
    for seed in SEEDS:
        path = raw_path_for(raw_dir, seed)
        if not path.exists():
            missing.append(path)
            continue
        preds_by_seed[seed] = list(iter_jsonl(path))
    if missing:
        example = "\n".join(f"- {path}" for path in missing[:10])
        raise FileNotFoundError(f"Missing UProp raw trajectory files:\n{example}")
    if len(preds_by_seed) != EXPECTED_REPEATED_TRAJECTORIES:
        raise ValueError(f"Expected exactly {EXPECTED_REPEATED_TRAJECTORIES} seeds, got {len(preds_by_seed)}.")
    return preds_by_seed


def strip_uncertainty_metadata(text: str) -> str:
    out = re.sub(
        r"<(?:agent|local)_uncertainty\b[^>]*>.*?</(?:agent|local)_uncertainty>",
        "",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    out = re.sub(r"<(?:agent|local)_uncertainty\b[^>]*/>", "", out, flags=re.IGNORECASE)
    out = re.sub(r"<message_adoption\b[^>]*/>", "", out, flags=re.IGNORECASE)
    out = re.sub(r"<alpha\b[^>]*/>", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def agent_text(agent: dict) -> str:
    for key in ("peer_output", "output", "raw_output", "content"):
        value = agent.get(key)
        if isinstance(value, str) and value.strip():
            return strip_uncertainty_metadata(value)
    return ""


def agent_negative_logprob(agent: dict, *, pe_normalization: str) -> float:
    token_logprobs = agent.get("token_logprobs")
    if not isinstance(token_logprobs, list) or not token_logprobs:
        raise ValueError(
            "UProp requires per-token log probabilities. Regenerate raw predictions with "
            "`--local_uncertainty_mode MSP` after the MAS trace has token_logprobs enabled."
        )
    values = [float(value) for value in token_logprobs if value is not None]
    if not values:
        raise ValueError("UProp found token_logprobs but no numeric log-probability values.")
    neg_logprob = -sum(values)
    if pe_normalization == "length":
        return neg_logprob / len(values)
    if pe_normalization == "sequence":
        return neg_logprob
    raise ValueError(f"Unsupported PE normalization: {pe_normalization}")


def trajectory_steps(pred: dict, *, pe_normalization: str) -> list[dict[str, float | str]]:
    agents = pred.get("agents")
    if isinstance(agents, list) and agents:
        ordered = sorted(agents, key=lambda item: int(item.get("index", 0)))
        steps = [
            {
                "text": agent_text(agent),
                "negative_logprob": agent_negative_logprob(agent, pe_normalization=pe_normalization),
            }
            for agent in ordered
            if not agent.get("skipped", False)
        ]
        if steps:
            return steps
    raise ValueError("UProp requires agent-level MAS traces with token probabilities.")


def token_f1_similarity(a: str, b: str, *, max_chars: int) -> float:
    tokens_a = re.findall(r"\w+", (a or "")[:max_chars].lower())
    tokens_b = re.findall(r"\w+", (b or "")[:max_chars].lower())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    overlap = sum((counts_a & counts_b).values())
    precision = overlap / len(tokens_a)
    recall = overlap / len(tokens_b)
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def similarity(a: str, b: str, *, max_chars: int, distance: str) -> float:
    if distance == "token_f1":
        return token_f1_similarity(a, b, max_chars=max_chars)
    a = (a or "")[:max_chars]
    b = (b or "")[:max_chars]
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=True).ratio()


def gaussian_kernel(distance: float, *, sharpness: float) -> float:
    distance = max(0.0, min(1.0, distance))
    return math.exp(-0.5 * sharpness * distance * distance)


def neighborhood_density(text: str, samples: list[str], *, max_chars: int, sharpness: float, distance: str) -> float:
    if not samples:
        return 1.0
    weights = []
    for sample in samples:
        text_distance = 1.0 - similarity(text, sample, max_chars=max_chars, distance=distance)
        weights.append(gaussian_kernel(text_distance, sharpness=sharpness))
    return max(sum(weights) / len(weights), 1e-12)


def tdp_score(
    trajectories: list[list[dict[str, float | str]]],
    *,
    target_idx: int,
    max_chars: int,
    kernel_sharpness: float,
    length_normalization: str,
    distance: str,
) -> tuple[float, float, float]:
    num_steps = min(len(traj) for traj in trajectories)
    iu_values: list[float] = []
    eu_values: list[float] = []

    for step_idx in range(num_steps):
        step_samples = [str(traj[step_idx]["text"]) for traj in trajectories]
        target_text = str(trajectories[target_idx][step_idx]["text"])
        iu = sum(float(traj[step_idx]["negative_logprob"]) for traj in trajectories) / len(trajectories)
        eu = 0.0
        for prev_idx in range(step_idx):
            prev_samples = [str(traj[prev_idx]["text"]) for traj in trajectories]
            prev_target = str(trajectories[target_idx][prev_idx]["text"])
            eu += -math.log(
                neighborhood_density(
                    prev_target,
                    prev_samples,
                    max_chars=max_chars,
                    sharpness=kernel_sharpness,
                    distance=distance,
                )
            )
        iu_values.append(iu)
        eu_values.append(eu)

    total_iu = sum(iu_values)
    total_eu = sum(eu_values)
    raw_total = total_iu + total_eu
    if length_normalization == "none":
        uncertainty = raw_total
    elif length_normalization == "mean":
        uncertainty = raw_total / num_steps if num_steps else 0.0
    elif length_normalization == "paper":
        denom = 0.0
        for iu, eu in zip(iu_values, eu_values):
            denom += 1.0 + (eu / max(iu, 1e-6))
        uncertainty = raw_total / max(denom, 1e-12)
    else:
        raise ValueError(f"Unsupported length normalization: {length_normalization}")
    return uncertainty, total_iu, total_eu


def target_indices_for_prediction(sample_preds: list[dict], target_idx: int) -> list[int]:
    target_prediction = str(sample_preds[target_idx].get("prediction", "")).strip().lower()
    matches = [
        idx
        for idx, pred in enumerate(sample_preds)
        if str(pred.get("prediction", "")).strip().lower() == target_prediction
    ]
    return matches or [target_idx]


def collect_records(args) -> tuple[list[UPropRecord], float]:
    preds_by_seed = load_raw_predictions(args.raw_dir)
    n_samples = min(len(preds) for preds in preds_by_seed.values())
    records: list[UPropRecord] = []
    scoring_time = 0.0

    for sample_idx in range(n_samples):
        sample_preds = [preds_by_seed[seed][sample_idx] for seed in SEEDS]
        target_idx = args.target_seed_index
        target_pred = sample_preds[target_idx]
        trajectories = [trajectory_steps(pred, pe_normalization=args.pe_normalization) for pred in sample_preds]
        target_indices = target_indices_for_prediction(sample_preds, target_idx)

        started = time.perf_counter()
        scored = [
            tdp_score(
                trajectories,
                target_idx=idx,
                max_chars=args.max_text_chars,
                kernel_sharpness=args.kernel_sharpness,
                length_normalization=args.length_normalization,
                distance=args.distance,
            )
            for idx in target_indices
        ]
        scoring_time += time.perf_counter() - started

        uncertainty = sum(item[0] for item in scored) / len(scored)
        intrinsic = sum(item[1] for item in scored) / len(scored)
        extrinsic = sum(item[2] for item in scored) / len(scored)
        records.append(
            UPropRecord(
                task=TASK,
                topology=TOPOLOGY,
                mas_prompt=MAS_PROMPT,
                sample_idx=sample_idx,
                gold=str(target_pred.get("gold", "")),
                prediction=str(target_pred.get("prediction", "")),
                correct=int(bool(target_pred.get("correct", False))),
                target_seed=SEEDS[target_idx],
                matched_trajectories=len(target_indices),
                num_repeated_trajectories=len(sample_preds),
                uprop_uncertainty=uncertainty,
                intrinsic_uncertainty=intrinsic,
                extrinsic_uncertainty=extrinsic,
                mean_step_pe=intrinsic / max(min(len(traj) for traj in trajectories), 1),
            )
        )
    return records, scoring_time


def auroc(labels: list[int], scores: list[float]) -> Optional[float]:
    n = len(labels)
    if n == 0 or n != len(scores):
        return None
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum_pos = 0.0
    i = 0
    rank = 1
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum_pos += avg_rank
        rank += j - i
        i = j
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    return u_stat / (n_pos * n_neg)


def rejection_curve_auc(labels: list[int], rejection_order: list[int], max_rejections: int) -> Optional[float]:
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


def prr(labels: list[int], confidences: list[float]) -> Optional[float]:
    n = len(labels)
    if n == 0 or n != len(confidences):
        return None
    max_rejections = min(n - 1, int(math.floor(PRR_MAX_REJECTION_RATE * n)))
    if max_rejections <= 0:
        return None
    uncertainty_order = sorted(range(n), key=lambda idx: (confidences[idx], idx))
    oracle_order = sorted(range(n), key=lambda idx: (labels[idx], confidences[idx], idx))
    uncertainty_auc = rejection_curve_auc(labels, uncertainty_order, max_rejections)
    oracle_auc = rejection_curve_auc(labels, oracle_order, max_rejections)
    if uncertainty_auc is None or oracle_auc is None:
        return None
    random_auc = (sum(labels) / n) * (max_rejections / n)
    denom = oracle_auc - random_auc
    if abs(denom) <= 1e-12:
        return None
    return (uncertainty_auc - random_auc) / denom


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(records: list[UPropRecord]) -> dict[str, object]:
    labels = [record.correct for record in records]
    confidences = [-record.uprop_uncertainty for record in records]
    n = len(records)
    return {
        "group": f"{TOPOLOGY},{MAS_PROMPT}",
        "n": n,
        "accuracy": sum(labels) / n if n else None,
        "auroc": auroc(labels, confidences),
        "prr": prr(labels, confidences),
        "mean_uprop_uncertainty": sum(record.uprop_uncertainty for record in records) / n if n else None,
        "mean_intrinsic_uncertainty": sum(record.intrinsic_uncertainty for record in records) / n if n else None,
        "mean_extrinsic_uncertainty": sum(record.extrinsic_uncertainty for record in records) / n if n else None,
        "mean_step_pe": sum(record.mean_step_pe for record in records) / n if n else None,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def record_rows(records: list[UPropRecord]) -> list[dict[str, object]]:
    return [
        {
            "task": record.task,
            "topology": record.topology,
            "mas_prompt": record.mas_prompt,
            "sample_idx": record.sample_idx,
            "gold": record.gold,
            "prediction": record.prediction,
            "correct": record.correct,
            "target_seed": record.target_seed,
            "matched_trajectories": record.matched_trajectories,
            "num_repeated_trajectories": record.num_repeated_trajectories,
            "uprop_uncertainty": record.uprop_uncertainty,
            "intrinsic_uncertainty": record.intrinsic_uncertainty,
            "extrinsic_uncertainty": record.extrinsic_uncertainty,
            "mean_step_pe": record.mean_step_pe,
        }
        for record in records
    ]


def write_results_md(path: Path, summary: dict[str, object], args) -> None:
    lines = [
        "# UProp Reproduction on medqa",
        "",
        "This reproduces a UProp-style uncertainty estimator from `12720_UProp_Investigating_the_ (1).pdf` using token log-probabilities from 10 repeated Qwen3-8B sequential-role MAS trajectories.",
        "",
        "Configuration:",
        "- Task: `medqa`",
        "- MAS structure: `sequential, role`",
        f"- Model: `{TARGET_MODEL}`",
        "- Repeated trajectories: exactly 10 seeds (`42-51`)",
        f"- Target prediction seed: `{SEEDS[args.target_seed_index]}`",
        f"- Decision distance: `{args.distance}`",
        f"- Intrinsic uncertainty: PE from token log-probabilities (`{args.pe_normalization}` normalization)",
        f"- Kernel sharpness: `{args.kernel_sharpness}`",
        f"- Length normalization: `{args.length_normalization}`",
        "",
        "## Main Results",
        "",
        "| MAS config | n | Accuracy | AUROC | PRR | Mean UProp | Mean IU | Mean EU | Mean Step PE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {summary['group']} | {summary['n']} | {fmt(summary['accuracy'])} | "
            f"{fmt(summary['auroc'])} | {fmt(summary['prr'])} | "
            f"{fmt(summary['mean_uprop_uncertainty'])} | {fmt(summary['mean_intrinsic_uncertainty'])} | "
            f"{fmt(summary['mean_extrinsic_uncertainty'])} | {fmt(summary['mean_step_pe'])} |"
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_generation_runtime(path: Path) -> tuple[list[dict], Optional[float]]:
    if not path.exists():
        return [], None
    rows: list[dict] = []
    total = 0.0
    if path.is_dir():
        for item in sorted(path.glob("*.json")):
            row = json.loads(item.read_text(encoding="utf-8"))
            row.pop("runtime_log", None)
            rows.append(row)
            value = row.get("total_time_sec")
            if value is not None:
                total += float(value)
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row.pop("runtime_log", None)
                rows.append(row)
                value = row.get("total_time_sec")
                if value is not None:
                    total += float(value)
    return rows, total


def write_runtime(
    path: Path,
    *,
    records: list[UPropRecord],
    scoring_sec: float,
    total_sec: float,
    generation_runtime_path: Path,
    started_at: str,
    finished_at: str,
) -> None:
    n = len(records)
    generation_rows, generation_sec = load_generation_runtime(generation_runtime_path)
    end_to_end_sec = generation_sec + scoring_sec if generation_sec is not None else None
    payload = {
        "task": TASK,
        "topology": TOPOLOGY,
        "mas_prompt": MAS_PROMPT,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "num_uncertainty_examples": n,
        "uncertainty_example_unit": "(topology, mas_prompt, sample_idx)",
        "num_repeated_trajectories": EXPECTED_REPEATED_TRAJECTORIES,
        "seeds": list(SEEDS),
        "generation_runtime_path": display_path(generation_runtime_path),
        "generation_runs": generation_rows,
        "trajectory_generation_elapsed_sec": generation_sec,
        "trajectory_generation_time_per_example_sec": generation_sec / n if generation_sec is not None and n else None,
        "uncertainty_scoring_elapsed_sec": scoring_sec,
        "uncertainty_scoring_time_per_example_sec": scoring_sec / n if n else None,
        "end_to_end_elapsed_sec": end_to_end_sec,
        "time_per_example_sec": end_to_end_sec / n if end_to_end_sec is not None and n else None,
        "time_per_example_note": "Trajectory generation for all 10 repeated samples plus UProp uncertainty scoring.",
        "script_total_elapsed_sec": total_sec,
        "script_total_time_per_example_sec": total_sec / n if n else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce UProp on MedQA sequential-role MAS trajectories.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-seed-index", type=int, default=0, choices=range(EXPECTED_REPEATED_TRAJECTORIES))
    parser.add_argument("--distance", choices=("token_f1", "difflib"), default="token_f1")
    parser.add_argument("--max-text-chars", type=int, default=1000)
    parser.add_argument("--kernel-sharpness", type=float, default=float(EXPECTED_REPEATED_TRAJECTORIES))
    parser.add_argument("--length-normalization", choices=("paper", "mean", "none"), default="paper")
    parser.add_argument("--pe-normalization", choices=("sequence", "length"), default="sequence")
    parser.add_argument("--generation-runtime-path", type=Path, default=DEFAULT_GENERATION_RUNTIME_PATH)
    return parser


def main() -> None:
    args = build_cli().parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    records, scoring_sec = collect_records(args)
    summary = summarize(records)
    total_sec = time.perf_counter() - start_time
    finished_at = datetime.now(timezone.utc).isoformat()

    task_dir = args.output_dir / TASK
    task_dir.mkdir(parents=True, exist_ok=True)
    write_csv(task_dir / "uprop_medqa_scores.csv", record_rows(records))
    write_csv(task_dir / "uprop_medqa_results.csv", [summary])
    write_results_md(task_dir / "uprop_medqa_results.md", summary, args)
    write_runtime(
        task_dir / "uprop_medqa_runtime.json",
        records=records,
        scoring_sec=scoring_sec,
        total_sec=total_sec,
        generation_runtime_path=args.generation_runtime_path,
        started_at=started_at,
        finished_at=finished_at,
    )

    print(f"Wrote {task_dir / 'uprop_medqa_results.md'}")
    print(f"{summary['group']}: AUROC={fmt(summary['auroc'])} PRR={fmt(summary['prr'])} n={summary['n']}")


if __name__ == "__main__":
    main()
