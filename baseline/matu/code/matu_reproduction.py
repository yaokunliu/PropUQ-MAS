#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Protocol

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "outputs" / "ASK4CONF"
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_TASK = "medqa"
TASK_ALIASES = {
    "mbbpplus": "mbppplus",
}
TASK_CHOICES = ("medqa", "gsm8k", "mbppplus")
TARGET_MODEL = "Qwen/Qwen3-8B"
DEFAULT_REPEATED_SEEDS = tuple(range(42, 52))
EXPECTED_REPEATED_TRAJECTORIES = 10
# MATU results are reported with the topology names used by this repository.
# Legacy raw files may use older topology names, so the scorer accepts both.
TARGET_CONFIGS = (
    ("sequential", "role", ("sequential", "chain")),
    ("hierarchical", "role", ("hierarchical", "star_convergent")),
)


@dataclass(frozen=True)
class RunSpec:
    seed: int
    topology: str
    mas_prompt: str
    path: Path


@dataclass
class ScoreRecord:
    task: str
    topology: str
    mas_prompt: str
    sample_idx: int
    question: str
    gold: str
    prediction: str
    correct: int
    label_mode: str
    num_repeated_trajectories: int
    num_correct_trajectories: int
    matu_uncertainty: float


class TextEmbedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


def normalize_task(task: str) -> str:
    normalized = TASK_ALIASES.get(task.strip().lower(), task.strip().lower())
    if normalized not in TASK_CHOICES:
        choices = ", ".join(TASK_CHOICES)
        raise argparse.ArgumentTypeError(f"Unknown task '{task}'. Expected one of: {choices}.")
    return normalized


def raw_path_for(raw_dir: Path, model: str, task: str, raw_topology: str, mas_prompt: str, seed: int) -> Path:
    model_dir = raw_dir
    for part in model.split("/"):
        model_dir /= part
    prompt_part = "_role" if mas_prompt == "role" else ""
    return model_dir / "raw" / f"raw_preds_mas_{task}_{raw_topology}_nodes4{prompt_part}_seed{seed}_n-1.jsonl"


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return tuple(seeds)


def validate_repeated_seeds(seeds: tuple[int, ...]) -> tuple[int, ...]:
    if len(seeds) != EXPECTED_REPEATED_TRAJECTORIES:
        raise ValueError(
            f"Strict MATU reproduction requires exactly {EXPECTED_REPEATED_TRAJECTORIES} repeated seeds; "
            f"got {len(seeds)}: {','.join(str(seed) for seed in seeds)}"
        )
    return seeds


def discover_specs(raw_dir: Path, seeds: tuple[int, ...], model: str, task: str) -> list[RunSpec]:
    specs: list[RunSpec] = []
    missing: list[Path] = []
    for topology, mas_prompt, raw_topologies in TARGET_CONFIGS:
        for seed in seeds:
            candidates = [raw_path_for(raw_dir, model, task, raw_topology, mas_prompt, seed) for raw_topology in raw_topologies]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                missing.extend(candidates)
                continue
            specs.append(RunSpec(seed=seed, topology=topology, mas_prompt=mas_prompt, path=path))
    if missing:
        example = "\n".join(f"- {path}" for path in missing[:12])
        suffix = "" if len(missing) <= 12 else f"\n... and {len(missing) - 12} more"
        raise FileNotFoundError(
            "Missing repeated-trajectory raw prediction files. "
            "Generate them first with baseline/matu/scripts/matu_reproduction_generate_repeats.sh.\n"
            f"{example}{suffix}"
        )
    return specs


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_preds(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def strip_structured_uncertainty_blocks(text: str) -> str:
    out = re.sub(
        r"<agent_uncertainty\b[^>]*>.*?</agent_uncertainty>",
        "",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    out = re.sub(r"<agent_uncertainty\b[^>]*/>", "", out, flags=re.IGNORECASE)
    out = re.sub(r"<message_adoption\b[^>]*/>", "", out, flags=re.IGNORECASE)
    out = re.sub(r"(?is)<think>.*?</think>", "", out)
    return out.strip()


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class HashEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        clean = strip_structured_uncertainty_blocks(text)
        tokens = TOKEN_RE.findall(clean.lower())
        if not tokens:
            tokens = [clean.lower()] if clean else ["empty"]

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little", signed=False)
            idx = value % self.dim
            sign = 1.0 if ((value >> 8) & 1) == 0 else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode_one(text) for text in texts], axis=0)


class Qwen3Embedder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        batch_size: int,
        max_length: int,
        local_files_only: bool,
    ) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_source = model_name
        if local_files_only:
            model_source = snapshot_download(model_name, local_files_only=True)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_source, local_files_only=local_files_only)
        self.model.to(self.device)
        self.model.eval()
        self.dim = int(getattr(self.model.config, "hidden_size"))

    def encode(self, texts: list[str]) -> np.ndarray:
        outputs: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = [strip_structured_uncertainty_blocks(text) for text in texts[start : start + self.batch_size]]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                model_out = self.model(**encoded)
                hidden = model_out.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.cpu().numpy().astype(np.float64, copy=False))
        return np.concatenate(outputs, axis=0)


def build_embedder(args) -> TextEmbedder:
    if args.embedding_backend == "hash":
        return HashEmbedder(args.embedding_dim)
    return Qwen3Embedder(
        args.embedding_model,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        local_files_only=args.local_files_only,
    )


def trajectory_texts(pred: dict) -> list[str]:
    agents = pred.get("agents") or []
    texts = [str(agent.get("output", "")) for agent in agents]
    if not texts:
        texts = [str(pred.get("raw_prediction", ""))]
    return texts


def trajectory_vectors(preds: list[dict], embedder: TextEmbedder) -> list[np.ndarray]:
    grouped_texts = [trajectory_texts(pred) for pred in preds]
    flat_texts = [text for texts in grouped_texts for text in texts]
    flat_embeddings = embedder.encode(flat_texts)

    vectors: list[np.ndarray] = []
    cursor = 0
    for texts in grouped_texts:
        count = len(texts)
        vectors.append(flat_embeddings[cursor : cursor + count].reshape(count * embedder.dim))
        cursor += count
    return vectors


def align_feature_dim(vectors: list[np.ndarray]) -> np.ndarray:
    max_dim = max(vec.shape[0] for vec in vectors)
    matrix = np.zeros((len(vectors), max_dim), dtype=np.float64)
    for idx, vec in enumerate(vectors):
        matrix[idx, : vec.shape[0]] = vec
    return matrix


def matu_reconstruction_uncertainty(
    trajectories: list[np.ndarray],
    *,
    max_rank: int,
) -> float:
    x = align_feature_dim(trajectories)
    n_runs = x.shape[0]
    if n_runs < 2:
        return 0.0

    usable_rank = max(1, min(max_rank, n_runs - 1, x.shape[1]))
    try:
        u, s, vt = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        mean = x.mean(axis=0, keepdims=True)
        return float(np.linalg.norm(x - mean))

    uncertainty = 0.0
    for rank in range(1, usable_rank + 1):
        reconstructed = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        uncertainty += float(np.linalg.norm(x - reconstructed))
    return uncertainty


def sample_label_and_prediction(preds: list[dict], *, label_mode: str) -> tuple[str, int, int]:
    correct_values = [1 if pred.get("correct", False) else 0 for pred in preds]
    num_correct = sum(correct_values)

    if label_mode == "first":
        return str(preds[0].get("prediction", "")), correct_values[0], num_correct
    if label_mode == "majority":
        prediction_counts: dict[str, int] = {}
        for pred in preds:
            prediction = str(pred.get("prediction", ""))
            prediction_counts[prediction] = prediction_counts.get(prediction, 0) + 1
        prediction = max(prediction_counts.items(), key=lambda item: (item[1], item[0]))[0]
        return prediction, 1 if num_correct >= math.ceil(len(preds) / 2) else 0, num_correct
    if label_mode == "all":
        return str(preds[0].get("prediction", "")), 1 if num_correct == len(preds) else 0, num_correct
    if label_mode == "any":
        return str(preds[0].get("prediction", "")), 1 if num_correct > 0 else 0, num_correct
    raise ValueError(f"Unsupported label mode: {label_mode}")


def collect_scores(args) -> tuple[list[ScoreRecord], float]:
    seeds = validate_repeated_seeds(parse_seeds(args.seeds))
    task = normalize_task(args.task)
    specs = discover_specs(args.raw_dir, seeds, args.target_model, task)
    embedder = build_embedder(args)

    specs_by_config: dict[tuple[str, str], list[RunSpec]] = {}
    for spec in specs:
        specs_by_config.setdefault((spec.topology, spec.mas_prompt), []).append(spec)

    records: list[ScoreRecord] = []
    uncertainty_compute_sec = 0.0
    for topology, mas_prompt, _raw_topology in TARGET_CONFIGS:
        config_specs = sorted(specs_by_config[(topology, mas_prompt)], key=lambda spec: seeds.index(spec.seed))
        preds_by_seed = {spec.seed: load_preds(spec.path) for spec in config_specs}
        n_samples = min(len(preds) for preds in preds_by_seed.values())

        for sample_idx in range(n_samples):
            sample_preds = [preds_by_seed[seed][sample_idx] for seed in seeds]
            uncertainty_start = time.perf_counter()
            trajectories = trajectory_vectors(sample_preds, embedder)
            uncertainty = matu_reconstruction_uncertainty(trajectories, max_rank=args.max_rank)
            uncertainty_compute_sec += time.perf_counter() - uncertainty_start
            prediction, correct, num_correct = sample_label_and_prediction(sample_preds, label_mode=args.label_mode)
            first_pred = sample_preds[0]

            records.append(
                ScoreRecord(
                    task=task,
                    topology=topology,
                    mas_prompt=mas_prompt,
                    sample_idx=sample_idx,
                    question=str(first_pred.get("question", "")),
                    gold=str(first_pred.get("gold", "")),
                    prediction=prediction,
                    correct=correct,
                    label_mode=args.label_mode,
                    num_repeated_trajectories=len(sample_preds),
                    num_correct_trajectories=num_correct,
                    matu_uncertainty=uncertainty,
                )
            )
    return records, uncertainty_compute_sec


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
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
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


def prr(labels: list[int], confidences: list[float], max_rejection_rate: float = 0.5) -> Optional[float]:
    n = len(labels)
    if n == 0 or n != len(confidences):
        return None
    max_rejections = min(n - 1, int(math.floor(max_rejection_rate * n)))
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


def metric_row(records: list[ScoreRecord], *, group: str) -> dict[str, object]:
    labels = [record.correct for record in records]
    confidences = [-record.matu_uncertainty for record in records]
    correct = sum(labels)
    n = len(labels)
    return {
        "group": group,
        "n": n,
        "accuracy": correct / n if n else None,
        "auroc": auroc(labels, confidences),
        "prr": prr(labels, confidences),
    }


def summarize(records: list[ScoreRecord]) -> list[dict[str, object]]:
    main_rows: list[dict[str, object]] = []

    for topology, mas_prompt, _raw_topologies in TARGET_CONFIGS:
        subset = [record for record in records if record.topology == topology and record.mas_prompt == mas_prompt]
        main_rows.append(metric_row(subset, group=f"{topology},{mas_prompt}"))
    return main_rows


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def score_records_as_rows(records: list[ScoreRecord]) -> list[dict[str, object]]:
    return [
        {
            "task": record.task,
            "topology": record.topology,
            "mas_prompt": record.mas_prompt,
            "sample_idx": record.sample_idx,
            "gold": record.gold,
            "prediction": record.prediction,
            "correct": record.correct,
            "label_mode": record.label_mode,
            "num_repeated_trajectories": record.num_repeated_trajectories,
            "num_correct_trajectories": record.num_correct_trajectories,
            "matu_uncertainty": record.matu_uncertainty,
        }
        for record in records
    ]


def write_markdown(path: Path, main_rows: list[dict[str, object]], args) -> None:
    seeds = validate_repeated_seeds(parse_seeds(args.seeds))
    task = normalize_task(args.task)
    lines = [
        f"# MATU Reproduction on {task}",
        "",
        "This reproduces MATU-style low-rank reconstruction uncertainty from `2604.08708v1.pdf` on repeated Qwen3-8B MAS trajectories.",
        "",
        "Configuration:",
        "- UQ source runs: ASK4CONF raw MAS outputs",
        f"- Task: `{task}`",
        "- MAS structures: `sequential, role`; `hierarchical, role`",
        f"- Model: `{args.target_model}`",
        f"- Repeated trajectories: {len(seeds)} seeds (`{args.seeds}`)",
        f"- Correctness label mode: `{args.label_mode}`",
        f"- Embedding backend: `{args.embedding_backend}`",
        f"- Embedding model: `{args.embedding_model}`",
        f"- Low-rank reconstruction ranks: 1..{args.max_rank}, capped by available runs",
        "",
        "## Main Results",
        "",
        "| MAS config | n | Accuracy | AUROC | PRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['group']} | {row['n']} | {fmt(row['accuracy'])} | {fmt(row['auroc'])} | {fmt(row['prr'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def config_group(topology: str, mas_prompt: str) -> str:
    return f"{topology},{mas_prompt}"


def load_generation_time_by_config(args) -> dict[str, float]:
    if args.generation_time_per_uncertainty_example_json is None:
        return {}
    path = args.generation_time_per_uncertainty_example_json
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    values: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            value = value.get("time_per_uncertainty_example_sec")
        if value is None:
            continue
        values[str(key)] = float(value)
    return values


def generation_runtime_summary(
    records: list[ScoreRecord],
    generation_time_by_config: dict[str, float],
) -> tuple[list[dict[str, object]], Optional[float], Optional[float]]:
    counts: dict[str, int] = {}
    for record in records:
        group = config_group(record.topology, record.mas_prompt)
        counts[group] = counts.get(group, 0) + 1

    rows: list[dict[str, object]] = []
    total_sec = 0.0
    all_available = bool(counts)
    ordered_groups = [config_group(topology, mas_prompt) for topology, mas_prompt, _ in TARGET_CONFIGS]
    ordered_groups.extend(group for group in counts if group not in ordered_groups)
    for group in ordered_groups:
        if group not in counts:
            continue
        n = counts[group]
        per_example = generation_time_by_config.get(group)
        elapsed = per_example * n if per_example is not None else None
        if elapsed is None:
            all_available = False
        else:
            total_sec += elapsed
        rows.append(
            {
                "mas_config": group,
                "num_uncertainty_examples": n,
                "trajectory_generation_time_per_uncertainty_example_sec": per_example,
                "trajectory_generation_elapsed_sec": elapsed,
            }
        )

    if not all_available:
        return rows, None, None
    n_records = len(records)
    per_example = total_sec / n_records if n_records else None
    return rows, total_sec, per_example


def write_runtime_files(
    output_dir: Path,
    *,
    args,
    records: list[ScoreRecord],
    elapsed_sec: float,
    uncertainty_compute_sec: float,
    started_at: str,
    finished_at: str,
) -> None:
    seeds = validate_repeated_seeds(parse_seeds(args.seeds))
    task = normalize_task(args.task)
    n_uncertainty_examples = len(records)
    n_configs = len(TARGET_CONFIGS)
    n_unique_samples = len({record.sample_idx for record in records})
    scoring_per_uncertainty_example_sec = (
        uncertainty_compute_sec / n_uncertainty_examples if n_uncertainty_examples else None
    )
    generation_time_by_config = load_generation_time_by_config(args)
    generation_rows, generation_elapsed_sec, generation_per_uncertainty_example_sec = generation_runtime_summary(
        records, generation_time_by_config
    )
    end_to_end_elapsed_sec = (
        generation_elapsed_sec + uncertainty_compute_sec if generation_elapsed_sec is not None else None
    )
    end_to_end_per_uncertainty_example_sec = (
        end_to_end_elapsed_sec / n_uncertainty_examples
        if end_to_end_elapsed_sec is not None and n_uncertainty_examples
        else None
    )
    time_by_config: list[dict[str, object]] = []
    for row in generation_rows:
        generation_per_example = row["trajectory_generation_time_per_uncertainty_example_sec"]
        end_to_end_per_example = (
            generation_per_example + scoring_per_uncertainty_example_sec
            if generation_per_example is not None and scoring_per_uncertainty_example_sec is not None
            else None
        )
        time_by_config.append(
            {
                **row,
                "uncertainty_scoring_time_per_uncertainty_example_sec": scoring_per_uncertainty_example_sec,
                "time_per_uncertainty_example_sec": end_to_end_per_example,
            }
        )

    payload = {
        "task": task,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "uncertainty_example_unit": "(topology, mas_prompt, sample_idx)",
        "num_uncertainty_examples": n_uncertainty_examples,
        "num_unique_samples": n_unique_samples,
        "num_mas_configs": n_configs,
        "num_repeated_trajectories": len(seeds),
        "seeds": list(seeds),
        "time_per_uncertainty_example_sec": end_to_end_per_uncertainty_example_sec,
        "time_per_uncertainty_example_note": (
            "End-to-end MATU time per uncertainty example: trajectory generation for all repeated "
            "trajectories plus uncertainty scoring. Null when trajectory-generation runtime is unavailable."
        ),
        "end_to_end_elapsed_sec": end_to_end_elapsed_sec,
        "end_to_end_elapsed_min": end_to_end_elapsed_sec / 60.0 if end_to_end_elapsed_sec is not None else None,
        "trajectory_generation_elapsed_sec": generation_elapsed_sec,
        "trajectory_generation_elapsed_min": (
            generation_elapsed_sec / 60.0 if generation_elapsed_sec is not None else None
        ),
        "trajectory_generation_time_per_uncertainty_example_sec": generation_per_uncertainty_example_sec,
        "uncertainty_scoring_elapsed_sec": uncertainty_compute_sec,
        "uncertainty_scoring_elapsed_min": uncertainty_compute_sec / 60.0,
        "uncertainty_scoring_time_per_uncertainty_example_sec": scoring_per_uncertainty_example_sec,
        "time_by_mas_config": time_by_config,
        "scoring_script_total_elapsed_sec": elapsed_sec,
        "scoring_script_total_elapsed_min": elapsed_sec / 60.0,
        "target_model": args.target_model,
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "embedding_batch_size": args.embedding_batch_size,
        "embedding_max_length": args.embedding_max_length,
        "max_rank": args.max_rank,
        "label_mode": args.label_mode,
        "raw_dir": display_path(args.raw_dir),
        "output_dir": display_path(output_dir),
    }

    json_path = output_dir / f"matu_{task}_runtime.json"
    md_path = output_dir / f"matu_{task}_runtime.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# MATU Runtime on {task}",
        "",
        f"- Started UTC: `{started_at}`",
        f"- Finished UTC: `{finished_at}`",
        "- Uncertainty example unit: `(topology, mas_prompt, sample_idx)`",
        f"- Uncertainty examples: `{n_uncertainty_examples}`",
        f"- Unique samples: `{n_unique_samples}`",
        f"- MAS configs: `{n_configs}`",
        f"- Repeated trajectories per sample: `{len(seeds)}`",
        (
            f"- Time per uncertainty example: `{end_to_end_per_uncertainty_example_sec:.4f}` seconds"
            if end_to_end_per_uncertainty_example_sec is not None
            else "- Time per uncertainty example: `NA`"
        ),
        (
            f"- End-to-end elapsed: `{end_to_end_elapsed_sec:.2f}` seconds (`{end_to_end_elapsed_sec / 60.0:.2f}` minutes)"
            if end_to_end_elapsed_sec is not None
            else "- End-to-end elapsed: `NA`"
        ),
        (
            f"- Trajectory generation elapsed: `{generation_elapsed_sec:.2f}` seconds (`{generation_elapsed_sec / 60.0:.2f}` minutes)"
            if generation_elapsed_sec is not None
            else "- Trajectory generation elapsed: `NA`"
        ),
        (
            f"- Trajectory generation time per uncertainty example: `{generation_per_uncertainty_example_sec:.4f}` seconds"
            if generation_per_uncertainty_example_sec is not None
            else "- Trajectory generation time per uncertainty example: `NA`"
        ),
        f"- Uncertainty scoring elapsed: `{uncertainty_compute_sec:.2f}` seconds (`{uncertainty_compute_sec / 60.0:.2f}` minutes)",
        (
            f"- Uncertainty scoring time per uncertainty example: `{scoring_per_uncertainty_example_sec:.4f}` seconds"
            if scoring_per_uncertainty_example_sec is not None
            else "- Uncertainty scoring time per uncertainty example: `NA`"
        ),
        f"- Scoring script total elapsed: `{elapsed_sec:.2f}` seconds (`{elapsed_sec / 60.0:.2f}` minutes)",
        "",
        "## Time By MAS Config",
        "",
        "| MAS config | Uncertainty examples | Trajectory generation / example | Uncertainty scoring / example | Time / uncertainty example |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in time_by_config:
        generation_per_example = row["trajectory_generation_time_per_uncertainty_example_sec"]
        scoring_per_example = row["uncertainty_scoring_time_per_uncertainty_example_sec"]
        end_to_end_per_example = row["time_per_uncertainty_example_sec"]
        lines.append(
            f"| {row['mas_config']} | {row['num_uncertainty_examples']} | "
            f"{fmt(generation_per_example)} | {fmt(scoring_per_example)} | {fmt(end_to_end_per_example)} |"
        )

    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- Model: `{args.target_model}`",
            f"- Embedding backend: `{args.embedding_backend}`",
            f"- Embedding model: `{args.embedding_model}`",
            f"- Embedding device: `{args.embedding_device}`",
            f"- Embedding batch size: `{args.embedding_batch_size}`",
            f"- Max rank: `{args.max_rank}`",
            f"- Label mode: `{args.label_mode}`",
            f"- Raw dir: `{display_path(args.raw_dir)}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce MATU uncertainty on repeated Qwen3-8B MAS outputs.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task", type=normalize_task, default=DEFAULT_TASK)
    parser.add_argument("--target-model", type=str, default=TARGET_MODEL)
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_REPEATED_SEEDS),
        help="Comma-separated repeated seeds. Strict MATU reproduction requires exactly 10 seeds.",
    )
    parser.add_argument("--max-rank", type=int, default=3)
    parser.add_argument("--embedding-backend", choices=("qwen3", "hash"), default="qwen3")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-device", type=str, default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=256, help="Only used by --embedding-backend hash.")
    parser.add_argument(
        "--generation-time-per-uncertainty-example-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping 'topology,mas_prompt' to seconds per uncertainty example "
            "for generating all repeated trajectories."
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=("first", "majority", "all", "any"),
        default="first",
        help="How to convert repeated trajectory correctness into one sample-level correctness label.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    args.task = normalize_task(args.task)
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    records, uncertainty_compute_sec = collect_scores(args)
    main_rows = summarize(records)
    elapsed_sec = time.perf_counter() - start_time
    finished_at = datetime.now(timezone.utc).isoformat()

    score_rows = score_records_as_rows(records)
    output_dir = args.output_dir / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / f"matu_{args.task}_scores.csv", score_rows)
    write_csv(output_dir / f"matu_{args.task}_main_results.csv", main_rows)
    write_markdown(output_dir / f"matu_{args.task}_results.md", main_rows, args)
    write_runtime_files(
        output_dir,
        args=args,
        records=records,
        elapsed_sec=elapsed_sec,
        uncertainty_compute_sec=uncertainty_compute_sec,
        started_at=started_at,
        finished_at=finished_at,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"matu_{args.task}_scores.csv", score_rows)
    write_csv(args.output_dir / f"matu_{args.task}_main_results.csv", main_rows)
    write_markdown(args.output_dir / f"matu_{args.task}_results.md", main_rows, args)
    write_runtime_files(
        args.output_dir,
        args=args,
        records=records,
        elapsed_sec=elapsed_sec,
        uncertainty_compute_sec=uncertainty_compute_sec,
        started_at=started_at,
        finished_at=finished_at,
    )

    print(f"Wrote {output_dir / f'matu_{args.task}_results.md'}")
    for row in main_rows:
        print(f"{row['group']}: AUROC={fmt(row['auroc'])} PRR={fmt(row['prr'])} n={row['n']}")


if __name__ == "__main__":
    main()
