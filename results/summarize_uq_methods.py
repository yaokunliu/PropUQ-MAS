#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


RESULT_JSON_PATTERN = re.compile(r'^\{"method":.*"accuracy":.*\}$')
FIELD_PATTERNS = {
    "model": re.compile(r"^MODEL:\s+(.+)$", re.MULTILINE),
    "prompt": re.compile(r"^PROMPT:\s+(.+)$", re.MULTILINE),
    "task": re.compile(r"^TASK:\s+(.+)$", re.MULTILINE),
    "args": re.compile(r"^ARGS:\s+(.+)$", re.MULTILINE),
}


@dataclass
class AccuracyRecord:
    mode: str
    model: str
    prompt: str
    task: str
    accuracy: float
    correct: int | None
    max_samples: int | None
    log_path: Path


@dataclass
class UQMetricRecord:
    mode: str
    model: str
    prompt: str
    task: str
    accuracy: float | None
    metrics: dict[str, dict[str, float | int]]
    best_auroc_metric: str | None
    best_auroc: float | None
    best_ece_metric: str | None
    best_ece: float | None
    best_brier_metric: str | None
    best_brier: float | None
    metrics_path: Path


def build_cli() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Summarize experiment results for different UQ methods into comparison tables.",
    )
    parser.add_argument("--log-dir", type=Path, default=root / "log")
    parser.add_argument("--uq-dir", type=Path, default=root / "preds" / "uq")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    return parser


def extract_field(text: str, field: str) -> str | None:
    match = FIELD_PATTERNS[field].search(text)
    return match.group(1).strip() if match else None


def find_last_result_json(text: str) -> dict | None:
    result = None
    for line in text.splitlines():
        line = line.strip()
        if not RESULT_JSON_PATTERN.match(line):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "accuracy" in payload:
            result = payload
    return result


def detect_mode(log_name: str, args_line: str | None) -> str:
    if "no_uq" in log_name:
        return "no_uq"
    if args_line and "--uncertainty_mode anchor" in args_line:
        return "anchor"
    if args_line and "--uncertainty_mode continuous" in args_line:
        return "continuous"
    if "anchor" in log_name:
        return "anchor"
    if "cont" in log_name:
        return "continuous"
    return "unknown"


def collect_accuracy_records(log_dir: Path) -> list[AccuracyRecord]:
    records: list[AccuracyRecord] = []
    for log_path in sorted(log_dir.glob("*.out")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        payload = find_last_result_json(text)
        if payload is None:
            continue
        model = extract_field(text, "model") or payload.get("model")
        prompt = extract_field(text, "prompt")
        task = extract_field(text, "task")
        args_line = extract_field(text, "args")
        mode = detect_mode(log_path.name, args_line)
        if not model or not prompt or not task or mode == "unknown":
            continue
        records.append(
            AccuracyRecord(
                mode=mode,
                model=model,
                prompt=prompt,
                task=task,
                accuracy=float(payload["accuracy"]),
                correct=payload.get("correct"),
                max_samples=payload.get("max_samples"),
                log_path=log_path,
            )
        )
    return records


def choose_best_accuracy(records: list[AccuracyRecord]) -> dict[tuple[str, str, str, str], AccuracyRecord]:
    best: dict[tuple[str, str, str, str], AccuracyRecord] = {}
    for record in records:
        key = (record.task, record.prompt, record.model, record.mode)
        current = best.get(key)
        if current is None or record.accuracy > current.accuracy:
            best[key] = record
    return best


def load_uq_metric_records(uq_dir: Path) -> list[UQMetricRecord]:
    records: list[UQMetricRecord] = []
    for path in sorted(uq_dir.glob("uq_metrics_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        metrics = payload.get("metrics", {})
        mode = metadata.get("uncertainty_mode", "unknown")
        best_auroc_name, best_auroc = select_metric(metrics, "auroc", higher_is_better=True)
        best_ece_name, best_ece = select_metric(metrics, "ece", higher_is_better=False)
        best_brier_name, best_brier = select_metric(metrics, "brier", higher_is_better=False)
        records.append(
            UQMetricRecord(
                mode=mode,
                model=metadata.get("model", ""),
                prompt=metadata.get("prompt", ""),
                task=metadata.get("task", ""),
                accuracy=metadata.get("accuracy"),
                metrics=metrics,
                best_auroc_metric=best_auroc_name,
                best_auroc=best_auroc,
                best_ece_metric=best_ece_name,
                best_ece=best_ece,
                best_brier_metric=best_brier_name,
                best_brier=best_brier,
                metrics_path=path,
            )
        )
    return records


def select_metric(metrics: dict, metric_name: str, higher_is_better: bool) -> tuple[str | None, float | None]:
    best_name = None
    best_value = None
    for name, values in metrics.items():
        value = values.get(metric_name)
        if value is None:
            continue
        value = float(value)
        if best_value is None:
            best_name = name
            best_value = value
            continue
        if higher_is_better and value > best_value:
            best_name = name
            best_value = value
        if not higher_is_better and value < best_value:
            best_name = name
            best_value = value
    return best_name, best_value


def format_float(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def format_csv_value(value: object) -> str:
    text = "" if value is None else str(value)
    if any(ch in text for ch in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_accuracy_tables(best_records: dict[tuple[str, str, str, str], AccuracyRecord], output_dir: Path) -> str:
    discovered_modes = {key[3] for key in best_records}
    preferred_order = ["no_uq", "anchor", "continuous"]
    modes = [mode for mode in preferred_order if mode in discovered_modes]
    modes.extend(sorted(discovered_modes - set(modes)))
    combos = sorted({key[:3] for key in best_records})
    rows: list[dict[str, object]] = []
    delta_modes = [mode for mode in modes if mode != "no_uq"]
    markdown_lines = [
        "| task | prompt | model | "
        + " | ".join(f"{mode}_acc" for mode in modes)
        + " | "
        + " | ".join(f"{mode}_vs_no_uq" for mode in delta_modes)
        + " | best_method | best_acc | best_vs_no_uq |",
        "| --- | --- | --- | "
        + " | ".join("---:" for _ in modes)
        + " | "
        + " | ".join("---:" for _ in delta_modes)
        + " | --- | ---: | ---: |",
    ]

    for task, prompt, model in combos:
        row: dict[str, object] = {"task": task, "prompt": prompt, "model": model}
        mode_to_acc: dict[str, float] = {}
        mode_to_log: dict[str, str] = {}
        for mode in modes:
            record = best_records.get((task, prompt, model, mode))
            acc = record.accuracy if record else None
            row[f"{mode}_acc"] = acc
            row[f"{mode}_log"] = str(record.log_path) if record else ""
            if acc is not None:
                mode_to_acc[mode] = acc
                mode_to_log[mode] = str(record.log_path)

        if mode_to_acc:
            best_acc = max(mode_to_acc.values())
            best_methods = sorted(mode for mode, acc in mode_to_acc.items() if acc == best_acc)
            best_method = "/".join(best_methods)
        else:
            best_acc = None
            best_method = "NA"

        no_uq_acc = mode_to_acc.get("no_uq")
        for mode in delta_modes:
            mode_acc = mode_to_acc.get(mode)
            row[f"{mode}_vs_no_uq"] = None if mode_acc is None or no_uq_acc is None else mode_acc - no_uq_acc
        best_vs_no_uq = None if best_acc is None or no_uq_acc is None else best_acc - no_uq_acc
        row["best_method"] = best_method
        row["best_acc"] = best_acc
        row["best_vs_no_uq"] = best_vs_no_uq
        rows.append(row)

        markdown_lines.append(
            f"| {task} | {prompt} | {model} | "
            + " | ".join(format_float(row[f"{mode}_acc"]) if isinstance(row[f"{mode}_acc"], float) else "NA" for mode in modes)
            + " | "
            + " | ".join(
                f"{row[f'{mode}_vs_no_uq']:+.4f}" if isinstance(row[f"{mode}_vs_no_uq"], float) else "NA"
                for mode in delta_modes
            )
            + f" | {best_method} | {format_float(best_acc)} | "
            + (f"{best_vs_no_uq:+.4f}" if best_vs_no_uq is not None else "NA")
            + " |"
        )

    csv_headers = (
        ["task", "prompt", "model"]
        + [f"{mode}_acc" for mode in modes]
        + [f"{mode}_vs_no_uq" for mode in delta_modes]
        + [f"{mode}_log" for mode in modes]
        + ["best_method", "best_acc", "best_vs_no_uq"]
    )
    csv_lines = [",".join(csv_headers)]
    for row in rows:
        csv_lines.append(",".join(format_csv_value(row.get(header)) for header in csv_headers))

    json_path = output_dir / "uq_method_accuracy_summary.json"
    md_path = output_dir / "uq_method_accuracy_summary.md"
    csv_path = output_dir / "uq_method_accuracy_summary.csv"
    write_text(json_path, json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    write_text(md_path, "# UQ Method Accuracy Summary\n\n" + "\n".join(markdown_lines) + "\n")
    write_text(csv_path, "\n".join(csv_lines) + "\n")
    return f"Accuracy table: {md_path}"


def build_uq_metric_tables(records: list[UQMetricRecord], output_dir: Path) -> str:
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    summary_markdown_lines = [
        "| task | prompt | model | mode | accuracy | best_auroc_metric | best_auroc | best_ece_metric | best_ece | best_brier_metric | best_brier |",
        "| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: |",
    ]
    detail_markdown_lines = [
        "| task | prompt | model | mode | accuracy | metric_name | auroc | auroc_gap_to_best | ece | ece_gap_to_best | brier | brier_gap_to_best | n_with_uncertainty |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for record in sorted(records, key=lambda x: (x.task, x.prompt, x.model, x.mode)):
        summary_row = {
            "task": record.task,
            "prompt": record.prompt,
            "model": record.model,
            "mode": record.mode,
            "accuracy": record.accuracy,
            "best_auroc_metric": record.best_auroc_metric,
            "best_auroc": record.best_auroc,
            "best_ece_metric": record.best_ece_metric,
            "best_ece": record.best_ece,
            "best_brier_metric": record.best_brier_metric,
            "best_brier": record.best_brier,
            "metrics_path": str(record.metrics_path),
        }
        summary_rows.append(summary_row)
        summary_markdown_lines.append(
            f"| {record.task} | {record.prompt} | {record.model} | {record.mode} | "
            f"{format_float(record.accuracy)} | {record.best_auroc_metric or 'NA'} | {format_float(record.best_auroc)} | "
            f"{record.best_ece_metric or 'NA'} | {format_float(record.best_ece)} | "
            f"{record.best_brier_metric or 'NA'} | {format_float(record.best_brier)} |"
        )

        for metric_name, metric_values in sorted(record.metrics.items()):
            auroc = metric_values.get("auroc")
            ece = metric_values.get("ece")
            brier = metric_values.get("brier")
            n_with_uncertainty = metric_values.get("n_with_uncertainty")
            auroc_gap = None if auroc is None or record.best_auroc is None else record.best_auroc - float(auroc)
            ece_gap = None if ece is None or record.best_ece is None else float(ece) - record.best_ece
            brier_gap = None if brier is None or record.best_brier is None else float(brier) - record.best_brier
            detail_row = {
                "task": record.task,
                "prompt": record.prompt,
                "model": record.model,
                "mode": record.mode,
                "accuracy": record.accuracy,
                "metric_name": metric_name,
                "auroc": auroc,
                "auroc_gap_to_best": auroc_gap,
                "ece": ece,
                "ece_gap_to_best": ece_gap,
                "brier": brier,
                "brier_gap_to_best": brier_gap,
                "n_with_uncertainty": n_with_uncertainty,
                "metrics_path": str(record.metrics_path),
            }
            detail_rows.append(detail_row)
            detail_markdown_lines.append(
                f"| {record.task} | {record.prompt} | {record.model} | {record.mode} | "
                f"{format_float(record.accuracy)} | {metric_name} | "
                f"{format_float(float(auroc) if auroc is not None else None)} | "
                f"{f'+{auroc_gap:.4f}' if isinstance(auroc_gap, float) else 'NA'} | "
                f"{format_float(float(ece) if ece is not None else None)} | "
                f"{f'+{ece_gap:.4f}' if isinstance(ece_gap, float) else 'NA'} | "
                f"{format_float(float(brier) if brier is not None else None)} | "
                f"{f'+{brier_gap:.4f}' if isinstance(brier_gap, float) else 'NA'} | "
                f"{n_with_uncertainty if n_with_uncertainty is not None else 'NA'} |"
            )

    summary_csv_headers = [
        "task",
        "prompt",
        "model",
        "mode",
        "accuracy",
        "best_auroc_metric",
        "best_auroc",
        "best_ece_metric",
        "best_ece",
        "best_brier_metric",
        "best_brier",
        "metrics_path",
    ]
    summary_csv_lines = [",".join(summary_csv_headers)]
    for row in summary_rows:
        summary_csv_lines.append(",".join(format_csv_value(row.get(header)) for header in summary_csv_headers))

    detail_csv_headers = [
        "task",
        "prompt",
        "model",
        "mode",
        "accuracy",
        "metric_name",
        "auroc",
        "auroc_gap_to_best",
        "ece",
        "ece_gap_to_best",
        "brier",
        "brier_gap_to_best",
        "n_with_uncertainty",
        "metrics_path",
    ]
    detail_csv_lines = [",".join(detail_csv_headers)]
    for row in detail_rows:
        detail_csv_lines.append(",".join(format_csv_value(row.get(header)) for header in detail_csv_headers))

    summary_json_path = output_dir / "uq_metric_summary.json"
    summary_md_path = output_dir / "uq_metric_summary.md"
    summary_csv_path = output_dir / "uq_metric_summary.csv"
    detail_json_path = output_dir / "uq_metric_details.json"
    detail_md_path = output_dir / "uq_metric_details.md"
    detail_csv_path = output_dir / "uq_metric_details.csv"
    write_text(summary_json_path, json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n")
    write_text(summary_md_path, "# UQ Metric Summary\n\n" + "\n".join(summary_markdown_lines) + "\n")
    write_text(summary_csv_path, "\n".join(summary_csv_lines) + "\n")
    write_text(detail_json_path, json.dumps(detail_rows, indent=2, ensure_ascii=False) + "\n")
    write_text(detail_md_path, "# UQ Metric Details\n\n" + "\n".join(detail_markdown_lines) + "\n")
    write_text(detail_csv_path, "\n".join(detail_csv_lines) + "\n")
    return f"UQ metric tables: {summary_md_path}, {detail_md_path}"


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    target_output_dir = args.output_dir / "uq_metric"

    accuracy_records = collect_accuracy_records(args.log_dir)
    best_accuracy = choose_best_accuracy(accuracy_records)
    metric_records = load_uq_metric_records(args.uq_dir)

    if not best_accuracy:
        raise SystemExit(f"No experiment logs found in {args.log_dir}")

    accuracy_msg = build_accuracy_tables(best_accuracy, target_output_dir)
    metric_msg = build_uq_metric_tables(metric_records, target_output_dir)

    print(f"Collected {len(accuracy_records)} log records, {len(best_accuracy)} best accuracy records.")
    print(accuracy_msg)
    print(f"Collected {len(metric_records)} UQ metric files.")
    print(metric_msg)


if __name__ == "__main__":
    main()
