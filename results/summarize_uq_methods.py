#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


DETAIL_METRIC_ORDER = [
    "Uncertainty-Final",
    "Uncertainty-Max",
    "Uncertainty-Mean",
    "Formula-Uncertainty-Final",
    "Formula-Uncertainty-Max",
    "Formula-Uncertainty-Mean",
    "Hazard-h-Final",
    "Hazard-h-Max",
    "Hazard-h-Mean",
    "System-Uncertainty-Final",
]

METRIC_GROUPS = {
    "mean": ("Mean", ("-Mean", "System-Uncertainty-Final")),
    "max": ("Max", ("-Max", "System-Uncertainty-Final")),
    "final": ("Final", ("-Final", "System-Uncertainty-Final")),
}

FILE_RE = re.compile(
    r"^uq_metrics_"
    r"(?P<method>[^_]+)_"
    r"(?P<task>[^_]+)_"
    r"(?P<topology>.+?)"
    r"_nodes(?P<nodes>\d+)"
    r"(?:_edges(?P<edges>\d+))?"
    r"_seed(?P<seed>-?\d+)"
    r"_n(?P<max_samples>-?\d+)"
    r"(?:_uq(?P<uncertainty_mode_slug>[a-z0-9_]+)|_uq_(?P<uncertainty_mode_slug_alt>[a-z0-9_]+))?"
    r"(?:_adopt(?P<adoption_mode_slug>[a-z0-9_]+))?"
    r"\.json$"
)


@dataclass
class MetricRecord:
    method: str
    model: str
    task: str
    topology: str
    node_num: int | None
    random_edge_count: int | None
    seed: int | None
    max_samples: int | None
    uncertainty_mode: str
    uq_adoption_mode: str
    accuracy: float | None
    correct: int | None
    total_time_sec: float | None
    time_per_sample_sec: float | None
    metrics: dict[str, dict[str, float | int]]
    best_auroc_metric: str | None
    best_auroc: float | None
    best_ece_metric: str | None
    best_ece: float | None
    best_brier_metric: str | None
    best_brier: float | None
    metrics_path: Path
    raw_preds_path: str | None


def build_cli() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Summarize MAS_UQ UQ metric JSON files into markdown and csv tables."
    )
    parser.add_argument("--uq-dir", type=Path, default=root / "outputs")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    return parser


def parse_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def normalize_uncertainty_mode(value: object) -> str:
    if value is None:
        return "ASK4CONF"
    text = str(value).strip()
    if not text or text.lower() == "continuous":
        return "ASK4CONF"
    if "__" in text:
        return text
    if text.startswith("_"):
        text = text[1:]
    mapping = {
        "ask4conf": "ASK4CONF",
        "msp": "MSP",
        "nll": "NLL",
    }
    return mapping.get(text.lower(), text)


def normalize_uq_adoption_mode(value: object) -> str:
    if value is None:
        return "original"
    text = str(value).strip().lower()
    if text in {"", "original"}:
        return "original"
    if text in {"all_one", "all1", "adoptall1"}:
        return "all_one"
    return text


def select_metric(
    metrics: dict[str, dict[str, float | int]],
    metric_name: str,
    *,
    higher_is_better: bool,
    allowed_metric_names: set[str] | None = None,
) -> tuple[str | None, float | None]:
    best_name = None
    best_value = None
    for name, values in ordered_metric_items(metrics):
        if allowed_metric_names is not None and name not in allowed_metric_names:
            continue
        value = parse_float(values.get(metric_name))
        if value is None:
            continue
        if best_value is None:
            best_name = name
            best_value = value
            continue
        if higher_is_better and value > best_value:
            best_name = name
            best_value = value
        elif not higher_is_better and value < best_value:
            best_name = name
            best_value = value
        elif abs(value - best_value) < 1e-12:
            best_name = name
            best_value = value
    return best_name, best_value


def grouped_metric_names(metrics: dict[str, dict[str, float | int]], group_key: str) -> set[str]:
    _, suffixes = METRIC_GROUPS[group_key]
    return {
        metric_name
        for metric_name in metrics
        if any(metric_name.endswith(suffix) for suffix in suffixes)
    }


def ordered_metric_items(metrics: dict[str, dict[str, float | int]]) -> list[tuple[str, dict[str, float | int]]]:
    ordered: list[tuple[str, dict[str, float | int]]] = []
    seen: set[str] = set()
    for metric_name in DETAIL_METRIC_ORDER:
        values = metrics.get(metric_name)
        if values is None:
            continue
        ordered.append((metric_name, values))
        seen.add(metric_name)
    for metric_name in sorted(metrics):
        if metric_name in seen:
            continue
        ordered.append((metric_name, metrics[metric_name]))
    return ordered


def parse_filename(path: Path) -> dict[str, int | str | None]:
    match = FILE_RE.match(path.name)
    if not match:
        return {}
    groups = match.groupdict()
    return {
        "method": groups["method"],
        "task": groups["task"],
        "topology": groups["topology"],
        "node_num": parse_int(groups["nodes"]),
        "random_edge_count": parse_int(groups["edges"]),
        "seed": parse_int(groups["seed"]),
        "max_samples": parse_int(groups["max_samples"]),
        "uncertainty_mode": normalize_uncertainty_mode(
            groups.get("uncertainty_mode_slug_alt") or groups.get("uncertainty_mode_slug")
        ),
        "uq_adoption_mode": normalize_uq_adoption_mode(groups.get("adoption_mode_slug")),
    }


def infer_uq_adoption_mode_from_path(path: Path) -> str | None:
    parts = path.parts
    if "all_one" in parts:
        return "all_one"
    if "original" in parts:
        return "original"
    return None


def is_standard_uq_metrics_path(path: Path) -> bool:
    parts = path.parts
    if "uq_formula_adoption_ablation" in parts:
        return False
    parent_names = {parent.name for parent in path.parents}
    if "uq" not in parent_names:
        return False
    return True


def load_metric_records(uq_dir: Path, *, include_ablation: bool = False) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for path in sorted(uq_dir.rglob("uq_metrics_*.json")):
        if not include_ablation and not is_standard_uq_metrics_path(path):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        metrics_payload = payload.get("metrics", {})
        parsed = parse_filename(path)
        path_adoption_mode = infer_uq_adoption_mode_from_path(path)

        method = metadata.get("method") or parsed.get("method") or ""
        task = metadata.get("task") or parsed.get("task") or ""
        topology = metadata.get("mas_topology") or parsed.get("topology") or ""
        node_num = parse_int(metadata.get("mas_node_num"))
        if node_num is None:
            node_num = parse_int(parsed.get("node_num"))
        seed = parse_int(metadata.get("seed"))
        if seed is None:
            seed = parse_int(parsed.get("seed"))
        max_samples = parse_int(metadata.get("max_samples"))
        if max_samples is None:
            max_samples = parse_int(parsed.get("max_samples"))
        random_edge_count = parse_int(parsed.get("random_edge_count"))

        metrics_by_method = metrics_payload.get("metrics_by_uq_method")
        if isinstance(metrics_by_method, dict) and metrics_by_method:
            items = [(normalize_uncertainty_mode(mode), values) for mode, values in metrics_by_method.items()]
        else:
            items = [(
                normalize_uncertainty_mode(metadata.get("uncertainty_mode") or parsed.get("uncertainty_mode")),
                metrics_payload,
            )]

        for uncertainty_mode, metrics in items:
            best_auroc_metric, best_auroc = select_metric(metrics, "auroc", higher_is_better=True)
            best_ece_metric, best_ece = select_metric(metrics, "ece", higher_is_better=False)
            best_brier_metric, best_brier = select_metric(metrics, "brier", higher_is_better=False)

            records.append(
                MetricRecord(
                    method=method,
                    model=str(metadata.get("model", "")),
                    task=task,
                    topology=topology,
                    node_num=node_num,
                    random_edge_count=random_edge_count,
                    seed=seed,
                    max_samples=max_samples,
                    uncertainty_mode=uncertainty_mode,
                    uq_adoption_mode=normalize_uq_adoption_mode(
                        path_adoption_mode or metadata.get("uq_adoption_mode") or parsed.get("uq_adoption_mode")
                    ),
                    accuracy=parse_float(metadata.get("accuracy")),
                    correct=parse_int(metadata.get("correct")),
                    total_time_sec=parse_float(metadata.get("total_time_sec")),
                    time_per_sample_sec=parse_float(metadata.get("time_per_sample_sec")),
                    metrics=metrics,
                    best_auroc_metric=best_auroc_metric,
                    best_auroc=best_auroc,
                    best_ece_metric=best_ece_metric,
                    best_ece=best_ece,
                    best_brier_metric=best_brier_metric,
                    best_brier=best_brier,
                    metrics_path=path,
                    raw_preds_path=metadata.get("raw_preds_path"),
                )
            )
    return records


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv_summary(records: list[MetricRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "model",
                "task",
                "topology",
                "node_num",
                "random_edge_count",
                "seed",
                "max_samples",
                "uncertainty_mode",
                "uq_adoption_mode",
                "accuracy",
                "correct",
                "best_auroc_metric",
                "best_auroc",
                "best_ece_metric",
                "best_ece",
                "best_brier_metric",
                "best_brier",
                "time_per_sample_sec",
                "metrics_path",
                "raw_preds_path",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.method,
                    record.model,
                    record.task,
                    record.topology,
                    record.node_num,
                    record.random_edge_count,
                    record.seed,
                    record.max_samples,
                    record.uncertainty_mode,
                    record.uq_adoption_mode,
                    record.accuracy,
                    record.correct,
                    record.best_auroc_metric,
                    record.best_auroc,
                    record.best_ece_metric,
                    record.best_ece,
                    record.best_brier_metric,
                    record.best_brier,
                    record.time_per_sample_sec,
                    str(record.metrics_path),
                    record.raw_preds_path or "",
                ]
            )


def build_summary_markdown(records: list[MetricRecord]) -> str:
    def is_main_setting(record: MetricRecord) -> bool:
        if record.node_num != 5:
            return False
        if record.topology == "random":
            return record.random_edge_count == 7
        return True

    def is_scaling_setting(record: MetricRecord) -> bool:
        if record.uncertainty_mode != "ASK4CONF":
            return False
        if record.topology == "chain":
            return record.node_num in {2, 4, 6, 8, 10}
        if record.topology == "random":
            return record.node_num == 5 and record.random_edge_count in {4, 5, 6, 7, 8, 9, 10}
        if record.topology in {"star_convergent", "star_divergent"}:
            return record.node_num in {4, 6, 8, 10}
        return False

    def render_table(section_records: list[MetricRecord]) -> list[str]:
        table_lines = [
            "| method | uq_mode | adoption | model | task | topology | node_num | edge_count | acc | best_auroc | auroc_metric | best_ece | ece_metric | best_brier | brier_metric |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |",
        ]
        for record in section_records:
            table_lines.append(
                "| "
                + " | ".join(
                    [
                        record.method or "NA",
                        record.uncertainty_mode or "ASK4CONF",
                        record.uq_adoption_mode or "original",
                        record.model or "NA",
                        record.task or "NA",
                        record.topology or "NA",
                        str(record.node_num) if record.node_num is not None else "NA",
                        str(record.random_edge_count) if record.random_edge_count is not None else "NA",
                        format_float(record.accuracy),
                        format_float(record.best_auroc),
                        record.best_auroc_metric or "NA",
                        format_float(record.best_ece),
                        record.best_ece_metric or "NA",
                        format_float(record.best_brier),
                        record.best_brier_metric or "NA",
                    ]
                )
                + " |"
            )
        if len(table_lines) == 2:
            table_lines.append("| NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
        return table_lines

    main_records = [record for record in records if is_main_setting(record)]
    scaling_records = [record for record in records if is_scaling_setting(record)]

    lines = [
        "# UQ Summary",
        "",
        f"Total runs: {len(records)}",
        "",
        "## Main Setting",
        "",
        f"Runs in section: {len(main_records)}",
        "",
        *render_table(main_records),
        "",
        "## Scaling",
        "",
        f"Runs in section: {len(scaling_records)}",
        "",
        *render_table(scaling_records),
    ]
    return "\n".join(lines) + "\n"


def build_grouped_summary_markdown(records: list[MetricRecord]) -> str:
    def is_main_setting(record: MetricRecord) -> bool:
        if record.node_num != 5:
            return False
        if record.topology == "random":
            return record.random_edge_count == 7
        return True

    def is_scaling_setting(record: MetricRecord) -> bool:
        if record.uncertainty_mode != "ASK4CONF":
            return False
        if record.topology == "chain":
            return record.node_num in {2, 4, 6, 8, 10}
        if record.topology == "random":
            return record.node_num == 5 and record.random_edge_count in {4, 5, 6, 7, 8, 9, 10}
        if record.topology in {"star_convergent", "star_divergent"}:
            return record.node_num in {4, 6, 8, 10}
        return False

    def render_table(section_records: list[MetricRecord], group_key: str) -> list[str]:
        table_lines = [
            "| method | uq_mode | adoption | model | task | topology | node_num | edge_count | acc | best_auroc | auroc_metric | best_ece | ece_metric | best_brier | brier_metric |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |",
        ]
        for record in section_records:
            allowed_metric_names = grouped_metric_names(record.metrics, group_key)
            best_auroc_metric, best_auroc = select_metric(
                record.metrics,
                "auroc",
                higher_is_better=True,
                allowed_metric_names=allowed_metric_names,
            )
            best_ece_metric, best_ece = select_metric(
                record.metrics,
                "ece",
                higher_is_better=False,
                allowed_metric_names=allowed_metric_names,
            )
            best_brier_metric, best_brier = select_metric(
                record.metrics,
                "brier",
                higher_is_better=False,
                allowed_metric_names=allowed_metric_names,
            )
            table_lines.append(
                "| "
                + " | ".join(
                    [
                        record.method or "NA",
                        record.uncertainty_mode or "ASK4CONF",
                        record.uq_adoption_mode or "original",
                        record.model or "NA",
                        record.task or "NA",
                        record.topology or "NA",
                        str(record.node_num) if record.node_num is not None else "NA",
                        str(record.random_edge_count) if record.random_edge_count is not None else "NA",
                        format_float(record.accuracy),
                        format_float(best_auroc),
                        best_auroc_metric or "NA",
                        format_float(best_ece),
                        best_ece_metric or "NA",
                        format_float(best_brier),
                        best_brier_metric or "NA",
                    ]
                )
                + " |"
            )
        if len(table_lines) == 2:
            table_lines.append("| NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
        return table_lines

    def render_section(title: str, section_records: list[MetricRecord]) -> list[str]:
        lines = [
            f"## {title}",
            "",
            f"Runs in section: {len(section_records)}",
            "",
        ]
        for group_key, (group_title, _) in METRIC_GROUPS.items():
            lines.extend(
                [
                    f"### {group_title}",
                    "",
                    *render_table(section_records, group_key),
                    "",
                ]
            )
        return lines

    main_records = [record for record in records if is_main_setting(record)]
    scaling_records = [record for record in records if is_scaling_setting(record)]

    lines = [
        "# UQ Summary By Metric Group",
        "",
        "Each group includes its suffix-matched metrics plus System-Uncertainty-Final.",
        "",
        f"Total runs: {len(records)}",
        "",
        *render_section("Main Setting", main_records),
        *render_section("Scaling", scaling_records),
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_detail_markdown(records: list[MetricRecord]) -> str:
    lines = ["# UQ Metric Details", ""]
    for record in records:
        title = (
            f"{record.method} | {record.uncertainty_mode or 'ASK4CONF'} | {record.model} | {record.task} | "
            f"{record.topology} | nodes={record.node_num}"
        )
        if record.random_edge_count is not None:
            title += f" | edges={record.random_edge_count}"
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- accuracy: {format_float(record.accuracy)}")
        lines.append(f"- correct: {record.correct if record.correct is not None else 'NA'}")
        lines.append(f"- seed: {record.seed if record.seed is not None else 'NA'}")
        lines.append(f"- max_samples: {record.max_samples if record.max_samples is not None else 'NA'}")
        lines.append(f"- uncertainty_mode: {record.uncertainty_mode or 'NA'}")
        lines.append(f"- uq_adoption_mode: {record.uq_adoption_mode or 'original'}")
        lines.append(f"- best_auroc: {format_float(record.best_auroc)} ({record.best_auroc_metric or 'NA'})")
        lines.append(f"- best_ece: {format_float(record.best_ece)} ({record.best_ece_metric or 'NA'})")
        lines.append(f"- best_brier: {format_float(record.best_brier)} ({record.best_brier_metric or 'NA'})")
        lines.append(f"- metrics_path: {record.metrics_path}")
        if record.raw_preds_path:
            lines.append(f"- raw_preds_path: {record.raw_preds_path}")
        lines.append("")
        lines.append("| metric | n_with_uncertainty | auroc | ece | brier |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for metric_name, values in ordered_metric_items(record.metrics):
            lines.append(
                "| "
                + " | ".join(
                    [
                        metric_name,
                        str(values.get("n_with_uncertainty", "NA")),
                        format_float(parse_float(values.get("auroc"))),
                        format_float(parse_float(values.get("ece"))),
                        format_float(parse_float(values.get("brier"))),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    records = load_metric_records(args.uq_dir)
    if not records:
        raise SystemExit(f"No uq_metrics_*.json files found under {args.uq_dir}")

    grouped: dict[str, list[MetricRecord]] = {}
    for record in records:
        grouped.setdefault(record.uq_adoption_mode or "original", []).append(record)

    for uq_adoption_mode, mode_records in sorted(grouped.items()):
        mode_records.sort(
            key=lambda r: (
                r.task,
                r.uncertainty_mode,
                r.model,
                r.topology,
                r.node_num if r.node_num is not None else -1,
                r.random_edge_count if r.random_edge_count is not None else -1,
                r.seed if r.seed is not None else -1,
                r.metrics_path.name,
            )
        )

        summary_md = build_summary_markdown(mode_records)
        grouped_summary_md = build_grouped_summary_markdown(mode_records)
        detail_md = build_detail_markdown(mode_records)

        mode_output_dir = args.output_dir / uq_adoption_mode
        summary_md_path = mode_output_dir / "uq_summary.md"
        grouped_summary_md_path = mode_output_dir / "uq_summary_by_group.md"
        summary_csv_path = mode_output_dir / "uq_summary.csv"
        detail_md_path = mode_output_dir / "uq_metric_details.md"

        write_text(summary_md_path, summary_md)
        write_text(grouped_summary_md_path, grouped_summary_md)
        write_text(detail_md_path, detail_md)
        write_csv_summary(mode_records, summary_csv_path)

        print(f"[{uq_adoption_mode}] Wrote markdown summary to: {summary_md_path}")
        print(f"[{uq_adoption_mode}] Wrote grouped markdown summary to: {grouped_summary_md_path}")
        print(f"[{uq_adoption_mode}] Wrote csv summary to: {summary_csv_path}")
        print(f"[{uq_adoption_mode}] Wrote metric details to: {detail_md_path}")


if __name__ == "__main__":
    main()
