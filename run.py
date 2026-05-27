import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import (
    load_gsm8k,
    load_mbppplus,
    load_medqa
)
from methods.mas import MASMethod, MAS_PROMPT_CHOICES, resolve_mas_prompt
from methods.uncertainty_quantification import (
    apply_posthoc_uncertainty,
    append_preds_jsonl,
    build_uncertainty_metrics_summary,
    export_metrics_json,
    export_preds_jsonl,
    load_preds_jsonl,
    print_posthoc_sample_reports,
    print_uncertainty_metrics_summary,
)
from mas_graph import build_mas_graph, export_mas_graph_artifacts
from models import ModelWrapper
from utils import auto_device, set_seed
import time


def build_log_root() -> Path:
    return Path(__file__).resolve().parent / "outputs"


LOCAL_UNCERTAINTY_MODE_CHOICES = ["Verb", "MSP"]
MODEL_CHOICES = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "google/gemma-3-12b-it",
]
TASK_CHOICES = ["gsm8k", "medqa", "mbppplus"]


def _normalize_one_local_uncertainty_mode(value: str | None) -> str:
    if value is None:
        return "Verb"
    normalized = str(value).strip()
    if not normalized:
        return "Verb"
    for choice in LOCAL_UNCERTAINTY_MODE_CHOICES:
        if normalized.lower() == choice.lower():
            return choice
    raise ValueError(f"Unsupported uncertainty mode: {value}")


def _normalize_local_uncertainty_modes(values) -> List[str]:
    if values is None:
        return ["Verb"]
    mode = _normalize_one_local_uncertainty_mode(values)
    return [mode]


def _uses_verb(args: argparse.Namespace) -> bool:
    return "Verb" in getattr(args, "local_uncertainty_modes", ["Verb"])


def _local_uncertainty_mode_slug(values) -> str:
    if isinstance(values, list):
        modes = list(values)
    else:
        modes = _normalize_local_uncertainty_modes(values)
    return "_" + "_".join(mode.lower().replace(" ", "_") for mode in modes)


def _local_uncertainty_mode_dirname(values) -> str:
    if isinstance(values, list):
        modes = list(values)
    else:
        modes = _normalize_local_uncertainty_modes(values)
    return "_".join(mode.replace(" ", "_") for mode in modes)


def _model_output_dir(args: argparse.Namespace) -> Path:
    root = build_log_root() / _local_uncertainty_mode_dirname(args.local_uncertainty_modes)
    model_parts = [part for part in str(args.model_name).split("/") if part]
    if not model_parts:
        return root / "unknown_model"
    for part in model_parts:
        root = root / part
    return root


def _uq_output_dir(args: argparse.Namespace) -> Path:
    return _model_output_dir(args) / "uq"


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def _build_cache_stem(args: argparse.Namespace) -> str:
    parts = [
        "mas",
        args.task,
        args.mas_topology,
        f"nodes{args.mas_node_num}",
        resolve_mas_prompt(args) if resolve_mas_prompt(args) != "norole" else None,
        f"seed{args.seed}",
        f"n{args.max_samples}",
    ]
    return "_".join(str(part) for part in parts if part)


def _build_uq_cache_stem(args: argparse.Namespace) -> str:
    return _build_cache_stem(args)


def _build_graph_cache_stem(args: argparse.Namespace) -> str:
    parts = [
        args.method,
        args.mas_topology,
        f"nodes{args.mas_node_num}",
        f"seed{args.seed}",
    ]
    return "_".join(str(part) for part in parts if part)


def _find_available_path(base_dir: Path, stem: str, suffix: str) -> Path:
    candidate = base_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        alt = candidate.with_name(f"{stem}_{idx}{suffix}")
        if not alt.exists():
            return alt
        idx += 1


def build_default_raw_preds_path(args: argparse.Namespace) -> Path:
    base_dir = _model_output_dir(args) / "raw"
    return base_dir / f"raw_preds_{_build_cache_stem(args)}.jsonl"


def build_default_uq_preds_path(args: argparse.Namespace) -> Path:
    base_dir = _uq_output_dir(args)
    return _find_available_path(base_dir, f"uq_preds_{_build_uq_cache_stem(args)}", ".jsonl")


def build_default_uq_metrics_path(args: argparse.Namespace) -> Path:
    base_dir = _uq_output_dir(args)
    return _find_available_path(base_dir, f"uq_metrics_{_build_uq_cache_stem(args)}", ".json")

# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict], int]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, [], 0
    current_batch = batch[:remaining]
    results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    batch_correct = 0
    for offset, res in enumerate(results):
        batch_correct += 1 if res.get("correct", False) else 0
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            incoming_agents = a.get("incoming_agents", []) or []
            outgoing_agents = a.get("outgoing_agents", []) or []
            agent_header = f"----- Agent: {name} -----"
            print(agent_header)
            if args.method == "mas":
                print(f"[Incoming] {', '.join(incoming_agents) if incoming_agents else '(none)'}")
                print(f"[Outgoing] {', '.join(outgoing_agents) if outgoing_agents else '(none)'}")
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            print("[To Tokenize]")
            print(agent_input)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        if res.get("skipped"):
            print(f"Result: SKIPPED | Reason={res.get('skip_reason')}")
            continue
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, results, batch_correct


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["mas"], default="mas",
                        help="Experiment method. This release implements PropUQ-MAS.")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=MODEL_CHOICES,
        default="",
        help="HF model name to use for experiments.",
    )
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=TASK_CHOICES, default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--mas_topology", type=str, choices=["sequential", "hierarchical", "decentralized"], default="sequential",
                        help="Topology of the MAS graph.")
    parser.add_argument("--mas_node_num", type=int, default=4,
                        help="Number of nodes/agents in the MAS graph.")
    parser.add_argument(
        "--mas_prompt",
        type=str,
        choices=list(MAS_PROMPT_CHOICES),
        default="norole",
        help="MAS prompt style. 'norole' keeps the generic-agent prompt; 'role' enables role-specific prompts for sequential/hierarchical when mas_node_num=4.",
    )
    parser.add_argument(
        "--render_mas_graph_svg",
        action="store_true",
        help="Render MAS graph SVG/DOT with Graphviz. JSON/TXT are always exported.",
    )

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=16, help="Batch size for generation")
    parser.add_argument(
        "--local_uncertainty_mode",
        type=str,
        choices=LOCAL_UNCERTAINTY_MODE_CHOICES,
        default=None,
        help="Choose exactly one of Verb or MSP.",
    )
    parser.add_argument("--mas_context_length", type=int, default=-1, help="MAS context length limit; <=0 means no truncation")
    parser.add_argument(
        "--mas_inter_agent_think",
        type=str,
        choices=["strip", "pass"],
        default="strip",
        help="Whether to strip <think>...</think> from inter-agent context in mas. 'strip' avoids passing hidden reasoning noise.",
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--force_generation",
        action="store_true",
        help="Ignore an existing raw prediction cache and rerun model generation.",
    )
    parser.add_argument(
        "--force_uq",
        action="store_true",
        help="Overwrite existing posthoc UQ outputs instead of generating a new suffixed file.",
    )
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Run generation and write raw_preds only; skip posthoc UQ replay and UQ output files.",
    )
    parser.add_argument(
        "--prepare_mas_graph_only",
        action="store_true",
        help="Prepare MAS graph artifacts only and exit before loading models or running experiments.",
    )

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    args = parser.parse_args()
    if not args.prepare_mas_graph_only and not args.model_name:
        parser.error("--model_name is required unless --prepare_mas_graph_only is set.")
    try:
        args.local_uncertainty_modes = _normalize_local_uncertainty_modes(args.local_uncertainty_mode)
    except ValueError as exc:
        parser.error(str(exc))
    args.local_uncertainty_mode = args.local_uncertainty_mode or "Verb"
    args.enable_uncertainty_explanation = _uses_verb(args)
    args.enable_peer_influence = True
    args.resolved_mas_prompt = resolve_mas_prompt(args)
    raw_preds_path = build_default_raw_preds_path(args)
    if args.mas_prompt != "norole":
        print(
            "MAS prompt requested: "
            f"{args.mas_prompt} | resolved: {args.resolved_mas_prompt}"
        )
    mas_graph = build_mas_graph(
        args.mas_topology,
        args.mas_node_num,
    )
    graph_output_dir = build_log_root() / "mas_graphs" / _build_graph_cache_stem(args)
    graph_artifacts = export_mas_graph_artifacts(
        mas_graph,
        graph_output_dir,
        render_svg=args.render_mas_graph_svg,
    )
    print(f"MAS graph: {mas_graph.edge_strings()}")
    print(f"MAS graph density: {mas_graph.density():.6f}")
    if graph_artifacts["reused_existing"]:
        print(f"Reused MAS graph artifacts from: {graph_output_dir}")
    if graph_artifacts["svg"] is not None:
        print(f"MAS graph SVG saved to: {graph_artifacts['svg']}")
        print(f"MAS graph DOT saved to: {graph_artifacts['dot']}")
    else:
        print(f"Warning: skipped MAS graph SVG export. {graph_artifacts['svg_warning']}")
    print(f"MAS message-passing info saved to: {graph_artifacts['text']}")
    print(f"MAS graph JSON saved to: {graph_artifacts['json']}")
    if args.prepare_mas_graph_only:
        print("Prepared MAS graph artifacts only. Exiting before experiment execution.")
        return

    generated_this_run = False
    start_time = time.time()
    if raw_preds_path.exists() and not args.force_generation:
        preds = load_preds_jsonl(raw_preds_path)
        total_time = time.time() - start_time
        print(f"Raw preds cache hit: {raw_preds_path}")
    else:
        set_seed(args.seed)
        device = auto_device(args.device)
        model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)

        common_kwargs = dict(
            temperature=args.temperature,
            top_p=args.top_p,
        )

        method = MASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )

        processed = 0
        correct = 0
        batch: List[Dict] = []
        if raw_preds_path.exists():
            raw_preds_path.unlink()

        if args.task == "gsm8k":
            dataset_iter = load_gsm8k(split=args.split)
        elif args.task == "mbppplus":
            dataset_iter = load_mbppplus(split='test')
        elif args.task == "medqa":
            dataset_iter = load_medqa(split='test')
        else:
            raise ValueError(f'no {args.task} support')

        if args.max_samples == -1:
            dataset_iter = list(dataset_iter)
            args.max_samples = len(dataset_iter)

        progress = tqdm(total=args.max_samples)

        for item in dataset_iter:
            if processed >= args.max_samples:
                break
            batch.append(item)
            if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
                processed, batch_results, batch_correct = process_batch(
                    method,
                    batch,
                    processed,
                    progress,
                    args.max_samples,
                    args,
                )
                append_preds_jsonl(batch_results, raw_preds_path)
                correct += batch_correct
                batch = []
                if processed >= args.max_samples:
                    break

        if batch and processed < args.max_samples:
            processed, batch_results, batch_correct = process_batch(
                method,
                batch,
                processed,
                progress,
                max_samples=args.max_samples,
                args=args,
            )
            append_preds_jsonl(batch_results, raw_preds_path)
            correct += batch_correct
        progress.close()

        generated_this_run = True
        total_time = time.time() - start_time
        print(f"Raw preds cache written to: {raw_preds_path}")
        preds = None

    if args.max_samples == -1 and preds is not None:
        args.max_samples = len(preds)

    if preds is not None:
        acc, correct = evaluate(preds)
    else:
        acc = correct / args.max_samples if args.max_samples > 0 else 0.0

    uq_metrics = None
    should_run_posthoc_uq = (
        not args.generate_only
    )

    if should_run_posthoc_uq:
        if preds is None:
            preds = load_preds_jsonl(raw_preds_path)
        uq_preds_path = build_default_uq_preds_path(args)
        uq_metrics_path = build_default_uq_metrics_path(args)
        if args.force_uq:
            uq_base_dir = _uq_output_dir(args)
            uq_preds_path = uq_base_dir / f"uq_preds_{_build_uq_cache_stem(args)}.jsonl"
            uq_metrics_path = uq_base_dir / f"uq_metrics_{_build_uq_cache_stem(args)}.json"

        preds = apply_posthoc_uncertainty(preds, method=args.method)
        print_posthoc_sample_reports(preds, args)
        uq_metrics = build_uncertainty_metrics_summary(preds)
        export_preds_jsonl(preds, uq_preds_path)
        export_metrics_json(
            uq_metrics,
            uq_metrics_path,
            metadata={
                "method": "mas",
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "task": args.task,
                "mas_topology": args.mas_topology,
                "mas_node_num": args.mas_node_num,
                "mas_prompt": getattr(args, "mas_prompt", "norole"),
                "resolved_mas_prompt": getattr(args, "resolved_mas_prompt", "norole"),
                "local_uncertainty_mode": args.local_uncertainty_modes[0] if len(args.local_uncertainty_modes) == 1 else None,
                "local_uncertainty_modes": args.local_uncertainty_modes,
                "accuracy": acc,
                "correct": correct,
                "total_time_sec": round(total_time, 4),
                "time_per_sample_sec": round(total_time / args.max_samples, 4),
                "raw_preds_path": str(raw_preds_path),
                "generated_this_run": generated_this_run,
            },
        )
        print(f"UQ preds JSONL exported to: {uq_preds_path}")
        print(f"UQ metrics JSON exported to: {uq_metrics_path}")
    elif args.generate_only:
        print("Generate-only mode: skipped posthoc UQ replay and UQ file export.")

    # Load results in JSON format
    print(
        json.dumps(
            {
                "method": args.method,
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "accuracy": acc,
                "correct": correct,
                "total_time_sec": round(total_time,4),
                "time_per_sample_sec": round(total_time / args.max_samples, 4),
            },
            ensure_ascii=False,
        )
    )

    if uq_metrics is not None:
        print_uncertainty_metrics_summary(uq_metrics)



if __name__ == "__main__":
    main()
