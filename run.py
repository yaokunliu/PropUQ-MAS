import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.single_agent import SingleAgentMethod
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


UNCERTAINTY_MODE_CHOICES = ["ASK4CONF", "MSP", "NLL"]
LOGIT_UQ_BUNDLE = ["MSP", "NLL"]
UNCERTAINTY_MODE_BUNDLE_NAME = "LOGIT_UQ_BUNDLE"
UQ_ADOPTION_MODE_CHOICES = ["original", "all_one"]


def _normalize_one_uncertainty_mode(value: str | None) -> str:
    if value is None:
        return "ASK4CONF"
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "continuous":
        return "ASK4CONF"
    if normalized == UNCERTAINTY_MODE_BUNDLE_NAME:
        return UNCERTAINTY_MODE_BUNDLE_NAME
    for choice in UNCERTAINTY_MODE_CHOICES:
        if normalized.lower() == choice.lower():
            return choice
    raise ValueError(f"Unsupported uncertainty mode: {value}")


def _normalize_uncertainty_modes(values) -> List[str]:
    if values is None:
        return ["ASK4CONF"]
    mode = _normalize_one_uncertainty_mode(values)
    if mode == UNCERTAINTY_MODE_BUNDLE_NAME:
        return list(LOGIT_UQ_BUNDLE)
    return [mode]


def _uses_ask4conf(args: argparse.Namespace) -> bool:
    return "ASK4CONF" in getattr(args, "uncertainty_modes", ["ASK4CONF"])


def _uncertainty_mode_slug(values) -> str:
    if isinstance(values, list):
        modes = list(values)
    else:
        modes = _normalize_uncertainty_modes(values)
    return "_" + "_".join(mode.lower().replace(" ", "_") for mode in modes)


def _uncertainty_mode_dirname(values) -> str:
    if isinstance(values, list):
        modes = list(values)
    else:
        modes = _normalize_uncertainty_modes(values)
    return "_".join(mode.replace(" ", "_") for mode in modes)


def _model_output_dir(args: argparse.Namespace) -> Path:
    root = build_log_root() / _uncertainty_mode_dirname(args.uncertainty_modes)
    model_parts = [part for part in str(args.model_name).split("/") if part]
    if not model_parts:
        return root / "unknown_model"
    for part in model_parts:
        root = root / part
    return root


def _normalize_uq_adoption_mode(value: str | None) -> str:
    if value is None:
        return "original"
    normalized = str(value).strip().lower()
    if normalized in {"", "original"}:
        return "original"
    if normalized in {"all_one", "all1"}:
        return "all_one"
    raise ValueError(f"Unsupported uq adoption mode: {value}")


def _uq_output_dir(args: argparse.Namespace) -> Path:
    base_dir = _model_output_dir(args) / "uq"
    if getattr(args, "uq_adoption_mode", "original") == "original":
        return base_dir
    return base_dir / args.uq_adoption_mode


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def _build_cache_stem(args: argparse.Namespace) -> str:
    parts = [
        args.method,
        args.task,
        args.mas_topology if args.method == "mas" else None,
        f"nodes{args.mas_node_num}" if args.method == "mas" else None,
        resolve_mas_prompt(args) if args.method == "mas" and resolve_mas_prompt(args) != "norole" else None,
        f"edges{args.random_edge_count}" if args.method == "mas" and args.mas_topology == "random" and args.random_edge_count is not None else None,
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
        f"edges{args.random_edge_count}" if args.mas_topology == "random" and args.random_edge_count is not None else None,
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
    parser.add_argument("--method", choices=["single_agent", "mas"], required=True,
                        help="Which method to run: 'single_agent' or 'mas'.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="HF model name to use for experiments (e.g. 'Qwen/Qwen3-14B', 'google/gemma-3-12b-it', 'mistralai/Ministral-3-8B-Instruct-2512').",
    )
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--mas_topology", type=str, choices=["chain", "star_convergent", "star_divergent", "tree", "net", "random"], default="chain",
                        help="Topology of the MAS graph.")
    parser.add_argument("--mas_node_num", type=int, default=4,
                        help="Number of nodes/agents in the MAS graph.")
    parser.add_argument(
        "--mas_prompt",
        type=str,
        choices=list(MAS_PROMPT_CHOICES),
        default="norole",
        help="MAS prompt style. 'norole' keeps the original generic-agent prompt; 'role' enables role-specific prompts for chain/star_convergent when mas_node_num=4.",
    )
    parser.add_argument(
        "--random_edge_count",
        type=int,
        default=None,
        help="For mas_topology=random only, fix the number of edges. Must satisfy n-1 <= edge_count <= n*(n-1)/2. If omitted, edge count is sampled randomly.",
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
        "--uncertainty_mode",
        type=str,
        choices=UNCERTAINTY_MODE_CHOICES + [UNCERTAINTY_MODE_BUNDLE_NAME],
        default=None,
        help="Choose exactly one of ASK4CONF, MSP, NLL, or "
             f"{UNCERTAINTY_MODE_BUNDLE_NAME}. {UNCERTAINTY_MODE_BUNDLE_NAME} runs both MSP and NLL.",
    )
    parser.add_argument(
        "--uq_adoption_mode",
        type=str,
        choices=UQ_ADOPTION_MODE_CHOICES,
        default="original",
        help="Posthoc MAS adoption mode for formula/hazard/system UQ. "
             "'original' uses recorded message adoption; 'all_one' forces every incoming adoption score to 1.",
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
        args.uncertainty_modes = _normalize_uncertainty_modes(args.uncertainty_mode)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.uq_adoption_mode = _normalize_uq_adoption_mode(args.uq_adoption_mode)
    except ValueError as exc:
        parser.error(str(exc))
    args.uncertainty_mode = args.uncertainty_mode or "ASK4CONF"
    args.enable_uncertainty_explanation = _uses_ask4conf(args)
    args.enable_peer_influence = True
    args.resolved_mas_prompt = resolve_mas_prompt(args) if args.method == "mas" else "norole"
    raw_preds_path = build_default_raw_preds_path(args)
    if args.method == "mas":
        if args.mas_prompt != "norole":
            print(
                "MAS prompt requested: "
                f"{args.mas_prompt} | resolved: {args.resolved_mas_prompt}"
            )
        mas_graph = build_mas_graph(
            args.mas_topology,
            args.mas_node_num,
            seed=args.seed,
            random_edge_count=args.random_edge_count,
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

        if args.method == "single_agent":
            method = SingleAgentMethod(
                model,
                max_new_tokens=args.max_new_tokens,
                **common_kwargs,
                generate_bs=args.generate_bs,
                use_vllm=args.use_vllm,
                args=args
            )
        elif args.method == "mas":
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
        elif args.task == "aime2024":
            dataset_iter = load_aime2024(split="train")
        elif args.task == "aime2025":
            dataset_iter = load_aime2025(split='train')
        elif args.task == "gpqa":
            dataset_iter = load_gpqa_diamond(split='test')
        elif args.task == "arc_easy":
            dataset_iter = load_arc_easy(split='test')
        elif args.task == "arc_challenge":
            dataset_iter = load_arc_challenge(split='test')
        elif args.task == "mbppplus":
            dataset_iter = load_mbppplus(split='test')
        elif args.task == "humanevalplus":
            dataset_iter = load_humanevalplus(split='test')
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

        preds = apply_posthoc_uncertainty(preds, method=args.method, adoption_mode=args.uq_adoption_mode)
        print_posthoc_sample_reports(preds, args)
        uq_metrics = build_uncertainty_metrics_summary(preds)
        export_preds_jsonl(preds, uq_preds_path)
        export_metrics_json(
            uq_metrics,
            uq_metrics_path,
            metadata={
                "method": args.method,
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "task": args.task,
                "mas_topology": args.mas_topology if args.method == "mas" else None,
                "mas_node_num": args.mas_node_num if args.method == "mas" else None,
                "mas_prompt": getattr(args, "mas_prompt", "norole") if args.method == "mas" else None,
                "resolved_mas_prompt": getattr(args, "resolved_mas_prompt", "norole") if args.method == "mas" else None,
                "uncertainty_mode": args.uncertainty_modes[0] if len(args.uncertainty_modes) == 1 else None,
                "uncertainty_modes": args.uncertainty_modes,
                "uq_adoption_mode": args.uq_adoption_mode,
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
