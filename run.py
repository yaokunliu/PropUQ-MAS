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
from methods.mas import MASMethod
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
from models import ModelWrapper
from utils import auto_device, set_seed
import time


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def _build_cache_stem(args: argparse.Namespace) -> str:
    model_name = str(args.model_name).replace("/", "_")
    parts = [
        args.method,
        model_name,
        args.task,
        args.prompt,
        args.uncertainty_mode or "no_uq",
        args.split,
        f"seed{args.seed}",
        f"n{args.max_samples}",
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
    base_dir = Path(__file__).resolve().parent / "preds" / "raw"
    return base_dir / f"raw_preds_{_build_cache_stem(args)}.jsonl"


def build_default_uq_preds_path(args: argparse.Namespace) -> Path:
    base_dir = Path(__file__).resolve().parent / "preds" / "uq"
    return _find_available_path(base_dir, f"uq_preds_{_build_cache_stem(args)}", ".jsonl")


def build_default_uq_metrics_path(args: argparse.Namespace) -> Path:
    base_dir = Path(__file__).resolve().parent / "preds" / "uq"
    return _find_available_path(base_dir, f"uq_metrics_{_build_cache_stem(args)}", ".json")

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
        hierarchical_name_map = {
            "planner": "Math Agent",
            "critic": "Science Agent",
            "refiner": "Code Agent",
            "judger": "Task Summarizer",
        }
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            display_name = name
            if args.method == "mas" and args.prompt == "hierarchical":
                display_name = hierarchical_name_map.get(str(role).lower(), name)
                agent_header = f"----- Agent: {display_name} -----"
            else:
                agent_header = f"----- Agent: {display_name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            print("[To Tokenize]")
            print(agent_input)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
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
        required=True,
        help="HF model name to use for experiments (e.g. 'Qwen/Qwen3-14B', 'google/gemma-3-12b-it', 'mistralai/Ministral-3-8B-Instruct-2512').",
    )
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--mas_context_length", type=int, default=4096, help="MAS context length limit; <=0 means no truncation")
    parser.add_argument(
        "--mas_inter_agent_think",
        type=str,
        choices=["strip", "pass"],
        default="strip",
        help="Whether to strip <think>...</think> from inter-agent context in mas. 'strip' avoids passing hidden reasoning noise.",
    )
    parser.add_argument(
        "--uncertainty_mode",
        type=str,
        choices=["anchor", "continuous"],
        default=None,
        help="Enable self-report uncertainty analysis using anchor levels or continuous [0,1] scores.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw_preds_path",
        type=str,
        default=None,
        help="Where to read/write raw prediction cache JSONL.",
    )
    parser.add_argument(
        "--uq_preds_path",
        type=str,
        default=None,
        help="Where to write the posthoc-enriched UQ preds JSONL.",
    )
    parser.add_argument(
        "--uq_metrics_path",
        type=str,
        default=None,
        help="Where to write the posthoc uncertainty metrics JSON.",
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

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    args = parser.parse_args()
    args.enable_uncertainty_explanation = args.uncertainty_mode in {"anchor", "continuous"}
    args.enable_peer_influence = args.uncertainty_mode in {"anchor", "continuous"}
    if args.raw_preds_path is None:
        args.raw_preds_path = str(build_default_raw_preds_path(args))
    raw_preds_path = Path(args.raw_preds_path)

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
        args.uncertainty_mode in {"anchor", "continuous"} and not args.generate_only
    )

    if should_run_posthoc_uq:
        if preds is None:
            preds = load_preds_jsonl(raw_preds_path)
        if args.uq_preds_path is None:
            args.uq_preds_path = str(build_default_uq_preds_path(args))
        if args.uq_metrics_path is None:
            args.uq_metrics_path = str(build_default_uq_metrics_path(args))
        if args.force_uq:
            args.uq_preds_path = str(Path(__file__).resolve().parent / "preds" / "uq" / f"uq_preds_{_build_cache_stem(args)}.jsonl")
            args.uq_metrics_path = str(Path(__file__).resolve().parent / "preds" / "uq" / f"uq_metrics_{_build_cache_stem(args)}.json")

        preds = apply_posthoc_uncertainty(preds, method=args.method, prompt=args.prompt)
        print_posthoc_sample_reports(preds, args)
        uq_metrics = build_uncertainty_metrics_summary(preds)
        export_preds_jsonl(preds, Path(args.uq_preds_path))
        export_metrics_json(
            uq_metrics,
            Path(args.uq_metrics_path),
            metadata={
                "method": args.method,
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "task": args.task,
                "prompt": args.prompt,
                "uncertainty_mode": args.uncertainty_mode,
                "accuracy": acc,
                "correct": correct,
                "total_time_sec": round(total_time, 4),
                "time_per_sample_sec": round(total_time / args.max_samples, 4),
                "raw_preds_path": str(raw_preds_path),
                "generated_this_run": generated_this_run,
            },
        )
        print(f"UQ preds JSONL exported to: {args.uq_preds_path}")
        print(f"UQ metrics JSON exported to: {args.uq_metrics_path}")
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
