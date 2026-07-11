#!/usr/bin/env python3
"""Compress and evaluate a causal language model with PBS2P."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time

import torch

from pbs2p.cfs import run_cfs
from pbs2p.data import get_loaders, get_tokenizer, set_seed
from pbs2p.eval_ppl import evaluate_perplexity
from pbs2p.model_utils import get_decoder_layers
from pbs2p.pipeline import compress_model, load_model, measure_sparsity

logger = logging.getLogger("pbs2p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PBS2P: Progressive Binarization with Semi-Structured Pruning"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", help="Hugging Face model name or local path.")
    parser.add_argument(
        "calibration_dataset",
        choices=("wikitext2", "ptb", "c4"),
        help="Dataset used for calibration.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("PBS2P_DATA_ROOT", "./data"),
        help="Root containing prepared WikiText2, PTB, and C4 datasets.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n", type=int, default=6, help="Retained N in N:M.")
    parser.add_argument("--m", type=int, default=8, help="Group size M in N:M.")
    parser.add_argument(
        "--cfs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the paper's Coarse-to-Fine Search.",
    )
    parser.add_argument(
        "--cfs-samples",
        type=int,
        default=1000,
        help="WikiText2 samples used by the CFS coarse stage.",
    )
    parser.add_argument("--cfs-seed", type=int, default=42)
    parser.add_argument(
        "--ablation",
        choices=("none", "one-shot"),
        default="none",
        help=(
            "Use 'one-shot' for the vanilla prune-then-binarize SPBO ablation."
        ),
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration sequences.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--partitions",
        type=int,
        default=2,
        help="Number of 1-bit non-salient partitions.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Alternating optimization iterations in every SPBO step.",
    )
    parser.add_argument("--damp-percent", type=float, default=0.01)
    parser.add_argument(
        "--eval-datasets",
        nargs="*",
        choices=("wikitext2", "ptb", "c4"),
        default=("wikitext2", "ptb", "c4"),
        help="Perplexity datasets to evaluate; pass no values to skip.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated legacy lm-eval zero-shot tasks.",
    )
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument(
        "--save-model",
        default="",
        help="Optional output directory for the compressed model.",
    )
    parser.add_argument(
        "--results-file",
        default="",
        help="Optional JSON file for evaluation results.",
    )
    parser.add_argument("--log-file", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(log_file: str, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available.")
    if not 0 < args.n <= args.m:
        raise ValueError(f"Expected 0 < N <= M, got {args.n}:{args.m}.")
    if args.cfs and args.n == args.m:
        raise ValueError("CFS requires pruning, so N must be smaller than M.")
    if args.block_size % args.m:
        raise ValueError("--block-size must be divisible by M.")
    if args.block_size < 50:
        raise ValueError("--block-size must be at least 50 for group search.")
    if args.partitions < 1:
        raise ValueError("--partitions must be positive.")


def evaluate_zero_shot(
    model: torch.nn.Module,
    model_name: str,
    tasks: str,
    num_fewshot: int,
    device: str,
) -> dict[str, object]:
    from lm_eval import evaluator
    from pbs2p.eval_lm import EvalLM

    tokenizer = get_tokenizer(model_name)
    evaluation_model = EvalLM(model, tokenizer, device=device)
    return evaluator.simple_evaluate(
        evaluation_model,
        tasks=tasks,
        num_fewshot=num_fewshot,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.verbose)
    validate_args(args)
    set_seed(args.seed)

    method = "PBS2P (SPBO + CFS)" if args.cfs else "PBS2P (SPBO)"
    if args.ablation == "one-shot":
        method = f"{method} one-shot ablation"
    logger.info("Method: %s", method)
    logger.info(
        "Model: %s | calibration: %s | sparsity: %d:%d",
        args.model,
        args.calibration_dataset,
        args.n,
        args.m,
    )

    layer_n: list[int] | None = None
    if args.cfs:
        logger.info("Running CFS coarse-stage layer allocation")
        _, layer_n = run_cfs(
            args.model,
            args.data_root,
            args.device,
            args.n,
            args.m,
            max_samples=args.cfs_samples,
            seed=args.cfs_seed,
        )

    logger.info("Loading model")
    model = load_model(args.model)
    if layer_n is None:
        layer_n = [args.n] * len(get_decoder_layers(model))

    logger.info("Loading calibration data")
    calibration_data, _ = get_loaders(
        args.calibration_dataset,
        args.model,
        args.data_root,
        num_samples=args.nsamples,
        seed=args.seed,
        sequence_length=model.seqlen,
    )

    started_at = time.time()
    compress_model(
        model,
        calibration_data,
        args.device,
        layer_n,
        args.m,
        block_size=args.block_size,
        damp_percent=args.damp_percent,
        num_partitions=args.partitions,
        iterations=args.iterations,
        stepwise=args.ablation != "one-shot",
    )
    elapsed = time.time() - started_at
    sparsity = measure_sparsity(model)
    logger.info(
        "Compression complete in %.1f minutes | measured sparsity %.2f%%",
        elapsed / 60,
        100 * sparsity,
    )

    if args.save_model:
        output_path = Path(args.save_model)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        get_tokenizer(args.model).save_pretrained(output_path)
        logger.info("Saved compressed model to %s", output_path)

    results: dict[str, object] = {
        "method": method,
        "model": args.model,
        "sparsity": {"n": args.n, "m": args.m, "measured": sparsity},
        "layer_n": layer_n,
        "perplexity": {},
    }
    for dataset in args.eval_datasets:
        logger.info("Loading %s evaluation data", dataset)
        _, encoded = get_loaders(
            dataset,
            args.model,
            args.data_root,
            num_samples=0,
            seed=args.seed,
            sequence_length=model.seqlen,
        )
        results["perplexity"][dataset] = evaluate_perplexity(
            model,
            encoded,
            args.device,
            dataset,
        )

    if args.tasks:
        logger.info("Evaluating zero-shot tasks: %s", args.tasks)
        results["zero_shot"] = evaluate_zero_shot(
            model,
            args.model,
            args.tasks,
            args.num_fewshot,
            args.device,
        )

    if args.results_file:
        results_path = Path(args.results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            json.dumps(results, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Saved results to %s", results_path)

    logger.info("Results: %s", json.dumps(results, default=str))


if __name__ == "__main__":
    main()
