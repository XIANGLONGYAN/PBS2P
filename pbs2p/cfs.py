"""Coarse-to-Fine Search (CFS) from the PBS2P paper."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_from_disk
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def block_influence(
    input_hidden_state: Tensor,
    output_hidden_state: Tensor,
) -> Tensor:
    """Return token-wise cosine distance across a transformer block."""

    hidden_size = input_hidden_state.shape[-1]
    inputs = input_hidden_state.reshape(-1, hidden_size)
    outputs = output_hidden_state.reshape(-1, hidden_size)
    similarities = torch.sum(inputs * outputs, dim=-1) / (
        inputs.norm(dim=-1) * outputs.norm(dim=-1)
    )
    return 1 - similarities.nan_to_num(nan=0.5)


def allocate_layer_sparsity(
    layer_importance: list[float],
    target_n: int,
    m: int,
) -> list[int]:
    """Allocate per-layer N values while preserving the target average."""

    if not 1 <= target_n < m:
        raise ValueError(f"CFS expects 1 <= N < M, got {target_n}:{m}.")

    num_layers = len(layer_importance)
    if num_layers == 1:
        return [target_n]

    ranked_layers = sorted(
        range(num_layers),
        key=layer_importance.__getitem__,
        reverse=True,
    )
    highest_n = min(m, target_n + 1)
    lowest_n = max(1, target_n - 1)
    allocation = [target_n] * num_layers

    for rank, layer_index in enumerate(ranked_layers):
        value = highest_n - (highest_n - lowest_n) * rank / (num_layers - 1)
        allocation[layer_index] = round(value)

    target_total = target_n * num_layers
    difference = target_total - sum(allocation)
    if difference > 0:
        for layer_index in ranked_layers:
            if allocation[layer_index] < highest_n:
                allocation[layer_index] += 1
                difference -= 1
                if difference == 0:
                    break
    elif difference < 0:
        for layer_index in reversed(ranked_layers):
            if allocation[layer_index] > lowest_n:
                allocation[layer_index] -= 1
                difference += 1
                if difference == 0:
                    break

    return allocation


class _LayerImportanceEvaluator:
    def __init__(self, model_name: str, device: str) -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.model.eval()
        self.model.config.use_cache = False

        if hasattr(self.model.model, "layers"):
            self.layers = self.model.model.layers
        elif hasattr(self.model.model, "decoder"):
            self.layers = self.model.model.decoder.layers
        else:
            raise NotImplementedError(
                f"Unsupported architecture: {self.model.config.model_type}"
            )
        self.importance = [0.0] * len(self.layers)

    @torch.inference_mode()
    def evaluate(
        self,
        prompts: list[str],
        max_sequence_length: int = 1024,
        stride: int = 256,
    ) -> None:
        tokens = self.tokenizer(
            prompts,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        max_prompt_length = int(attention_mask.sum(dim=-1).max())

        for start in range(0, max_prompt_length, stride):
            sequence_indices = (attention_mask.sum(dim=-1) > start).nonzero().squeeze()
            if sequence_indices.numel() == 0:
                continue
            if sequence_indices.ndim == 0:
                sequence_indices = sequence_indices.unsqueeze(0)
            window_ids = input_ids[
                sequence_indices,
                start : start + max_sequence_length,
            ].to(self.device)
            window_attention = attention_mask[
                sequence_indices,
                start : start + max_sequence_length,
            ].to(self.device)
            outputs = self.model(
                input_ids=window_ids,
                attention_mask=window_attention,
                output_hidden_states=True,
            )
            for layer_index in range(len(self.layers)):
                self.importance[layer_index] += block_influence(
                    outputs.hidden_states[layer_index],
                    outputs.hidden_states[layer_index + 1],
                ).sum().item()

    def free(self) -> None:
        self.model.cpu()
        del self.layers
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def coarse_stage_search(
    model_name: str,
    data_root: str | Path,
    device: str,
    *,
    max_samples: int = 1000,
    seed: int = 42,
) -> list[float]:
    """Compute block-influence scores for the CFS coarse stage."""

    dataset_path = Path(data_root) / "wikitext" / "testdata"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"CFS dataset not found at {dataset_path}. "
            "Run `python prepare_data.py --datasets wikitext` first."
        )

    dataset = load_from_disk(str(dataset_path))
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
    )
    evaluator = _LayerImportanceEvaluator(model_name, device)

    try:
        for batch in tqdm(dataloader, desc="CFS coarse search", unit="sample"):
            evaluator.evaluate(batch["text"])
        importance = evaluator.importance
    finally:
        evaluator.free()

    logger.info(
        "CFS layer importance range: %.4f to %.4f",
        np.min(importance),
        np.max(importance),
    )
    return importance


def run_cfs(
    model_name: str,
    data_root: str | Path,
    device: str,
    target_n: int,
    m: int,
    *,
    max_samples: int = 1000,
    seed: int = 42,
) -> tuple[list[float], list[int]]:
    """Run the CFS coarse stage and return per-layer N:M allocations."""

    importance = coarse_stage_search(
        model_name,
        data_root,
        device,
        max_samples=max_samples,
        seed=seed,
    )
    allocation = allocate_layer_sparsity(importance, target_n, m)
    logger.info("CFS per-layer N values: %s", allocation)
    return importance, allocation
