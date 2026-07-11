"""Memory-efficient perplexity evaluation."""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from .data import EncodedDataset
from .model_utils import (
    cleanup_memory,
    get_decoder_layers,
    get_input_modules,
    get_output_modules,
    layer_forward,
    move_modules,
)

logger = logging.getLogger(__name__)


class _StopForward(Exception):
    pass


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    encoded: EncodedDataset,
    device: str,
    dataset: str,
) -> float:
    input_ids = encoded.input_ids
    num_samples = input_ids.numel() // model.seqlen
    if num_samples == 0:
        raise ValueError(f"{dataset} contains fewer than {model.seqlen} tokens.")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_decoder_layers(model)
    input_modules = get_input_modules(model)
    move_modules(input_modules, device)
    layers[0] = layers[0].to(device)

    dtype = next(model.parameters()).dtype
    inputs = torch.zeros(
        (
            num_samples,
            model.seqlen,
            model.config.hidden_size,
        ),
        dtype=dtype,
        device=device,
    )
    cache: dict[str, object] = {
        "index": 0,
        "attention_mask": None,
        "position_ids": None,
        "position_embeddings": None,
    }

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, hidden_states: Tensor, **kwargs: object) -> None:
            index = int(cache["index"])
            inputs[index] = hidden_states
            cache["index"] = index + 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise _StopForward

    layers[0] = Catcher(layers[0])
    try:
        for sample_index in tqdm(
            range(num_samples),
            desc=f"Preparing {dataset}",
            unit="sample",
        ):
            batch = input_ids[
                :,
                sample_index * model.seqlen : (sample_index + 1) * model.seqlen,
            ].to(device)
            try:
                model(batch)
            except _StopForward:
                pass
    finally:
        layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    move_modules(input_modules, "cpu")
    cleanup_memory()

    outputs = torch.zeros_like(inputs)
    for layer_index in tqdm(
        range(len(layers)),
        desc=f"Evaluating {dataset}",
        unit="layer",
    ):
        layer = layers[layer_index].to(device)
        for sample_index in range(num_samples):
            outputs[sample_index] = layer_forward(
                layer,
                inputs[sample_index].unsqueeze(0),
                attention_mask=cache["attention_mask"],
                position_ids=cache["position_ids"],
                position_embeddings=cache["position_embeddings"],
            )
        layers[layer_index] = layer.cpu()
        del layer
        cleanup_memory()
        inputs, outputs = outputs, inputs

    output_modules = get_output_modules(model)
    move_modules(output_modules, device)
    labels = input_ids.to(device)
    negative_log_likelihoods = []

    for sample_index in range(num_samples):
        hidden_states = inputs[sample_index].unsqueeze(0)
        if hasattr(model.model, "decoder"):
            decoder = model.model.decoder
            if getattr(decoder, "final_layer_norm", None) is not None:
                hidden_states = decoder.final_layer_norm(hidden_states)
            if getattr(decoder, "project_out", None) is not None:
                hidden_states = decoder.project_out(hidden_states)
        elif getattr(model.model, "norm", None) is not None:
            hidden_states = model.model.norm(hidden_states)

        logits = model.lm_head(hidden_states)
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[
            :,
            sample_index * model.seqlen : (sample_index + 1) * model.seqlen,
        ][:, 1:]
        loss = nn.functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
        )
        negative_log_likelihoods.append(loss.float() * model.seqlen)

    perplexity = torch.exp(
        torch.stack(negative_log_likelihoods).sum()
        / (num_samples * model.seqlen)
    ).item()
    logger.info("%s perplexity: %.4f", dataset, perplexity)

    move_modules(output_modules, "cpu")
    model.config.use_cache = use_cache
    cleanup_memory()
    return perplexity
