"""End-to-end layer-wise PBS2P compression pipeline."""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from .gptq import PBS2PGPTQ
from .model_utils import (
    cleanup_memory,
    find_layers,
    get_decoder_layers,
    get_input_modules,
    layer_forward,
    move_modules,
)

logger = logging.getLogger(__name__)


class _StopForward(Exception):
    pass


def load_model(model_name: str) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.seqlen = min(
        2048,
        getattr(model.config, "max_position_embeddings", 2048),
    )
    model.eval()
    return model


@torch.no_grad()
def _capture_inputs(
    model: nn.Module,
    calibration_data: list[tuple[Tensor, Tensor]],
    device: str,
) -> tuple[
    Tensor,
    Tensor | None,
    Tensor | None,
    tuple[Tensor, Tensor] | None,
]:
    layers = get_decoder_layers(model)
    input_modules = get_input_modules(model)
    move_modules(input_modules, device)
    layers[0] = layers[0].to(device)

    num_samples = len(calibration_data)
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
        for sample, _ in tqdm(
            calibration_data,
            desc="Collecting calibration inputs",
            unit="sample",
        ):
            try:
                model(sample.to(device))
            except _StopForward:
                pass
    finally:
        layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    move_modules(input_modules, "cpu")
    cleanup_memory()
    return (
        inputs,
        cache["attention_mask"],
        cache["position_ids"],
        cache["position_embeddings"],
    )


@torch.no_grad()
def compress_model(
    model: nn.Module,
    calibration_data: list[tuple[Tensor, Tensor]],
    device: str,
    layer_n: list[int],
    m: int,
    *,
    block_size: int = 128,
    damp_percent: float = 0.01,
    num_partitions: int = 2,
    iterations: int = 15,
    stepwise: bool = True,
) -> None:
    """Compress every transformer layer with PBS2P."""

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_decoder_layers(model)
    if len(layer_n) != len(layers):
        raise ValueError(
            f"Expected {len(layers)} layer sparsity values, got {len(layer_n)}."
        )

    (
        inputs,
        attention_mask,
        position_ids,
        position_embeddings,
    ) = _capture_inputs(model, calibration_data, device)
    outputs = torch.zeros_like(inputs)

    progress = tqdm(range(len(layers)), desc="Compressing PBS2P", unit="layer")
    for layer_index in progress:
        n = layer_n[layer_index]
        progress.set_postfix_str(f"N:M={n}:{m}")
        layer = layers[layer_index].to(device)
        subset = find_layers(layer)
        compressors = {
            name: PBS2PGPTQ(
                sublayer,
                n=n,
                m=m,
                group_size=block_size,
            )
            for name, sublayer in subset.items()
        }

        handles = []
        for name, compressor in compressors.items():
            def add_batch(
                _: nn.Module,
                hook_inputs: tuple[Tensor, ...],
                hook_output: Tensor,
                *,
                compressor: PBS2PGPTQ = compressor,
            ) -> None:
                compressor.add_batch(hook_inputs[0].data, hook_output.data)

            handles.append(subset[name].register_forward_hook(add_batch))

        for sample_index in range(len(calibration_data)):
            outputs[sample_index] = layer_forward(
                layer,
                inputs[sample_index].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        for handle in handles:
            handle.remove()

        for name, compressor in compressors.items():
            logger.debug("Compressing layer %d %s", layer_index, name)
            compressor.quantize(
                block_size=block_size,
                damp_percent=damp_percent,
                num_partitions=num_partitions,
                iterations=iterations,
                stepwise=stepwise,
            )

        for sample_index in range(len(calibration_data)):
            outputs[sample_index] = layer_forward(
                layer,
                inputs[sample_index].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        layers[layer_index] = layer.cpu()
        del layer, compressors
        cleanup_memory()
        inputs, outputs = outputs, inputs

    model.config.use_cache = use_cache


def measure_sparsity(model: nn.Module) -> float:
    zero_count = 0
    parameter_count = 0
    for layer in get_decoder_layers(model):
        for sublayer in find_layers(layer).values():
            weights = sublayer.weight.data
            zero_count += int((weights == 0).sum())
            parameter_count += weights.numel()
    return zero_count / parameter_count
