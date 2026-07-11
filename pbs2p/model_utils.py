"""Model helpers shared by compression and evaluation."""

from __future__ import annotations

import gc

import torch
from torch import nn


def find_layers(
    module: nn.Module,
    layer_types: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear),
    name: str = "",
) -> dict[str, nn.Module]:
    if isinstance(module, layer_types):
        return {name: module}

    result: dict[str, nn.Module] = {}
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        result.update(find_layers(child, layer_types, full_name))
    return result


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise NotImplementedError(
        f"Unsupported architecture: {getattr(model.config, 'model_type', 'unknown')}"
    )


def get_input_modules(model: nn.Module) -> list[nn.Module]:
    if hasattr(model.model, "decoder"):
        decoder = model.model.decoder
        names = ("embed_tokens", "embed_positions", "project_in")
        return [
            getattr(decoder, name)
            for name in names
            if getattr(decoder, name, None) is not None
        ]

    names = ("embed_tokens", "rotary_emb")
    return [
        getattr(model.model, name)
        for name in names
        if getattr(model.model, name, None) is not None
    ]


def get_output_modules(model: nn.Module) -> list[nn.Module]:
    modules: list[nn.Module] = []
    if hasattr(model.model, "decoder"):
        decoder = model.model.decoder
        for name in ("final_layer_norm", "project_out"):
            module = getattr(decoder, name, None)
            if module is not None:
                modules.append(module)
    else:
        norm = getattr(model.model, "norm", None)
        if norm is not None:
            modules.append(norm)
    modules.append(model.lm_head)
    return modules


def move_modules(modules: list[nn.Module], device: str | torch.device) -> None:
    for module in modules:
        module.to(device)


def layer_forward(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    kwargs: dict[str, object] = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
    return layer(hidden_states, **kwargs)[0]


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
