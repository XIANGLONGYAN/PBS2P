"""Binary weight-group search used by PBS2P."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

from .spbo import (
    optimize_single_binary_mse,
    optimize_two_binary_mse,
    residual_binarize,
)

logger = logging.getLogger(__name__)

_ERROR_NORMALIZER = 2048 * 4096 * 128


def _split_remaining(
    weights: Tensor,
    assigned_mask: Tensor,
    threshold: Tensor,
) -> tuple[Tensor, Tensor]:
    remaining = ~assigned_mask
    magnitudes = torch.abs(weights * remaining)
    lower = (magnitudes < threshold) & remaining
    upper = (magnitudes >= threshold) & remaining
    return lower, upper


def _activation_error(
    original: Tensor,
    approximation: Tensor,
    covariance: Tensor,
) -> Tensor:
    residual = (original - approximation).T
    residual_product = torch.einsum("ik,jk->ij", residual, residual)
    return torch.sum(residual_product * covariance) / _ERROR_NORMALIZER


@torch.no_grad()
def search_binary_groups(
    scores: Tensor,
    covariance: Tensor,
    search_columns: int = 50,
    num_partitions: int = 2,
) -> list[Tensor]:
    """Partition each weight block into one 2-bit and several 1-bit groups."""

    if scores.shape[1] < search_columns:
        raise ValueError(
            f"Block width ({scores.shape[1]}) must be at least "
            f"search_columns ({search_columns})."
        )

    column_scores = scores.abs().sum(dim=0)
    candidate_columns = torch.topk(column_scores, search_columns).indices

    best_error = float("inf")
    best_count = 1
    for count in range(1, search_columns):
        high_mask = torch.zeros_like(scores, dtype=torch.bool)
        high_mask[:, candidate_columns[:count]] = True
        high = residual_binarize(scores, high_mask, order=2)
        low = residual_binarize(scores, ~high_mask, order=2)
        error = torch.mean((scores - high - low) ** 2)
        if error < best_error:
            best_error = error
            best_count = count

    high_mask = torch.zeros_like(scores, dtype=torch.bool)
    high_mask[:, candidate_columns[:best_count]] = True
    approximation = optimize_two_binary_mse(scores, high_mask)
    masks = [high_mask]

    for _ in range(num_partitions):
        remaining_scores = scores * ~high_mask
        flat_magnitudes = remaining_scores.abs().reshape(-1)
        quantiles = np.linspace(0.10, 0.90, 81)
        thresholds = torch.as_tensor(
            np.quantile(
                flat_magnitudes.detach().cpu().numpy(),
                q=quantiles,
            ),
            device=scores.device,
            dtype=scores.dtype,
        )

        best_error = float("inf")
        best_lower: Tensor | None = None
        best_upper: Tensor | None = None
        for threshold in thresholds:
            lower, upper = _split_remaining(scores, high_mask, threshold)
            lower_approximation = residual_binarize(scores, lower, order=1)
            upper_approximation = residual_binarize(scores, upper, order=1)
            error = _activation_error(
                scores,
                approximation + lower_approximation + upper_approximation,
                covariance,
            )
            if error < best_error:
                best_error = error
                best_lower = lower
                best_upper = upper

        if best_lower is None or best_upper is None:
            best_lower, best_upper = _split_remaining(
                scores,
                high_mask,
                thresholds[0],
            )

        upper_approximation = optimize_single_binary_mse(scores, best_upper)
        masks.append(best_upper)
        approximation += upper_approximation
        high_mask = high_mask | best_upper

    masks.append(best_lower)
    return masks


def build_binary_group_masks(
    weights: Tensor,
    hessian_factor: Tensor,
    covariance: Tensor,
    search_columns: int = 50,
    num_partitions: int = 2,
) -> list[Tensor]:
    """Build binary groups from the Hessian-weighted fine-stage scores."""

    diagonal = torch.diag(hessian_factor).reshape(1, -1)
    scores = weights.square() / diagonal.square()
    masks = search_binary_groups(
        scores,
        covariance,
        search_columns=search_columns,
        num_partitions=num_partitions,
    )
    logger.debug(
        "Binary group ratios: %s",
        ", ".join(f"{mask.float().mean().item():.2f}" for mask in masks),
    )
    return masks
