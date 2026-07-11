"""Stepwise semi-structured Pruning with Binarization Optimization (SPBO).

This module implements Section 3.2 of the PBS2P paper.  The public
``stepwise_binarize`` function progressively applies an N:M mask sequence and
re-estimates the binary scales and offsets after every pruning step.  The
``one_shot_binarize`` function is retained only for the paper's vanilla
prune-then-binarize ablation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    masked = torch.where(mask, values, torch.full_like(values, float("nan")))
    mean = torch.nanmean(masked, dim=1)
    return torch.nan_to_num(mean)


@torch.no_grad()
def residual_binarize(
    weights: Tensor,
    mask: Tensor,
    order: int = 2,
) -> Tensor:
    """Binarize residuals row-wise under ``mask``.

    This lightweight approximation is used while searching the binary weight
    groups.  SPBO itself additionally optimizes the parameters against the
    calibration activations.
    """

    approximation = torch.zeros_like(weights)
    masked_weights = weights * mask

    for _ in range(order):
        residual = masked_weights - approximation
        mean = _masked_mean(residual, mask)
        centered = torch.where(
            mask,
            residual - mean[:, None],
            torch.full_like(residual, float("nan")),
        )
        scale = torch.nan_to_num(torch.nanmean(centered.abs(), dim=1))
        binary = torch.sign(centered) * scale[:, None] + mean[:, None]
        approximation += binary * mask

    return approximation


@torch.no_grad()
def optimize_single_binary_mse(
    weights: Tensor,
    mask: Tensor,
    iterations: int = 15,
) -> Tensor:
    """Fit one binary residual using unweighted reconstruction error."""

    masked_weights = weights * mask
    mean = _masked_mean(masked_weights, mask)
    centered = torch.where(
        mask,
        masked_weights - mean[:, None],
        torch.full_like(weights, float("nan")),
    )
    scale = torch.nan_to_num(torch.nanmean(centered.abs(), dim=1))
    binary = torch.sign(centered)
    approximation = (scale[:, None] * binary + mean[:, None]) * mask

    for _ in range(iterations):
        mean += _masked_mean(masked_weights - approximation, mask)
        denominator = torch.sum((binary * mask) ** 2, dim=1) + 1e-8
        scale = torch.sum(
            binary * mask * (masked_weights - mean[:, None] * mask),
            dim=1,
        ) / denominator
        binary = torch.sign(masked_weights - mean[:, None] * mask)
        approximation = (scale[:, None] * binary + mean[:, None]) * mask

    return approximation


@torch.no_grad()
def optimize_two_binary_mse(
    weights: Tensor,
    mask: Tensor,
    iterations: int = 15,
) -> Tensor:
    """Fit two binary residuals using unweighted reconstruction error."""

    masked_weights = weights * mask
    approximation = torch.zeros_like(weights)
    binaries: list[Tensor] = []
    scales: list[Tensor] = []
    mean = torch.zeros(weights.shape[0], device=weights.device)

    for _ in range(2):
        residual = masked_weights - approximation
        residual_mean = _masked_mean(residual, mask)
        mean += residual_mean
        centered = torch.where(
            mask,
            residual - residual_mean[:, None],
            torch.full_like(residual, float("nan")),
        )
        scale = torch.nan_to_num(torch.nanmean(centered.abs(), dim=1))
        binary = torch.sign(centered)
        scales.append(scale)
        binaries.append(binary)
        approximation += (scale[:, None] * binary + residual_mean[:, None]) * mask

    approximation, _, _, _, _ = _optimize_two_binary_stage(
        weights,
        mask,
        binaries,
        scales,
        mean,
        approximation,
        covariance=None,
        local_iterations=iterations,
        activation_iterations=0,
    )
    return approximation


def _optimize_single_binary_stage(
    weights: Tensor,
    mask: Tensor,
    binary: Tensor,
    scale: Tensor,
    mean: Tensor,
    approximation: Tensor,
    covariance: Tensor | None,
    local_iterations: int,
    activation_iterations: int,
    refresh_after_activation: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    masked_weights = weights * mask

    for _ in range(local_iterations):
        mean += _masked_mean(masked_weights - approximation, mask)
        masked_binary = binary * mask
        denominator = torch.sum(masked_binary**2, dim=1) + 1e-8
        scale = torch.sum(
            masked_binary * (masked_weights - mean[:, None] * mask),
            dim=1,
        ) / denominator
        binary = torch.sign(masked_weights - mean[:, None] * mask)
        approximation = (scale[:, None] * binary + mean[:, None]) * mask

    if activation_iterations:
        if covariance is None:
            raise ValueError("Calibration covariance is required for SPBO.")
        pair_mask = mask[:, :, None] * mask[:, None, :]
        mean_denominator = (
            torch.sum(covariance * pair_mask, dim=(1, 2), dtype=torch.float32)
            + 1e-10
        )
        masked_binary = binary * mask
        scale_denominator = (
            torch.sum(
                covariance
                * masked_binary[:, :, None]
                * masked_binary[:, None, :],
                dim=(1, 2),
            )
            + 1e-10
        )

        for _ in range(activation_iterations):
            mean = torch.sum(
                covariance
                * (masked_weights - scale[:, None] * binary * mask)[:, :, None]
                * pair_mask,
                dim=(1, 2),
            ) / mean_denominator
            scale = torch.sum(
                covariance
                * masked_binary[:, :, None]
                * (masked_weights - mean[:, None] * mask)[:, None, :],
                dim=(1, 2),
            ) / scale_denominator

        if refresh_after_activation:
            approximation = (scale[:, None] * binary + mean[:, None]) * mask

    return approximation, binary, scale, mean


def _nearest_two_binary_codes(
    masked_weights: Tensor,
    mask: Tensor,
    mean: Tensor,
    scales: Sequence[Tensor],
) -> list[Tensor]:
    centered = (masked_weights - mean[:, None] * mask).unsqueeze(-1)
    levels = torch.stack(
        (
            -scales[0] - scales[1],
            -scales[0] + scales[1],
            scales[0] - scales[1],
            scales[0] + scales[1],
        ),
        dim=1,
    ).unsqueeze(1)
    indices = torch.argmin(torch.abs(centered - levels), dim=-1)

    first = torch.ones_like(indices)
    first[(indices == 0) | (indices == 1)] = -1
    second = torch.ones_like(indices)
    second[(indices == 0) | (indices == 2)] = -1
    return [first, second]


def _optimize_two_binary_stage(
    weights: Tensor,
    mask: Tensor,
    binaries: list[Tensor],
    scales: list[Tensor],
    mean: Tensor,
    approximation: Tensor,
    covariance: Tensor | None,
    local_iterations: int,
    activation_iterations: int,
    refresh_after_activation: bool = True,
) -> tuple[Tensor, list[Tensor], list[Tensor], Tensor, Tensor]:
    masked_weights = weights * mask

    for _ in range(local_iterations):
        mean += _masked_mean(masked_weights - approximation, mask)
        first = binaries[0] * mask
        second = binaries[1] * mask
        scales[0] = torch.sum(
            first
            * (
                masked_weights
                - mean[:, None] * mask
                - scales[1][:, None] * second
            ),
            dim=1,
        ) / (torch.sum(first**2, dim=1) + 1e-8)
        scales[1] = torch.sum(
            second
            * (
                masked_weights
                - mean[:, None] * mask
                - scales[0][:, None] * first
            ),
            dim=1,
        ) / (torch.sum(second**2, dim=1) + 1e-8)
        binaries = _nearest_two_binary_codes(masked_weights, mask, mean, scales)
        approximation = (
            scales[0][:, None] * binaries[0]
            + scales[1][:, None] * binaries[1]
            + mean[:, None]
        ) * mask

    if activation_iterations:
        if covariance is None:
            raise ValueError("Calibration covariance is required for SPBO.")
        pair_mask = mask[:, :, None] * mask[:, None, :]
        mean_denominator = torch.sum(covariance * pair_mask, dim=(1, 2)) + 1e-10
        masked_binaries = [binary * mask for binary in binaries]
        scale_denominators = [
            torch.sum(
                covariance
                * binary[:, :, None]
                * binary[:, None, :],
                dim=(1, 2),
            )
            + 1e-10
            for binary in masked_binaries
        ]

        for _ in range(activation_iterations):
            binary_sum = (
                scales[0][:, None] * binaries[0]
                + scales[1][:, None] * binaries[1]
            ) * mask
            mean = torch.sum(
                covariance
                * (masked_weights - binary_sum)[:, :, None]
                * pair_mask,
                dim=(1, 2),
            ) / mean_denominator
            centered = masked_weights - mean[:, None] * mask
            scales[0] = torch.sum(
                covariance
                * masked_binaries[0][:, :, None]
                * (
                    centered[:, None, :]
                    - (scales[1][:, None] * masked_binaries[1])[:, None, :]
                ),
                dim=(1, 2),
            ) / scale_denominators[0]
            scales[1] = torch.sum(
                covariance
                * masked_binaries[1][:, :, None]
                * (
                    centered[:, None, :]
                    - (scales[0][:, None] * masked_binaries[0])[:, None, :]
                ),
                dim=(1, 2),
            ) / scale_denominators[1]

        if refresh_after_activation:
            approximation = (
                scales[0][:, None] * binaries[0]
                + scales[1][:, None] * binaries[1]
                + mean[:, None]
            ) * mask

    return approximation, binaries, scales, mean, masked_weights


def _initialize_single_binary(
    weights: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    masked_weights = weights * mask
    mean = _masked_mean(masked_weights, mask)
    centered = torch.where(
        mask,
        masked_weights - mean[:, None],
        torch.full_like(weights, float("nan")),
    )
    scale = torch.nan_to_num(torch.nanmean(centered.abs(), dim=1))
    binary = torch.sign(centered)
    approximation = (scale[:, None] * binary + mean[:, None]) * mask
    return approximation, binary, scale, mean


def _initialize_two_binary(
    weights: Tensor,
    mask: Tensor,
) -> tuple[Tensor, list[Tensor], list[Tensor], Tensor]:
    masked_weights = weights * mask
    approximation = torch.zeros_like(weights)
    binaries: list[Tensor] = []
    scales: list[Tensor] = []
    mean = torch.zeros(weights.shape[0], device=weights.device)

    for _ in range(2):
        residual = masked_weights - approximation
        residual_mean = _masked_mean(residual, mask)
        mean += residual_mean
        centered = torch.where(
            mask,
            residual - residual_mean[:, None],
            torch.full_like(residual, float("nan")),
        )
        scale = torch.nan_to_num(torch.nanmean(centered.abs(), dim=1))
        binary = torch.sign(centered)
        binaries.append(binary)
        scales.append(scale)
        approximation += (scale[:, None] * binary + residual_mean[:, None]) * mask

    return approximation, binaries, scales, mean


@torch.no_grad()
def stepwise_binarize(
    weights: Tensor,
    group_mask: Tensor,
    pruning_masks: Sequence[Tensor],
    covariance: Tensor,
    order: int,
    iterations: int = 15,
) -> Tensor:
    """Run SPBO using a progressively stricter sequence of pruning masks."""

    mask = group_mask.clone()
    if order == 1:
        approximation, binary, scale, mean = _initialize_single_binary(weights, mask)
        approximation, binary, scale, mean = _optimize_single_binary_stage(
            weights,
            mask,
            binary,
            scale,
            mean,
            approximation,
            covariance,
            iterations,
            iterations,
        )
        for pruning_mask in pruning_masks:
            mask &= pruning_mask
            approximation, binary, scale, mean = _optimize_single_binary_stage(
                weights,
                mask,
                binary,
                scale,
                mean,
                approximation,
                covariance,
                iterations,
                iterations,
                refresh_after_activation=False,
            )
        return (scale[:, None] * binary + mean[:, None]) * mask

    if order == 2:
        approximation, binaries, scales, mean = _initialize_two_binary(weights, mask)
        approximation, binaries, scales, mean, _ = _optimize_two_binary_stage(
            weights,
            mask,
            binaries,
            scales,
            mean,
            approximation,
            covariance,
            iterations,
            iterations,
        )
        for pruning_mask in pruning_masks:
            mask &= pruning_mask
            approximation, binaries, scales, mean, _ = _optimize_two_binary_stage(
                weights,
                mask,
                binaries,
                scales,
                mean,
                approximation,
                covariance,
                iterations,
                iterations,
                refresh_after_activation=False,
            )
        return (
            scales[0][:, None] * binaries[0]
            + scales[1][:, None] * binaries[1]
            + mean[:, None]
        ) * mask

    raise ValueError(f"PBS2P supports binary residual orders 1 and 2, got {order}.")


@torch.no_grad()
def one_shot_binarize(
    weights: Tensor,
    group_mask: Tensor,
    pruning_mask: Tensor,
    covariance: Tensor,
    order: int,
    iterations: int = 15,
) -> Tensor:
    """Vanilla prune-then-binarize baseline used in the SPBO ablation."""

    mask = group_mask & pruning_mask
    if order == 1:
        approximation, binary, scale, mean = _initialize_single_binary(weights, mask)
        approximation, _, _, _ = _optimize_single_binary_stage(
            weights,
            mask,
            binary,
            scale,
            mean,
            approximation,
            covariance,
            iterations,
            iterations,
        )
        return approximation

    if order == 2:
        approximation, binaries, scales, mean = _initialize_two_binary(weights, mask)
        approximation, _, _, _, _ = _optimize_two_binary_stage(
            weights,
            mask,
            binaries,
            scales,
            mean,
            approximation,
            covariance,
            iterations,
            iterations,
        )
        return approximation

    raise ValueError(f"PBS2P supports binary residual orders 1 and 2, got {order}.")


class Binarizer(nn.Module):
    """Apply SPBO or its one-shot ablation to a binary weight group."""

    def __init__(self, group_size: int = 128) -> None:
        super().__init__()
        self.group_size = group_size

    def forward(
        self,
        weights: Tensor,
        group_mask: Tensor,
        pruning_masks: Sequence[Tensor],
        covariance: Tensor,
        order: int,
        iterations: int = 15,
        stepwise: bool = True,
    ) -> Tensor:
        if stepwise:
            return stepwise_binarize(
                weights,
                group_mask,
                pruning_masks,
                covariance,
                order,
                iterations,
            )
        return one_shot_binarize(
            weights,
            group_mask,
            pruning_masks[-1],
            covariance,
            order,
            iterations,
        )
