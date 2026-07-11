"""Layer-wise GPTQ error compensation for PBS2P."""

from __future__ import annotations

import logging
import math
import time

import torch
from torch import Tensor, nn
import transformers

from .grouping import build_binary_group_masks
from .spbo import Binarizer

logger = logging.getLogger(__name__)


def build_progressive_nm_masks(
    scores: Tensor,
    n: int,
    m: int,
) -> list[Tensor]:
    """Decompose an N:M mask into the M-N nested masks used by SPBO."""

    if not 0 < n <= m:
        raise ValueError(f"Expected 0 < N <= M, got {n}:{m}.")
    rows, columns = scores.shape
    if columns % m:
        raise ValueError(
            f"Weight block width ({columns}) must be divisible by M ({m})."
        )

    grouped_scores = scores.reshape(rows, columns // m, m)
    masks: list[Tensor] = []
    for step in range(m - n):
        keep = m - step - 1
        indices = torch.topk(grouped_scores, keep, dim=-1).indices
        grouped_mask = torch.zeros_like(grouped_scores, dtype=torch.bool)
        grouped_mask.scatter_(-1, indices, True)
        masks.append(grouped_mask.reshape(rows, columns))

    if not masks:
        masks.append(torch.ones_like(scores, dtype=torch.bool))
    return masks


class PBS2PGPTQ:
    """Compress one linear layer with PBS2P and GPTQ compensation."""

    def __init__(
        self,
        layer: nn.Module,
        n: int,
        m: int,
        group_size: int = 128,
    ) -> None:
        self.layer = layer
        self.device = layer.weight.device
        self.n = n
        self.m = m
        self.group_size = group_size
        self.binarizer = Binarizer(group_size=group_size)

        weights = self._matrix(layer.weight.data)
        self.rows, self.columns = weights.shape
        self.hessian = torch.zeros(
            (self.columns, self.columns),
            device=self.device,
        )
        self.inputs: list[Tensor] = []
        self.num_samples = 0

    def _matrix(self, weights: Tensor) -> Tensor:
        if isinstance(self.layer, nn.Conv2d):
            return weights.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            return weights.t()
        return weights

    def add_batch(self, inputs: Tensor, _: Tensor) -> None:
        """Accumulate calibration inputs for Hessian and activation objectives."""

        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        self.inputs.append(inputs)

        batch_size = inputs.shape[0]
        flattened = inputs
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if flattened.ndim == 3:
                flattened = flattened.reshape(-1, flattened.shape[-1])
            flattened = flattened.t()

        self.hessian *= self.num_samples / (self.num_samples + batch_size)
        self.num_samples += batch_size
        flattened = math.sqrt(2 / self.num_samples) * flattened.float()
        self.hessian += flattened.matmul(flattened.t())

    @torch.no_grad()
    def quantize(
        self,
        *,
        block_size: int = 128,
        damp_percent: float = 0.01,
        num_partitions: int = 2,
        iterations: int = 15,
        stepwise: bool = True,
    ) -> dict[str, float]:
        """Quantize the wrapped layer.

        ``stepwise=True`` is the paper's SPBO method.  ``False`` runs the
        vanilla prune-then-binarize ablation.
        """

        if self.group_size % block_size:
            raise ValueError("group_size must be divisible by block_size.")
        if block_size % self.m:
            raise ValueError("block_size must be divisible by M.")
        if not self.inputs:
            raise RuntimeError("No calibration inputs were collected.")

        started_at = time.time()
        weights = self._matrix(self.layer.weight.data.clone()).float()
        hessian = self.hessian
        self.hessian = None

        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weights[:, dead] = 0

        damp = damp_percent * torch.mean(torch.diag(hessian))
        diagonal_indices = torch.arange(self.columns, device=self.device)
        hessian[diagonal_indices, diagonal_indices] += damp
        hessian = torch.linalg.cholesky(hessian)
        hessian = torch.cholesky_inverse(hessian)
        hessian_factor = torch.linalg.cholesky(hessian, upper=True)

        losses = torch.zeros(self.rows, device=self.device)
        calibration_inputs = torch.cat(self.inputs)

        for column_start in range(0, self.columns, block_size):
            column_end = min(column_start + block_size, self.columns)
            block_weights = weights[:, column_start:column_end].clone()
            block_factor = hessian_factor[
                column_start:column_end,
                column_start:column_end,
            ]
            covariance = torch.matmul(
                calibration_inputs[:, :, column_start:column_end]
                .float()
                .transpose(1, 2),
                calibration_inputs[:, :, column_start:column_end].float(),
            ).mean(dim=0)

            group_masks = build_binary_group_masks(
                block_weights,
                block_factor,
                covariance,
                search_columns=50,
                num_partitions=num_partitions,
            )
            orders = [2] + [1] * (num_partitions + 1)
            if len(group_masks) != len(orders):
                raise RuntimeError("Binary group search returned an invalid mask count.")

            fine_scores = (
                block_weights.square()
                / torch.diag(block_factor).reshape(1, -1).square()
            )
            pruning_masks = build_progressive_nm_masks(
                fine_scores,
                self.n,
                self.m,
            )

            quantized_groups = [
                self.binarizer(
                    block_weights,
                    group_mask,
                    pruning_masks,
                    covariance,
                    order,
                    iterations,
                    stepwise,
                )
                for group_mask, order in zip(group_masks, orders)
            ]

            quantized = torch.zeros_like(block_weights)
            for group, group_mask in zip(quantized_groups, group_masks):
                quantized += group * group_mask

            inverse_diagonal = torch.diag(block_factor)
            block_losses = (
                (block_weights - quantized).square()
                / inverse_diagonal.reshape(1, -1).square()
            )
            errors = (
                (block_weights - quantized)
                / inverse_diagonal.reshape(1, -1)
            )
            weights[:, column_start:column_end] = quantized
            losses += torch.sum(block_losses, dim=1) / 2
            weights[:, column_end:] -= errors.matmul(
                hessian_factor[column_start:column_end, column_end:]
            )

        if weights.is_cuda:
            torch.cuda.synchronize(weights.device)

        if isinstance(self.layer, transformers.Conv1D):
            weights = weights.t()
        self.layer.weight.data = weights.reshape(self.layer.weight.shape).to(
            self.layer.weight.dtype
        )

        error = torch.sum(losses).item()
        elapsed = time.time() - started_at
        logger.debug(
            "Layer compressed in %.2fs (GPTQ error %.4f)",
            elapsed,
            error,
        )
        self.free()
        return {"error": error, "seconds": elapsed}

    def free(self) -> None:
        self.hessian = None
        self.inputs.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
