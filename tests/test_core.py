import torch

from pbs2p.cfs import allocate_layer_sparsity
from pbs2p.gptq import build_progressive_nm_masks
from pbs2p.spbo import one_shot_binarize, stepwise_binarize


def test_progressive_nm_masks_are_nested() -> None:
    scores = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    masks = build_progressive_nm_masks(scores, n=2, m=4)

    assert len(masks) == 2
    assert torch.all(masks[1] <= masks[0])
    assert torch.all(masks[-1].reshape(2, 4, 4).sum(dim=-1) == 2)


def test_spbo_respects_final_pruning_mask() -> None:
    torch.manual_seed(0)
    weights = torch.randn(3, 8)
    group_mask = torch.ones_like(weights, dtype=torch.bool)
    scores = torch.randn_like(weights)
    pruning_masks = build_progressive_nm_masks(scores, n=2, m=4)
    covariance_input = torch.randn(8, 8)
    covariance = covariance_input.T @ covariance_input

    stepwise = stepwise_binarize(
        weights,
        group_mask,
        pruning_masks,
        covariance,
        order=2,
        iterations=2,
    )
    one_shot = one_shot_binarize(
        weights,
        group_mask,
        pruning_masks[-1],
        covariance,
        order=2,
        iterations=2,
    )

    assert torch.isfinite(stepwise).all()
    assert torch.isfinite(one_shot).all()
    assert torch.all(stepwise[~pruning_masks[-1]] == 0)
    assert torch.all(one_shot[~pruning_masks[-1]] == 0)


def test_cfs_allocation_preserves_average_sparsity() -> None:
    importance = [0.1, 0.9, 0.4, 0.7, 0.2, 0.8, 0.3, 0.6]
    allocation = allocate_layer_sparsity(importance, target_n=6, m=8)

    assert sum(allocation) == 6 * len(importance)
    assert allocation[1] >= allocation[0]
    assert min(allocation) >= 5
    assert max(allocation) <= 7
