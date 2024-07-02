import torch
from utils.autosearch import structural_searching
from utils.mask import generate_structural_mask

import logging
logger = logging.getLogger()

'''
Used to generate masks for minor structural 2-bit salient data and split major 1-bit normal data according to different metric.
'''
def structural_guassian_distribution(tmp, H=None, metric="magnitude", up_lim=30):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split, mask3 = structural_searching(target_weights, up_lim)
    mask1, mask2 = generate_structural_mask(target_weights, mask3, optimal_split)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask1_ratio = mask1.sum() / mask1.numel()
    mask2_ratio = mask2.sum() / mask2.numel()
    mask3_ratio = mask3.sum() / mask3.numel()
    logger.info(f"mask1 ratio: {mask1_ratio:.2f}, mask2 ratio: {mask2_ratio:.2f}, mask3 ratio: {mask3_ratio:.2f}")

    return mask1, mask2, mask3
