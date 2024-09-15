import torch
from BiLLM.utils.autosearch import structural_searching, structural_searching_multip, structural_searching_multip_db, structural_searching_multip_search_clip, structural_searching_multip_refine, structural_searching_multip_refine_accelerate, structural_searching_multip_alternating, structural_searching_multip_alternating_refine_mean, structural_searching_multip_alternating_alternating_order1, structural_searching_multip_alternating_alternating_order1_group, structural_searching_multip_alternating_alternating_order1_group_math, structural_searching_multip_alternating_alternating_order1_group_x, structural_searching_multip_alternating_alternating_order1_group_percentile, structural_searching_multip_alternating_alternating_order1_group_xx, structural_searching_multip_alternating_alternating_order2_x, structural_searching_multip_alternating_alternating_order1_group_x_columnwise, structural_searching_multip_alternating_alternating_order1_group1and2_x
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


def structural_guassian_distribution_multip(tmp, H=None, metric="magnitude", up_lim=30, num_p=2):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split_list, mask_list = structural_searching_multip(target_weights, up_lim, num_p)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list

def structural_guassian_distribution_multip_db(tmp, H=None, metric="magnitude", up_lim=30, num_p=2):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split_list, mask_list = structural_searching_multip_db(target_weights, up_lim, num_p)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list

def structural_guassian_distribution_multip_search_clip(tmp, H=None, metric="magnitude", up_lim=30, num_p=2):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split_list, mask_list, k_list = structural_searching_multip_search_clip(target_weights, up_lim, num_p)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return k_list, mask_list

def structural_guassian_distribution_multip_refine(tmp, H=None, metric="magnitude", up_lim=30, num_p=2):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split_list, mask_list = structural_searching_multip_refine(target_weights, up_lim, num_p)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list

def structural_guassian_distribution_multip_refine_accelerate(tmp, H=None, metric="magnitude", up_lim=30, num_p=2):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split_list, mask_list = structural_searching_multip_refine_accelerate(target_weights, up_lim, num_p)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list

def structural_guassian_distribution_multip_alternating(tmp, H=None, metric="magnitude", up_lim=30, num_p=2):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split_list, mask_list = structural_searching_multip_alternating(target_weights, up_lim, num_p)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list

def structural_guassian_distribution_multip_alternating_refine_mean(tmp, H=None, metric="magnitude", up_lim=30, num_p=2):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split_list, mask_list = structural_searching_multip_alternating_refine_mean(target_weights, up_lim, num_p)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list

def structural_guassian_distribution_multip_alternating_alternating_order1(tmp, H=None, metric="magnitude", up_lim=30, num_p=2):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split_list, mask_list = structural_searching_multip_alternating_alternating_order1(target_weights, up_lim, num_p)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list

def structural_guassian_distribution_multip_alternating_alternating_order1_group(tmp, H=None, metric="magnitude", up_lim=30, num_p=2):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    # optimal_split_list, mask_list = structural_searching_multip_alternating_alternating_order1_group(target_weights, up_lim, num_p)
    optimal_split_list, mask_list = structural_searching_multip_alternating_alternating_order1_group_percentile(target_weights, up_lim, num_p)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list

def structural_guassian_distribution_multip_alternating_alternating_order1_group_math(tmp, H=None, metric="magnitude", up_lim=30, num_p=2):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split_list, mask_list = structural_searching_multip_alternating_alternating_order1_group_math(target_weights, up_lim, num_p)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list

def structural_guassian_distribution_multip_alternating_alternating_order1_group_x(tmp, H=None, metric="magnitude", up_lim=30, num_p=2, inp=None):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    # print(f'debug', inp)
    optimal_split_list, mask_list = structural_searching_multip_alternating_alternating_order1_group_x(target_weights, up_lim, num_p, inp)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list

def structural_guassian_distribution_multip_alternating_alternating_order1_group1and2_x(tmp, H=None, metric="magnitude", up_lim=30, num_p=2, inp=None):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    # print(f'debug', inp)
    optimal_split_list, mask_list = structural_searching_multip_alternating_alternating_order1_group1and2_x(target_weights, up_lim, num_p, inp)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list


def structural_guassian_distribution_multip_alternating_alternating_order1_group_x_columnwise(tmp, H=None, metric="magnitude", up_lim=30, num_p_c=2, num_p=2, inp=None):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    # print(f'debug', inp)
    optimal_split_list, mask_list = structural_searching_multip_alternating_alternating_order1_group_x_columnwise(target_weights, up_lim, num_p_c, num_p, inp)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list


def structural_guassian_distribution_multip_alternating_alternating_order2_x(tmp, H=None, metric="magnitude", up_lim=30, num_p=2, inp=None):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    # print(f'debug', inp)
    optimal_split_list, mask_list = structural_searching_multip_alternating_alternating_order2_x(target_weights, up_lim, num_p, inp)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list


def structural_guassian_distribution_multip_alternating_alternating_order1_group_xx(tmp, H=None, metric="magnitude", up_lim=30, num_p=2, S_hat=None, S_hathat=None):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    # print(f'debug', inp)
    optimal_split_list, mask_list = structural_searching_multip_alternating_alternating_order1_group_xx(target_weights, up_lim, num_p, S_hat, S_hathat)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    ratios_info = ", ".join([f"mask{idx+1} ratio: {ratio:.2f}" for idx, ratio in enumerate(mask_ratio)])
    logger.info(ratios_info)

    return mask_list
