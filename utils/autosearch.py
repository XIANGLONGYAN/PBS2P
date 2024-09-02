from re import L
import numpy as np
from pyparsing import line
import torch
from binary import high_order_residual, db_llm, high_order_residual_search_clip, high_order_residual_refine_accelerate, high_order_residual_refine, high_order_residual_alternating, high_order_residual_refine_mean, high_order_residual_alternating_order1, high_order_residual_alternating_mean
from utils.mask import generate_structural_mask, generate_multi_structural_mask

def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    return mse

def error_computing_x_all(origin_matrix, quantized_matrix, inp):
    # inps shape [128, 2048, 128]
    mse = torch.mean((inp.to(torch.float) @ (origin_matrix - quantized_matrix).T) ** 2)
    return mse

def error_computing_x_mean(origin_matrix, quantized_matrix, inp):
    # inps shape [2048, 128]
    mse = torch.mean((inp.to(torch.float) @ (origin_matrix - quantized_matrix).T) ** 2)
    return mse

def calculate_percentage_and_variance_original(weights, abs_weights, bin_edges):
    percentages = []
    variances = []
    accum_percentages = [0]
    total_elements = abs_weights.numel()
    for i in range(len(bin_edges) - 1):
        bin_mask = (abs_weights >= bin_edges[i]) & (abs_weights < bin_edges[i + 1])
        bin_weights = weights[bin_mask]
        percentages.append(bin_weights.numel() / total_elements * 100)
        accum_percentages.append(accum_percentages[-1] + percentages[-1])
        variances.append(torch.var(bin_weights))
    return percentages, variances, accum_percentages

'''
Include main method to search the rate for 2-bit salient data columns and the optimal split for 1-bit data
'''
def structural_searching(origin_matrix, up_lim=30):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        group3 = high_order_residual(origin_matrix, mask3, order=2)

        group4 = high_order_residual(origin_matrix, ~mask3, order=2)
        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)
        if quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i

    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual(origin_matrix, mask3, order=2)

    search_matrix = origin_matrix * (~mask3)

    flat_abs_tensor = torch.abs(search_matrix).view(-1)
    percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
    percentile_values = torch.tensor(
        np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
    ).to(origin_matrix.device)

    # search for the optimal split for the second group, high order=1,, non-structured search
    for split_value in percentile_values:
        mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
        group1 = high_order_residual(origin_matrix, mask1, order=1)
        group2 = high_order_residual(origin_matrix, mask2, order=1)

        quantize_error = error_computing(origin_matrix, group1+group2+group3)
        if quantize_error < minimal_value:
            minimal_value = quantize_error
            optimal_split = split_value
        tmp = torch.max(torch.abs(search_matrix)).item()
    
    return optimal_split, mask3

def structural_searching_multip(origin_matrix, up_lim=30, num_p=2):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    new_criteria = False
    alpha = 0.7
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        group3 = high_order_residual(origin_matrix, mask3, order=2)

        if new_criteria:
            group4 = high_order_residual(origin_matrix, ~mask3, order=1)
            group5 = high_order_residual(origin_matrix, mask3, order=1)
            quantize_error_order1 = error_computing(origin_matrix, group4+group5)
        else:
            group4 = high_order_residual(origin_matrix, ~mask3, order=2)


        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)

        if new_criteria:
            optimal_split_0 = i
            # if quantize_error_0 < alpha * quantize_error_order1:
            #     break
        elif quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i

    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual(origin_matrix, mask3, order=2)

    mask_list = [mask3]
    optimal_split_list = []
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
            group1 = high_order_residual(origin_matrix, mask1, order=1)
            group2 = high_order_residual(origin_matrix, mask2, order=1)

            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                optimal_group2 = group2
                best_mask2 = mask2
                best_mask1 = mask1
            tmp = torch.max(torch.abs(search_matrix)).item()

        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        mask3 = mask3 | best_mask2

    mask_list.append(best_mask1)

    return optimal_split_list, mask_list

def structural_searching_multip_db(origin_matrix, up_lim=30, num_p=2):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    new_criteria = False
    alpha = 0.7
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        # group3 = high_order_residual(origin_matrix, mask3, order=2)
        group3 = db_llm(origin_matrix, mask3)

        if new_criteria:
            group4 = high_order_residual(origin_matrix, ~mask3, order=1)
            group5 = high_order_residual(origin_matrix, mask3, order=1)
            quantize_error_order1 = error_computing(origin_matrix, group4+group5)
        else:
            # group4 = high_order_residual(origin_matrix, ~mask3, order=2)
            group4 = db_llm(origin_matrix, ~mask3)


        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)

        if new_criteria:
            optimal_split_0 = i
            # if quantize_error_0 < alpha * quantize_error_order1:
            #     break
        elif quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i

    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    # group3 = high_order_residual(origin_matrix, mask3, order=2)
    group3 = db_llm(origin_matrix, mask3)

    mask_list = [mask3]
    optimal_split_list = []
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
            group1 = high_order_residual(origin_matrix, mask1, order=1)
            group2 = high_order_residual(origin_matrix, mask2, order=1)

            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                optimal_group2 = group2
                best_mask2 = mask2
                best_mask1 = mask1
            tmp = torch.max(torch.abs(search_matrix)).item()

        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        mask3 = mask3 | best_mask2

    mask_list.append(best_mask1)

    return optimal_split_list, mask_list

def structural_searching_multip_search_clip(origin_matrix, up_lim=30, num_p=2):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    new_criteria = False
    alpha = 0.7
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        group3 = high_order_residual(origin_matrix, mask3, order=2)

        if new_criteria:
            group4 = high_order_residual(origin_matrix, ~mask3, order=1)
            group5 = high_order_residual(origin_matrix, mask3, order=1)
            quantize_error_order1 = error_computing(origin_matrix, group4+group5)
        else:
            group4 = high_order_residual(origin_matrix, ~mask3, order=2)


        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)

        if new_criteria:
            optimal_split_0 = i
            # if quantize_error_0 < alpha * quantize_error_order1:
            #     break
        elif quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i

    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual(origin_matrix, mask3, order=2)

    mask_list = [mask3]
    k_list = [0]
    optimal_split_list = []
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
            for k in range(-10, 10):
                group1 = high_order_residual_search_clip(origin_matrix, mask1, order=1, k=k)
                group2 = high_order_residual_search_clip(origin_matrix, mask2, order=1, k=k)

                quantize_error = error_computing(origin_matrix, group1+group2+group3)
                if quantize_error < minimal_value:
                    minimal_value = quantize_error
                    optimal_split = split_value
                    optimal_group2 = group2
                    best_mask2 = mask2
                    best_mask1 = mask1
                    best_k = k
                tmp = torch.max(torch.abs(search_matrix)).item()

        print(f'best k: {best_k}')

        mask_list.append(best_mask2)
        k_list.append(best_k)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        mask3 = mask3 | best_mask2

    mask_list.append(best_mask1)
    k_list.append(best_k)

    return optimal_split_list, mask_list, k_list

def structural_searching_multip_refine(origin_matrix, up_lim=30, num_p=2):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    new_criteria = False
    alpha = 0.7
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        # group3 = high_order_residual_refine(origin_matrix, mask3, order=2)
        group3 = high_order_residual(origin_matrix, mask3, order=2)   # for fair comparison and accelerate

        if new_criteria:
            group4 = high_order_residual(origin_matrix, ~mask3, order=1)
            group5 = high_order_residual(origin_matrix, mask3, order=1)
            quantize_error_order1 = error_computing(origin_matrix, group4+group5)
        else:
            # group4 = high_order_residual_refine(origin_matrix, ~mask3, order=2)
            group4 = high_order_residual(origin_matrix, ~mask3, order=2)   # for fair comparison and accelerate


        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)

        if new_criteria:
            optimal_split_0 = i
            # if quantize_error_0 < alpha * quantize_error_order1:
            #     break
        elif quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i


    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual_refine(origin_matrix, mask3, order=2)

    mask_list = [mask3]
    optimal_split_list = []
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
            group1 = high_order_residual(origin_matrix, mask1, order=1)
            group2 = high_order_residual(origin_matrix, mask2, order=1)

            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                optimal_group2 = group2
                best_mask2 = mask2
                best_mask1 = mask1
            tmp = torch.max(torch.abs(search_matrix)).item()

        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        mask3 = mask3 | best_mask2

    mask_list.append(best_mask1)

    return optimal_split_list, mask_list

def structural_searching_multip_refine_accelerate(origin_matrix, up_lim=30, num_p=2):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    new_criteria = False
    alpha = 0.7
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        # group3 = high_order_residual_refine_accelerate(origin_matrix, mask3, order=2)
        group3 = high_order_residual(origin_matrix, mask3, order=2)   # for fair comparison and accelerate

        if new_criteria:
            group4 = high_order_residual(origin_matrix, ~mask3, order=1)
            group5 = high_order_residual(origin_matrix, mask3, order=1)
            quantize_error_order1 = error_computing(origin_matrix, group4+group5)
        else:
            # group4 = high_order_residual_refine_accelerate(origin_matrix, ~mask3, order=2)
            group4 = high_order_residual(origin_matrix, ~mask3, order=2)   # for fair comparison and accelerate


        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)

        if new_criteria:
            optimal_split_0 = i
            # if quantize_error_0 < alpha * quantize_error_order1:
            #     break
        elif quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i

    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual_refine_accelerate(origin_matrix, mask3, order=2)

    mask_list = [mask3]
    optimal_split_list = []
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
            group1 = high_order_residual(origin_matrix, mask1, order=1)
            group2 = high_order_residual(origin_matrix, mask2, order=1)

            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                optimal_group2 = group2
                best_mask2 = mask2
                best_mask1 = mask1
            tmp = torch.max(torch.abs(search_matrix)).item()

        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        mask3 = mask3 | best_mask2

    mask_list.append(best_mask1)

    return optimal_split_list, mask_list

def structural_searching_multip_alternating(origin_matrix, up_lim=30, num_p=2):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    new_criteria = False
    alpha = 0.7
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        # group3 = high_order_residual_alternating(origin_matrix, mask3, order=2)
        group3 = high_order_residual(origin_matrix, mask3, order=2)   # for fair comparison and accelerate

        if new_criteria:
            group4 = high_order_residual(origin_matrix, ~mask3, order=1)
            group5 = high_order_residual(origin_matrix, mask3, order=1)
            quantize_error_order1 = error_computing(origin_matrix, group4+group5)
        else:
            # group4 = high_order_residual_alternating(origin_matrix, ~mask3, order=2)
            group4 = high_order_residual(origin_matrix, ~mask3, order=2)   # for fair comparison and accelerate


        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)

        if new_criteria:
            optimal_split_0 = i
            # if quantize_error_0 < alpha * quantize_error_order1:
            #     break
        elif quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i


    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual_alternating(origin_matrix, mask3, order=2)

    mask_list = [mask3]
    optimal_split_list = []
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
            group1 = high_order_residual(origin_matrix, mask1, order=1)
            group2 = high_order_residual(origin_matrix, mask2, order=1)

            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                optimal_group2 = group2
                best_mask2 = mask2
                best_mask1 = mask1
            tmp = torch.max(torch.abs(search_matrix)).item()

        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        mask3 = mask3 | best_mask2

    mask_list.append(best_mask1)

    return optimal_split_list, mask_list

def structural_searching_multip_alternating_refine_mean(origin_matrix, up_lim=30, num_p=2):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    new_criteria = False
    alpha = 0.7
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        # group3 = high_order_residual_alternating(origin_matrix, mask3, order=2)
        group3 = high_order_residual(origin_matrix, mask3, order=2)   # for fair comparison and accelerate

        if new_criteria:
            group4 = high_order_residual_refine_mean(origin_matrix, ~mask3, order=1)
            group5 = high_order_residual_refine_mean(origin_matrix, mask3, order=1)
            quantize_error_order1 = error_computing(origin_matrix, group4+group5)
        else:
            # group4 = high_order_residual_alternating(origin_matrix, ~mask3, order=2)
            group4 = high_order_residual(origin_matrix, ~mask3, order=2)   # for fair comparison and accelerate


        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)

        if new_criteria:
            optimal_split_0 = i
            # if quantize_error_0 < alpha * quantize_error_order1:
            #     break
        elif quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i


    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual_alternating(origin_matrix, mask3, order=2)

    mask_list = [mask3]
    optimal_split_list = []
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
            group1 = high_order_residual_refine_mean(origin_matrix, mask1, order=1)
            group2 = high_order_residual_refine_mean(origin_matrix, mask2, order=1)

            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                optimal_group2 = group2
                best_mask2 = mask2
                best_mask1 = mask1
            tmp = torch.max(torch.abs(search_matrix)).item()

        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        mask3 = mask3 | best_mask2

    mask_list.append(best_mask1)

    return optimal_split_list, mask_list

def structural_searching_multip_alternating_alternating_order1(origin_matrix, up_lim=30, num_p=2):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    new_criteria = False
    alpha = 0.7
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        # group3 = high_order_residual_alternating(origin_matrix, mask3, order=2)
        group3 = high_order_residual(origin_matrix, mask3, order=2)   # for fair comparison and accelerate

        if new_criteria:
            group4 = high_order_residual_alternating_order1(origin_matrix, ~mask3, order=1)
            group5 = high_order_residual_alternating_order1(origin_matrix, mask3, order=1)
            quantize_error_order1 = error_computing(origin_matrix, group4+group5)
        else:
            # group4 = high_order_residual_alternating(origin_matrix, ~mask3, order=2)
            group4 = high_order_residual(origin_matrix, ~mask3, order=2)   # for fair comparison and accelerate


        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)

        if new_criteria:
            optimal_split_0 = i
            # if quantize_error_0 < alpha * quantize_error_order1:
            #     break
        elif quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i


    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual_alternating_mean(origin_matrix, mask3, order=2)

    mask_list = [mask3]
    optimal_split_list = []
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
            group1 = high_order_residual_alternating_order1(origin_matrix, mask1, order=1)
            group2 = high_order_residual_alternating_order1(origin_matrix, mask2, order=1)

            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                optimal_group2 = group2
                best_mask2 = mask2
                best_mask1 = mask1
            tmp = torch.max(torch.abs(search_matrix)).item()

        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        mask3 = mask3 | best_mask2

    mask_list.append(best_mask1)

    return optimal_split_list, mask_list

def structural_searching_multip_alternating_alternating_order1_group(origin_matrix, up_lim=30, num_p=2, iter=0):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    new_criteria = False
    alpha = 0.7
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        # group3 = high_order_residual_alternating(origin_matrix, mask3, order=2)
        group3 = high_order_residual(origin_matrix, mask3, order=2)   # for fair comparison and accelerate

        if new_criteria:
            group4 = high_order_residual_alternating_order1(origin_matrix, ~mask3, order=1)
            group5 = high_order_residual_alternating_order1(origin_matrix, mask3, order=1)
            quantize_error_order1 = error_computing(origin_matrix, group4+group5)
        else:
            # group4 = high_order_residual_alternating(origin_matrix, ~mask3, order=2)
            group4 = high_order_residual(origin_matrix, ~mask3, order=2)   # for fair comparison and accelerate


        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)

        if new_criteria:
            optimal_split_0 = i
            # if quantize_error_0 < alpha * quantize_error_order1:
            #     break
        elif quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i


    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual_alternating_mean(origin_matrix, mask3, order=2)

    mask_list = [mask3]
    optimal_split_list = []
    group_list = [group3]
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
            group1 = high_order_residual(origin_matrix, mask1, order=1)
            group2 = high_order_residual(origin_matrix, mask2, order=1)

            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                best_mask2 = mask2
                best_mask1 = mask1

        optimal_group2 = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)

        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        group_list.append(optimal_group2)
        mask3 = mask3 | best_mask2

    optimal_group1 = high_order_residual_alternating_order1(origin_matrix, best_mask1, order=1)
    mask_list.append(best_mask1)
    group_list.append(optimal_group1)

    # update criteria
    # torch.set_printoptions(precision=10)
    # print('old', minimal_value)
    # new_group = [high_order_residual_alternating_mean(origin_matrix, mask_list[0], order=2)]
    # for i in range(1,5):
    #     new_group.append(high_order_residual_alternating_order1(origin_matrix, mask_list[i], order=1))
    # minimal_value = error_computing(origin_matrix, sum(new_group))
    # minimal_value = error_computing(origin_matrix, sum(group_list))
    # print('new', minimal_value)

    # alternating update
    for _ in range(iter):
        mask3 = mask_list[0] | mask_list[3] | mask_list[4]
        group3 = group_list[0] + group_list[3] + group_list[4]

        for i in range(num_p):
            search_matrix = origin_matrix * (~mask3)

            flat_abs_tensor = torch.abs(search_matrix).view(-1)
            percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
            percentile_values = torch.tensor(
                np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
            ).to(origin_matrix.device)

            # search for the optimal split for the second group, high order=1,, non-structured search
            flag = False
            for split_value in percentile_values:
                mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
                group1 = high_order_residual(origin_matrix, mask1, order=1)
                group2 = high_order_residual(origin_matrix, mask2, order=1)

                quantize_error = error_computing(origin_matrix, group1+group2+group3)
                if quantize_error < minimal_value:
                    minimal_value = quantize_error
                    optimal_split = split_value
                    best_mask2 = mask2
                    best_mask1 = mask1
                    flag = True
                    # print('debug', quantize_error)
                
            if flag:
                mask_list[i+1] = best_mask2
                group_list[i+1] = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)
                mask_list[i+2] = best_mask1

            # accelerate
            mask3 = mask_list[i+0] | mask_list[i+1] | mask_list[(i+4)%5]
            group3 = group_list[i+0] + group_list[i+1] + group_list[(i+4)%5]
            optimal_split_list[i] = optimal_split

    # new_group = [high_order_residual_alternating_mean(origin_matrix, mask_list[0], order=2)]
    # for i in range(1,5):
    #     new_group.append(high_order_residual_alternating_order1(origin_matrix, mask_list[i], order=1))
    # quantize_error = error_computing(origin_matrix, sum(new_group))
    # print('update', quantize_error)

    return optimal_split_list, mask_list

def structural_searching_multip_alternating_alternating_order1_group_math(origin_matrix, up_lim=30, num_p=2, iter=0):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    new_criteria = False
    alpha = 0.7
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        # group3 = high_order_residual_alternating(origin_matrix, mask3, order=2)
        group3 = high_order_residual(origin_matrix, mask3, order=2)   # for fair comparison and accelerate

        if new_criteria:
            group4 = high_order_residual_alternating_order1(origin_matrix, ~mask3, order=1)
            group5 = high_order_residual_alternating_order1(origin_matrix, mask3, order=1)
            quantize_error_order1 = error_computing(origin_matrix, group4+group5)
        else:
            # group4 = high_order_residual_alternating(origin_matrix, ~mask3, order=2)
            group4 = high_order_residual(origin_matrix, ~mask3, order=2)   # for fair comparison and accelerate


        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)

        if new_criteria:
            optimal_split_0 = i
            # if quantize_error_0 < alpha * quantize_error_order1:
            #     break
        elif quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i


    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual_alternating_mean(origin_matrix, mask3, order=2)

    mask_list = [mask3]
    optimal_split_list = []
    group_list = [group3]

    minimal_upper_bound = float('inf')
    search_matrix = origin_matrix * (~mask3)

    flat_abs_tensor = torch.abs(search_matrix).view(-1)
    percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
    percentile_values = torch.tensor(
        np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
    ).to(origin_matrix.device)

    # search for the optimal split for the second group, high order=1,, non-structured search
    split_combinations = torch.combinations(percentile_values, r=3, with_replacement=False)
    for splits in split_combinations:
        split1, split2, split3 = splits[0], splits[1], splits[2]
        mask1, mask2, mask3, mask4 = generate_multi_structural_mask(origin_matrix, mask3, split1, split2, split3)
        # group1 = high_order_residual(origin_matrix, mask1, order=1)
        # group2 = high_order_residual(origin_matrix, mask2, order=1)

        # quantize_error = error_computing(origin_matrix, group1+group2+group3)
        # if quantize_error < minimal_value:
        #     minimal_value = quantize_error
        #     optimal_split = split_value
        #     best_mask2 = mask2
        #     best_mask1 = mask1

        # math
        masked_x_tensor1 = torch.where(mask1, origin_matrix, torch.tensor(float('nan')))
        mean_tensor_all1 = torch.nanmean(masked_x_tensor1, dim=1)
        mean_tensor_all1 = torch.where(torch.isnan(mean_tensor_all1), torch.zeros_like(mean_tensor_all1), mean_tensor_all1)

        masked_x_tensor2 = torch.where(mask2, origin_matrix, torch.tensor(float('nan')))
        mean_tensor_all2 = torch.nanmean(masked_x_tensor2, dim=1)
        mean_tensor_all2 = torch.where(torch.isnan(mean_tensor_all2), torch.zeros_like(mean_tensor_all2), mean_tensor_all2)

        masked_x_tensor3 = torch.where(mask3, origin_matrix, torch.tensor(float('nan')))
        mean_tensor_all3 = torch.nanmean(masked_x_tensor3, dim=1)
        mean_tensor_all3 = torch.where(torch.isnan(mean_tensor_all3), torch.zeros_like(mean_tensor_all3), mean_tensor_all3)

        masked_x_tensor4 = torch.where(mask4, origin_matrix, torch.tensor(float('nan')))
        mean_tensor_all4 = torch.nanmean(masked_x_tensor4, dim=1)
        mean_tensor_all4 = torch.where(torch.isnan(mean_tensor_all4), torch.zeros_like(mean_tensor_all4), mean_tensor_all4)

        upper_bound = -(mean_tensor_all1**2 + mean_tensor_all2**2 + mean_tensor_all3**2 + mean_tensor_all4**2).sum()

        if upper_bound < minimal_upper_bound:
            minimal_upper_bound = upper_bound
            # optimal_split = split_value
            best_mask4 = mask4
            best_mask3 = mask3
            best_mask2 = mask2
            best_mask1 = mask1

        # optimal_group2 = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)

        # mask_list.append(best_mask2)
        # optimal_split_list.append(optimal_split)
        # group3 = group3 + optimal_group2
        # group_list.append(optimal_group2)
        # mask3 = mask3 | best_mask2

    # optimal_group1 = high_order_residual_alternating_order1(origin_matrix, best_mask1, order=1)
    mask_list.append(best_mask4)
    mask_list.append(best_mask3)
    mask_list.append(best_mask2)
    mask_list.append(best_mask1)
    # group_list.append(optimal_group1)

    # # update criteria
    # # torch.set_printoptions(precision=10)
    # # print('old', minimal_value)
    # # new_group = [high_order_residual_alternating_mean(origin_matrix, mask_list[0], order=2)]
    # # for i in range(1,5):
    # #     new_group.append(high_order_residual_alternating_order1(origin_matrix, mask_list[i], order=1))
    # # minimal_value = error_computing(origin_matrix, sum(new_group))
    # minimal_value = error_computing(origin_matrix, sum(group_list))
    # # print('new', minimal_value)

    # # alternating update
    # for _ in range(iter):
    #     mask3 = mask_list[0] | mask_list[3] | mask_list[4]
    #     group3 = group_list[0] + group_list[3] + group_list[4]

    #     for i in range(num_p):
    #         minimal_upper_bound = float('inf')
    #         search_matrix = origin_matrix * (~mask3)

    #         flat_abs_tensor = torch.abs(search_matrix).view(-1)
    #         percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
    #         percentile_values = torch.tensor(
    #             np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
    #         ).to(origin_matrix.device)

    #         # search for the optimal split for the second group, high order=1,, non-structured search
    #         flag = False
    #         for split_value in percentile_values:
    #             mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
    #             group1 = high_order_residual(origin_matrix, mask1, order=1)
    #             group2 = high_order_residual(origin_matrix, mask2, order=1)

    #             quantize_error = error_computing(origin_matrix, group1+group2+group3)
    #             if quantize_error < minimal_value:
    #                 minimal_value = quantize_error
    #                 optimal_split = split_value
    #                 best_mask2 = mask2
    #                 best_mask1 = mask1
    #                 flag = True
    #                 # print('debug', quantize_error)
                
    #         if flag:
    #             mask_list[i+1] = best_mask2
    #             group_list[i+1] = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)
    #             mask_list[i+2] = best_mask1

    #         # accelerate
    #         mask3 = mask_list[i+0] | mask_list[i+1] | mask_list[(i+4)%5]
    #         group3 = group_list[i+0] + group_list[i+1] + group_list[(i+4)%5]
    #         optimal_split_list[i] = optimal_split

    # # new_group = [high_order_residual_alternating_mean(origin_matrix, mask_list[0], order=2)]
    # # for i in range(1,5):
    # #     new_group.append(high_order_residual_alternating_order1(origin_matrix, mask_list[i], order=1))
    # # quantize_error = error_computing(origin_matrix, sum(new_group))
    # # print('update', quantize_error)

    return optimal_split_list, mask_list

def structural_searching_multip_alternating_alternating_order1_group_x(origin_matrix, up_lim=30, num_p=2, inp=None, iter=0):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    new_criteria = False
    alpha = 0.7
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        # group3 = high_order_residual_alternating(origin_matrix, mask3, order=2)
        group3 = high_order_residual(origin_matrix, mask3, order=2)   # for fair comparison and accelerate

        if new_criteria:
            group4 = high_order_residual_alternating_order1(origin_matrix, ~mask3, order=1)
            group5 = high_order_residual_alternating_order1(origin_matrix, mask3, order=1)
            quantize_error_order1 = error_computing(origin_matrix, group4+group5, inp)
        else:
            # group4 = high_order_residual_alternating(origin_matrix, ~mask3, order=2)
            group4 = high_order_residual(origin_matrix, ~mask3, order=2)   # for fair comparison and accelerate


        quantize_error_0 = error_computing(origin_matrix, group4+group3, inp)
        error.append(quantize_error_0.item())
        lines.append(i)

        if new_criteria:
            optimal_split_0 = i
            # if quantize_error_0 < alpha * quantize_error_order1:
            #     break
        elif quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i


    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual_alternating_mean(origin_matrix, mask3, order=2)

    mask_list = [mask3]
    optimal_split_list = []
    group_list = [group3]
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)

        flat_abs_tensor = torch.abs(search_matrix).view(-1)
        percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
        percentile_values = torch.tensor(
            np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
        ).to(origin_matrix.device)

        # search for the optimal split for the second group, high order=1,, non-structured search
        for split_value in percentile_values:
            mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
            group1 = high_order_residual(origin_matrix, mask1, order=1)
            group2 = high_order_residual(origin_matrix, mask2, order=1)

            quantize_error = error_computing(origin_matrix, group1+group2+group3, inp)
            if quantize_error < minimal_value:
                minimal_value = quantize_error
                optimal_split = split_value
                best_mask2 = mask2
                best_mask1 = mask1

        optimal_group2 = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)

        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        group_list.append(optimal_group2)
        mask3 = mask3 | best_mask2

    optimal_group1 = high_order_residual_alternating_order1(origin_matrix, best_mask1, order=1)
    mask_list.append(best_mask1)
    group_list.append(optimal_group1)

    # update criteria
    # torch.set_printoptions(precision=10)
    # print('old', minimal_value)
    # new_group = [high_order_residual_alternating_mean(origin_matrix, mask_list[0], order=2)]
    # for i in range(1,5):
    #     new_group.append(high_order_residual_alternating_order1(origin_matrix, mask_list[i], order=1))
    # minimal_value = error_computing(origin_matrix, sum(new_group))
    # minimal_value = error_computing(origin_matrix, sum(group_list))
    # print('new', minimal_value)

    # alternating update
    for _ in range(iter):
        mask3 = mask_list[0] | mask_list[3] | mask_list[4]
        group3 = group_list[0] + group_list[3] + group_list[4]

        for i in range(num_p):
            search_matrix = origin_matrix * (~mask3)

            flat_abs_tensor = torch.abs(search_matrix).view(-1)
            percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
            percentile_values = torch.tensor(
                np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
            ).to(origin_matrix.device)

            # search for the optimal split for the second group, high order=1,, non-structured search
            flag = False
            for split_value in percentile_values:
                mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
                group1 = high_order_residual(origin_matrix, mask1, order=1)
                group2 = high_order_residual(origin_matrix, mask2, order=1)

                quantize_error = error_computing(origin_matrix, group1+group2+group3)
                if quantize_error < minimal_value:
                    minimal_value = quantize_error
                    optimal_split = split_value
                    best_mask2 = mask2
                    best_mask1 = mask1
                    flag = True
                    # print('debug', quantize_error)
                
            if flag:
                mask_list[i+1] = best_mask2
                group_list[i+1] = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)
                mask_list[i+2] = best_mask1

            # accelerate
            mask3 = mask_list[i+0] | mask_list[i+1] | mask_list[(i+4)%5]
            group3 = group_list[i+0] + group_list[i+1] + group_list[(i+4)%5]
            optimal_split_list[i] = optimal_split

    # new_group = [high_order_residual_alternating_mean(origin_matrix, mask_list[0], order=2)]
    # for i in range(1,5):
    #     new_group.append(high_order_residual_alternating_order1(origin_matrix, mask_list[i], order=1))
    # quantize_error = error_computing(origin_matrix, sum(new_group))
    # print('update', quantize_error)

    return optimal_split_list, mask_list


def find_optimal_split(group_max, origin_matrix, border):
    optimal_split = None
    minimal_value = float('inf')
    searching_steps = torch.arange(0.1,0.8,0.01)
    searching_steps = searching_steps * group_max

    group3 = high_order_residual(origin_matrix, torch.abs(origin_matrix) > border, order=2)
    for split_value in searching_steps:

        group1 = high_order_residual(origin_matrix, (torch.abs(origin_matrix) > split_value) & (torch.abs(origin_matrix) <= border), order=1)
        group2 = high_order_residual(origin_matrix, torch.abs(origin_matrix) <= split_value, order=1)

        quantize_error = error_computing(origin_matrix, group1+group2+group3)
        if quantize_error < minimal_value:
            minimal_value = quantize_error
            optimal_split = split_value

    return optimal_split, minimal_value
