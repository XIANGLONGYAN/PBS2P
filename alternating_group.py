import torch
import numpy as np
from utils.mask import generate_structural_mask

def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    return mse

index = 0
errors = []
def high_order_residual(x, mask, order=2):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        binary= torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask
    
    return sum_order


def high_order_residual_alternating_order1(x, mask, order=2, iter=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    # Alternating update
    refine_mean = mean_tensor_all.clone()
    sum_order_alternating = sum_order.clone()

    for k in range(iter):
        # 1. Fix alpha and B, update mean
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        
        # 2. Fix mean and B, update alpha
        new_alpha = 1. / (torch.sum(new_binary * mask * new_binary * mask, dim=1) + 1e-8) * torch.sum(new_binary * mask * (new_matrix - refine_mean[:, None] * mask), dim=1)
        
        # 3. Fix mean and alpha, update B
        new_matrix_expanded = (new_matrix - refine_mean[:, None] * mask).unsqueeze(-1)
        v = torch.stack([-new_alpha, new_alpha], dim=1)
        v_expanded = v.unsqueeze(1)

        differences = torch.abs(new_matrix_expanded - v_expanded)
        min_indices = torch.argmin(differences, dim=-1)

        new_binary = torch.ones_like(min_indices)
        new_binary[min_indices == 0] = -1

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask


    return sum_order_alternating

num_p = 3
for seed in range(40):
    torch.manual_seed(seed)

    minimal_value = float('inf')

    origin_matrix = 3*torch.randn(16,128) + 4.567


    # mask = torch.tensor([
    #     [1,1,1],
    #     [0,0,0]
    # ],dtype=torch.bool)

    mask3 = torch.randint(0, 2, (16, 128)).bool()
    group3 = torch.zeros_like(origin_matrix)

    mask_list = [mask3]
    optimal_split_list = []
    group_list = [group3]
    for i in range(num_p):
        search_matrix = origin_matrix * (~mask3)
        minimal_upper_bound = float('inf')

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

            # # math
            # masked_x_tensor1 = torch.where(mask1, origin_matrix, torch.tensor(float('nan')))
            # mean_tensor_all1 = torch.nanmean(masked_x_tensor1, dim=1)
            # mean_tensor_all1 = torch.where(torch.isnan(mean_tensor_all1), torch.zeros_like(mean_tensor_all1), mean_tensor_all1)

            # masked_x_tensor2 = torch.where(mask2, origin_matrix, torch.tensor(float('nan')))
            # mean_tensor_all2 = torch.nanmean(masked_x_tensor2, dim=1)
            # mean_tensor_all2 = torch.where(torch.isnan(mean_tensor_all2), torch.zeros_like(mean_tensor_all2), mean_tensor_all2)

            # upper_bound = -(mean_tensor_all1**2 + mean_tensor_all2**2).sum()

            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            if quantize_error < minimal_value:
                # print(i, '999999')
                minimal_value = quantize_error
                optimal_split = split_value
                best_mask2 = mask2
                best_mask1 = mask1

            # if upper_bound < minimal_upper_bound:
            #     minimal_upper_bound = upper_bound
            #     optimal_split = split_value
            #     best_mask2 = mask2
            #     best_mask1 = mask1

        optimal_group2 = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)

        mask_list.append(best_mask2)
        optimal_split_list.append(optimal_split)
        group3 = group3 + optimal_group2
        group_list.append(optimal_group2)
        mask3 = mask3 | best_mask2


    optimal_group1 = high_order_residual_alternating_order1(origin_matrix, best_mask1, order=1)
    mask_list.append(best_mask1)
    group_list.append(optimal_group1)

    # new_group = [torch.zeros_like(origin_matrix)]
    # for i in range(1,5):
    #     new_group.append(high_order_residual_alternating_order1(origin_matrix, mask_list[i], order=1))
    # quantize_error = error_computing(origin_matrix, sum(new_group))
    # print('final error0', quantize_error)
    # print(minimal_value)
    minimal_value = error_computing(origin_matrix, sum(group_list))
    # print('final error0', minimal_value)
    minimal_value0 = minimal_value

    # alternating update
    iter = 0
    for _ in range(iter):
        mask3 = mask_list[0] | mask_list[3] | mask_list[4]
        group3 = group_list[0] + group_list[3] + group_list[4]

        flag = False
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
                    # print(i, '999999')
                    minimal_value = quantize_error
                    optimal_split = split_value
                    best_mask2 = mask2
                    best_mask1 = mask1
                    flag = True
                
            if flag:
                mask_list[i+1] = best_mask2
                group_list[i+1] = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)
                mask_list[i+2] = best_mask1

            # accelerate
            mask3 = mask_list[i+0] | mask_list[i+1] | mask_list[(i+4)%5]
            group3 = group_list[i+0] + group_list[i+1] + group_list[(i+4)%5]
            optimal_split_list[i] = optimal_split

        if flag:
            # mask_list[-1] = best_mask1
            group_list[-1] = high_order_residual_alternating_order1(origin_matrix, best_mask1, order=1)


    # print(i, optimal_split)
    # print(i, optimal_split2)
    # if minimal_value < minimal_value0:
        # print('ori', minimal_value0)
        # print(i, 'update error', minimal_value)

        # mask_ratio = []
        # for i in range(len(mask_list)):
        #     mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

        # print(sum(mask_ratio))

    new_group = [torch.zeros_like(origin_matrix)]
    for i in range(1,5):
        new_group.append(high_order_residual_alternating_order1(origin_matrix, mask_list[i], order=1))
    quantize_error = error_computing(origin_matrix, sum(new_group))
    print('final error', quantize_error)

    
    errors.append(quantize_error)
print(f'total: {sum(errors)}')

    # if quantize_error > minimal_value0:
    #     print(seed)

    # quantize_error2 = error_computing(origin_matrix, sum(group_list))
    # print('final error2', quantize_error2)

    # mask_list = [mask3]
    # optimal_split_list = []
    # group_list = [group3]
    # for i in range(num_p):
    #     search_matrix = origin_matrix * (~mask3)

    #     flat_abs_tensor = torch.abs(search_matrix).view(-1)
    #     percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
    #     percentile_values = torch.tensor(
    #         np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
    #     ).to(origin_matrix.device)

    #     # search for the optimal split for the second group, high order=1,, non-structured search
    #     for split_value in percentile_values:
    #         mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
    #         group1 = high_order_residual_alternating_order1(origin_matrix, mask1, order=1)
    #         group2 = high_order_residual_alternating_order1(origin_matrix, mask2, order=1)

    #         quantize_error = error_computing(origin_matrix, group1+group2+group3)
    #         if quantize_error < minimal_value:
    #             minimal_value = quantize_error
    #             optimal_split = split_value
    #             best_mask2 = mask2
    #             best_mask1 = mask1

    #     optimal_group2 = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)

    #     mask_list.append(best_mask2)
    #     optimal_split_list.append(optimal_split)
    #     group3 = group3 + optimal_group2
    #     group_list.append(optimal_group2)
    #     mask3 = mask3 | best_mask2


    # optimal_group1 = high_order_residual_alternating_order1(origin_matrix, best_mask1, order=1)
    # mask_list.append(best_mask1)
    # group_list.append(optimal_group1)

    # # alternating update
    # iter = 0
    # for _ in range(iter):
    #     mask3 = mask_list[0] | mask_list[3] | mask_list[4]
    #     group3 = group_list[0] + group_list[3] + group_list[4]

    #     flag = False
    #     for i in range(num_p):
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
                
    #         if flag:
    #             mask_list[i+1] = best_mask2
    #             group_list[i+1] = high_order_residual_alternating_order1(origin_matrix, best_mask2, order=1)

    #         # accelerate
    #         mask3 = mask_list[i+0] | mask_list[i+1] | mask_list[(i+4)%5]
    #         group3 = group_list[i+0] + group_list[i+1] + group_list[(i+4)%5]
    #         optimal_split_list[i] = optimal_split

    #     if flag:
    #         mask_list[-1] = best_mask1
    #         group_list[-1] = high_order_residual_alternating_order1(origin_matrix, best_mask1, order=1)


    # # print(i, optimal_split)
    # # print(i, optimal_split2)
    # print(i, 'update error', minimal_value)

    # mask_ratio = []
    # for i in range(len(mask_list)):
    #     mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    # print(sum(mask_ratio))

    # new_group = [torch.zeros_like(origin_matrix)]
    # for i in range(1,5):
    #     new_group.append(high_order_residual_alternating_order1(origin_matrix, mask_list[i], order=1))
    # quantize_error = error_computing(origin_matrix, sum(new_group))
    # print('final error', quantize_error)

    # quantize_error2 = error_computing(origin_matrix, sum(group_list))
    # print('final error2', quantize_error2)



# # math
# masked_x_tensor1 = torch.where(mask1, origin_matrix, torch.tensor(float('nan')))
# mean_tensor_all1 = torch.nanmean(masked_x_tensor1, dim=1)
# mean_tensor_all1 = torch.where(torch.isnan(mean_tensor_all1), torch.zeros_like(mean_tensor_all1), mean_tensor_all1)

# masked_x_tensor2 = torch.where(mask2, origin_matrix, torch.tensor(float('nan')))
# mean_tensor_all2 = torch.nanmean(masked_x_tensor2, dim=1)
# mean_tensor_all2 = torch.where(torch.isnan(mean_tensor_all2), torch.zeros_like(mean_tensor_all2), mean_tensor_all2)

# upper_bound = (mean_tensor_all1**2 + mean_tensor_all2**2).sum()