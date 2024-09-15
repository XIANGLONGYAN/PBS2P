import torch

def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    return mse

index = 0

@torch.no_grad()
def high_order_residual_alternating_order1_x(x, mask, order=2, S=None, iter=15, iter2=15):
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
    new_alpha = scale_tensor_all.clone()

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
        new_binary = torch.sign(new_matrix - refine_mean[:, None] * mask)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    # return sum_order
    return sum_order, sum_order_alternating

def high_order_residual_alternating_order1_x_rc(x, mask, order=2, S=None, iter=15, iter2=20):
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

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        # # alpha column
        # scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        # scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()

        # alpha column
        alpha_r_B = scale_tensor_all_r[:, None] * new_binary * mask
        scale_tensor_all_c = torch.sum(alpha_r_B * (new_matrix - mean_tensor_all[:, None] * mask), dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    # Alternating update
    refine_mean = mean_tensor_all.clone()
    sum_order_alternating = sum_order.clone()
    new_alpha_r = scale_tensor_all_r.clone()
    new_alpha_c = scale_tensor_all_c.clone()

    for k in range(iter):
        # 1. Fix alpha row, alpha column, and B, update mean
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        
        # 2-1. Fix mean, alpha column, and B, update alpha row
        W_tilde = new_matrix - refine_mean[:, None] * mask
        alpha_c_B = new_alpha_c[None, :] * new_binary * mask
        new_alpha_r = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)
        
        # 2-2. Fix mean, alpha row, and B, update alpha column
        alpha_r_B = new_alpha_r[:, None] * new_binary * mask
        new_alpha_c = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # 3. Fix mean and alpha, update B
        new_binary = torch.sign(W_tilde)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary + refine_mean[:, None]) * mask

    # MM = mask[:, :, None] * mask[:, None, :]
    # refine_mean_den = torch.sum(S * MM, dim=(1,2)) + 1e-10
    # masked_B = new_alpha_c[None, :] * new_binary * mask
    # BB = masked_B[:, :, None] * masked_B[:, None, :]
    # new_alpha_den = torch.sum(S * BB, dim=(1,2)) + 1e-10
    # diag_S = torch.diag(S)
    # gamma = 1
    # for kk in range(iter2):
    #     # X error update mean
    #     R = new_matrix - new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary * mask
    #     refine_mean = torch.sum(S * R[:, :, None] * MM, dim=(1,2)) / refine_mean_den

    #     # X error update alpha row
    #     masked_W_mu = new_matrix - refine_mean[:, None] * mask
    #     BW = masked_B[:, :, None] * masked_W_mu[:, None, :]
    #     new_alpha_r = torch.sum(S * BW, dim=(1,2)) / new_alpha_den

    # # X error update alpha column (newton)
    # alpha_r_B = new_alpha_r[:, None] * new_binary * mask
    # new_alpha_c -= gamma * torch.sum(S * torch.sum(alpha_r_B[:, :, None] * (new_alpha_c[None, :] * alpha_r_B - masked_W_mu)[:, None, :], dim=0), dim=1) / (diag_S * torch.sum(alpha_r_B**2, dim=0) + 1e-10)
    # gamma *= 0.9

    # sum_order_alternating2 = torch.zeros_like(x) + (new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order, sum_order_alternating


@torch.no_grad()
def high_order_residual_alternating_order1_x_rc2(x, mask, order=2, S=None, iter=15, iter2=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # mean row
        mean_tensor_all_r = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all_r = torch.where(torch.isnan(mean_tensor_all_r), torch.zeros_like(mean_tensor_all_r), mean_tensor_all_r)
        masked_x_tensor -= mean_tensor_all_r[:, None]
        # mean column
        mean_tensor_all_c = torch.nanmean(masked_x_tensor, dim=0)
        mean_tensor_all_c = torch.where(torch.isnan(mean_tensor_all_c), torch.zeros_like(mean_tensor_all_c), mean_tensor_all_c)
        masked_x_tensor -= mean_tensor_all_c[None, :]

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        binary += mean_tensor_all_r[:, None] + mean_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    # Alternating update
    refine_mean_r = mean_tensor_all_r.clone()
    refine_mean_c = mean_tensor_all_c.clone()
    sum_order_alternating = sum_order.clone()
    new_alpha_r = scale_tensor_all_r.clone()
    new_alpha_c = scale_tensor_all_c.clone()

    for k in range(iter):
        # 1-1. Fix alpha row, alpha column, and B, update mean row
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all_r = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all_r = torch.where(torch.isnan(mean_tensor_all_r), torch.zeros_like(mean_tensor_all_r), mean_tensor_all_r)
        refine_mean_r += mean_tensor_all_r.clone()
        # 1-2. Fix alpha row, alpha column, and B, update mean column
        mean_tensor_all_c = torch.nanmean(masked_x_tensor - mean_tensor_all_r[:, None], dim=0)
        mean_tensor_all_c = torch.where(torch.isnan(mean_tensor_all_c), torch.zeros_like(mean_tensor_all_c), mean_tensor_all_c)
        refine_mean_c += mean_tensor_all_c.clone()
        
        # 2-1. Fix mean, alpha column, and B, update alpha row
        W_tilde = new_matrix - (refine_mean_r[:, None] + refine_mean_c[None, :]) * mask
        alpha_c_B = new_alpha_c[None, :] * new_binary * mask
        new_alpha_r = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)
        
        # 2-2. Fix mean, alpha row, and B, update alpha column
        alpha_r_B = new_alpha_r[:, None] * new_binary * mask
        new_alpha_c = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # 3. Fix mean and alpha, update B
        new_binary = torch.sign(W_tilde)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary + refine_mean_r[:, None] + refine_mean_c[None, :]) * mask

    # return sum_order
    return sum_order, sum_order_alternating


@torch.no_grad()
def high_order_residual_alternating_order1_x_rc_nomean(x, mask, order=2, S=None, iter=15, iter2=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    # Alternating update
    sum_order_alternating = sum_order.clone()
    new_alpha_r = scale_tensor_all_r.clone()
    new_alpha_c = scale_tensor_all_c.clone()
    for k in range(iter):        
        # 1-1. Fix mean, alpha column, and B, update alpha row
        alpha_c_B = new_alpha_c[None, :] * new_binary * mask
        new_alpha_r = torch.sum(alpha_c_B * new_matrix, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)
        
        # 1-2. Fix mean, alpha row, and B, update alpha column
        alpha_r_B = new_alpha_r[:, None] * new_binary * mask
        new_alpha_c = torch.sum(alpha_r_B * new_matrix, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary * mask

    # masked_B = new_alpha_c[None, :] * new_binary * mask
    # new_alpha_r = torch.sum(S * masked_B[:, :, None] * new_matrix[:, None, :], dim=(1,2)) / (torch.sum(S * masked_B[:, :, None] * masked_B[:, None, :], dim=(1,2)) + 1e-10)

    sum_order_alternating = torch.zeros_like(x) + new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary * mask

    return sum_order, sum_order_alternating


def high_order_residual_alternating_order1_x_rc2_split(x, mask, order=2, S=None, iter=15, iter2=15, split=4):
    x = x.reshape(split, 4096 // split, -1)
    mask = mask.reshape(split, 4096 // split, -1)
    # print(x.shape)   # [4, 1024, 128]
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # mean row
        mean_tensor_all_r = torch.nanmean(masked_x_tensor, dim=2)
        mean_tensor_all_r = torch.where(torch.isnan(mean_tensor_all_r), torch.zeros_like(mean_tensor_all_r), mean_tensor_all_r)
        # print(mean_tensor_all_r.shape)   # [4, 1024]
        masked_x_tensor -= mean_tensor_all_r[:, :, None]
        # mean column
        mean_tensor_all_c = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all_c = torch.where(torch.isnan(mean_tensor_all_c), torch.zeros_like(mean_tensor_all_c), mean_tensor_all_c)
        # print(mean_tensor_all_c.shape)   # [4, 128]
        masked_x_tensor -= mean_tensor_all_c[:, None, :]

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=2)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, :, None]), dim=1)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)
        # print(scale_tensor_all_r.shape)   # [4, 1024]
        # print(scale_tensor_all_c.shape)   # [4, 128]

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
        binary *= scale_tensor_all_r[:, :, None]
        binary *= scale_tensor_all_c[:, None, :]
        binary += mean_tensor_all_r[:, :, None] + mean_tensor_all_c[:, None, :]
        sum_order = sum_order + binary*mask

    # Alternating update
    refine_mean_r = mean_tensor_all_r.clone()
    refine_mean_c = mean_tensor_all_c.clone()
    sum_order_alternating = sum_order.clone()
    new_alpha_r = scale_tensor_all_r.clone()
    new_alpha_c = scale_tensor_all_c.clone()

    for k in range(iter):
        # 1-1. Fix alpha row, alpha column, and B, update mean row
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all_r = torch.nanmean(masked_x_tensor, dim=2)
        mean_tensor_all_r = torch.where(torch.isnan(mean_tensor_all_r), torch.zeros_like(mean_tensor_all_r), mean_tensor_all_r)
        refine_mean_r += mean_tensor_all_r.clone()
        # 1-2. Fix alpha row, alpha column, and B, update mean column
        mean_tensor_all_c = torch.nanmean(masked_x_tensor - mean_tensor_all_r[:, :, None], dim=1)
        mean_tensor_all_c = torch.where(torch.isnan(mean_tensor_all_c), torch.zeros_like(mean_tensor_all_c), mean_tensor_all_c)
        refine_mean_c += mean_tensor_all_c.clone()
        
        # 2-1. Fix mean, alpha column, and B, update alpha row
        W_tilde = new_matrix - (refine_mean_r[:, :, None] + refine_mean_c[:, None, :]) * mask
        alpha_c_B = new_alpha_c[:, None, :] * new_binary * mask
        new_alpha_r = torch.sum(alpha_c_B * W_tilde, dim=2) / (torch.sum(alpha_c_B * alpha_c_B, dim=2) + 1e-8)
        
        # 2-2. Fix mean, alpha row, and B, update alpha column
        alpha_r_B = new_alpha_r[:, :, None] * new_binary * mask
        new_alpha_c = torch.sum(alpha_r_B * W_tilde, dim=1) / (torch.sum(alpha_r_B * alpha_r_B, dim=1) + 1e-8)

        # 3. Fix mean and alpha, update B
        new_binary = torch.sign(W_tilde)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha_c[:, None, :] * new_alpha_r[:, :, None] * new_binary + refine_mean_r[:, :, None] + refine_mean_c[:, None, :]) * mask

    return sum_order.reshape(4096, -1), sum_order_alternating.reshape(4096, -1)


def high_order_residual_alternating_order2_rc(x, mask, order=2, S=None, iter=15, iter2=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list_r = []
    alpha_list_c = []
    refine_mean = torch.zeros(x.shape[0], device=x.device)
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        masked_x_tensor -= mean_tensor_all[:, None]

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        alpha_list_r.append(scale_tensor_all_r.clone())
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)
        alpha_list_c.append(scale_tensor_all_c.clone())

        binary= torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    # Alternating update
    sum_order_alternating = sum_order.clone()
    new_alpha_r = scale_tensor_all_r.clone()
    new_alpha_c = scale_tensor_all_c.clone()

    for k in range(iter):
        # 1. Fix alpha row, alpha column, and B, update mean
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        
        # 2-1. Fix mean, alpha column, and B, update alpha row 1
        W_tilde = new_matrix - (refine_mean[:, None] + alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1]) * mask
        alpha_c_B = alpha_list_c[0][None, :] * binary_list[0] * mask
        new_alpha_r0 = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)
        
        # 2-2. Fix mean, alpha row, and B, update alpha column 1
        alpha_r_B =  alpha_list_r[0][:, None] * binary_list[0] * mask
        new_alpha_c0 = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # 2-3. Fix mean, alpha column, and B, update alpha row 2
        W_tilde = new_matrix - (refine_mean[:, None] + new_alpha_c0[None, :] * new_alpha_r0[:, None] * binary_list[0]) * mask
        alpha_c_B = alpha_list_c[1][None, :] * binary_list[1] * mask
        new_alpha_r1 = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)
        
        # 2-4. Fix mean, alpha row, and B, update alpha column 2
        alpha_r_B =  alpha_list_r[1][:, None] * binary_list[1] * mask
        new_alpha_c1 = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # # 3. Fix mean and alpha, update B
        # new_matrix_expanded = (new_matrix - refine_mean[:, None] * mask).unsqueeze(-1)
        # comb0 = new_alpha_r0.reshape(-1, 1) @ new_alpha_c0.reshape(1, -1)
        # comb1 = new_alpha_r1.reshape(-1, 1) @ new_alpha_c1.reshape(1, -1)
        # v = torch.stack([-comb0 - comb1, -comb0 + comb1, 
        #             comb0 - comb1, comb0 + comb1], dim=2)

        # differences = torch.abs(new_matrix_expanded - v)
        # min_indices = torch.argmin(differences, dim=-1)

        # new_binary0 = torch.ones_like(min_indices)
        # new_binary0[(min_indices == 0) | (min_indices == 1)] = -1
        # new_binary1 = torch.ones_like(min_indices)
        # new_binary1[(min_indices == 0) | (min_indices == 2)] = -1 

        new_binary0 = binary_list[0].clone()
        new_binary1 = binary_list[1].clone()

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha_c0[None, :] * new_alpha_r0[:, None] * new_binary0 + new_alpha_c1[None, :] * new_alpha_r1[:, None] * new_binary1 + refine_mean[:, None]) * mask

        # update parameters
        alpha_list_r[0] = new_alpha_r0.clone()
        alpha_list_r[1] = new_alpha_r1.clone()
        alpha_list_c[0] = new_alpha_c0.clone()
        alpha_list_c[1] = new_alpha_c1.clone()
        binary_list[0] = new_binary0.clone()
        binary_list[1] = new_binary1.clone()

    return sum_order, sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_order2_rcmean(x, mask, order=2, num_iters=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list = []
    refine_mean_r = torch.zeros(x.shape[0], device=x.device)
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all_r = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all_r = torch.where(torch.isnan(mean_tensor_all_r), torch.zeros_like(mean_tensor_all_r), mean_tensor_all_r)
        refine_mean_r += mean_tensor_all_r.clone()
        masked_x_tensor -= mean_tensor_all_r[:, None]

        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)
        alpha_list.append(scale_tensor_all.clone())

        binary = torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all_r[:, None]
        sum_order = sum_order + binary*mask

    residual = new_matrix - sum_order
    masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
    # mean column
    mean_tensor_all_c = torch.nanmean(masked_x_tensor, dim=0)
    mean_tensor_all_c = torch.where(torch.isnan(mean_tensor_all_c), torch.zeros_like(mean_tensor_all_c), mean_tensor_all_c)
    # masked_x_tensor -= mean_tensor_all_c[None, :]
    refine_mean_c = mean_tensor_all_c.clone()
    sum_order = sum_order + refine_mean_c[None, :] * mask


    new_matrix = x.clone() * mask
    sum_order_alternating = sum_order.clone()
    
    for k in range(num_iters):
        # 1-1. Fix alpha1, alpha2, B1, and B2, update mean row
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all_r = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all_r = torch.where(torch.isnan(mean_tensor_all_r), torch.zeros_like(mean_tensor_all_r), mean_tensor_all_r)
        refine_mean_r += mean_tensor_all_r.clone()
        # # 1-2. Fix alpha1, alpha2, B1, and B2, update mean column
        # mean_tensor_all_c = torch.nanmean(masked_x_tensor - mean_tensor_all_r[:, None], dim=0)
        # mean_tensor_all_c = torch.where(torch.isnan(mean_tensor_all_c), torch.zeros_like(mean_tensor_all_c), mean_tensor_all_c)
        # refine_mean_c += mean_tensor_all_c.clone()

        # 2. Fix mean, B1, and B2, update alpha1 and alpha2
        new_alpha0 = 1. / (torch.sum(binary_list[0] * mask * binary_list[0] * mask, dim=1) + 1e-8) * torch.sum(binary_list[0] * mask * (new_matrix - (refine_mean_r[:, None] + refine_mean_c[None, :] + alpha_list[1][:, None] * binary_list[1]) * mask), dim=1)
        new_alpha1 = 1. / (torch.sum(binary_list[1] * mask * binary_list[1] * mask, dim=1) + 1e-8) * torch.sum(binary_list[1] * mask * (new_matrix - (refine_mean_r[:, None] + refine_mean_c[None, :] + new_alpha0[:, None] * binary_list[0]) * mask), dim=1)

        # 3. Fix mean, alpha1, and alpha2, update B1 and B2
        new_matrix_expanded = (new_matrix - (refine_mean_r[:, None] + refine_mean_c[None, :]) * mask).unsqueeze(-1)
        v = torch.stack([-new_alpha0 - new_alpha1, -new_alpha0 + new_alpha1, 
                    new_alpha0 - new_alpha1, new_alpha0 + new_alpha1], dim=1)
        v_expanded = v.unsqueeze(1)

        differences = torch.abs(new_matrix_expanded - v_expanded)
        min_indices = torch.argmin(differences, dim=-1)

        new_binary0 = torch.ones_like(min_indices)
        new_binary0[(min_indices == 0) | (min_indices == 1)] = -1
        new_binary1 = torch.ones_like(min_indices)
        new_binary1[(min_indices == 0) | (min_indices == 2)] = -1 

        # 4. Fix alpha1, alpha2, B1, and B2, update mean column
        residual = new_matrix - (new_alpha0[:, None] * new_binary0 + new_alpha1[:, None] * new_binary1 + refine_mean_r[:, None]) * mask
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        # mean column
        mean_tensor_all_c = torch.nanmean(masked_x_tensor, dim=0)
        mean_tensor_all_c = torch.where(torch.isnan(mean_tensor_all_c), torch.zeros_like(mean_tensor_all_c), mean_tensor_all_c)
        refine_mean_c = mean_tensor_all_c.clone()

        sum_order_alternating = torch.zeros_like(x) + (new_alpha0[:, None] * new_binary0 + new_alpha1[:, None] * new_binary1 + refine_mean_r[:, None] + refine_mean_c[None, :]) * mask
        
        # update parameters
        alpha_list[0] = new_alpha0.clone()
        alpha_list[1] = new_alpha1.clone()
        binary_list[0] = new_binary0.clone()
        binary_list[1] = new_binary1.clone()

    return sum_order, sum_order_alternating

def high_order_residual_alternating_order1_x_rc_nomean(x, mask, order=2, S=None, iter=15, iter2=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    # Alternating update
    sum_order_alternating = sum_order.clone()
    new_alpha_r = scale_tensor_all_r.clone()
    new_alpha_c = scale_tensor_all_c.clone()
    for k in range(iter):        
        # 1-1. Fix mean, alpha column, and B, update alpha row
        alpha_c_B = new_alpha_c[None, :] * new_binary * mask
        new_alpha_r = torch.sum(alpha_c_B * new_matrix, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)
        
        # 1-2. Fix mean, alpha row, and B, update alpha column
        alpha_r_B = new_alpha_r[:, None] * new_binary * mask
        new_alpha_c = torch.sum(alpha_r_B * new_matrix, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary * mask

    # masked_B = new_alpha_c[None, :] * new_binary * mask
    # new_alpha_r = torch.sum(S * masked_B[:, :, None] * new_matrix[:, None, :], dim=(1,2)) / (torch.sum(S * masked_B[:, :, None] * masked_B[:, None, :], dim=(1,2)) + 1e-10)

    # sum_order_alternating = torch.zeros_like(x) + new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary * mask

    return sum_order, sum_order_alternating

def high_order_residual_alternating_mean(x, mask, order=2, num_iters=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list = []
    refine_mean = torch.zeros(x.shape[0], device=x.device)
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)
        alpha_list.append(scale_tensor_all.clone())

        binary = torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    new_matrix = x.clone() * mask
    sum_order_alternating = sum_order.clone()
    
    for k in range(num_iters):
        # 1. Fix alpha1, alpha2, B1, and B2, update mean
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()

        # 2. Fix mean, B1, and B2, update alpha1 and alpha2
        new_alpha0 = 1. / (torch.sum(binary_list[0] * mask * binary_list[0] * mask, dim=1) + 1e-8) * torch.sum(binary_list[0] * mask * (new_matrix - refine_mean[:, None] * mask - alpha_list[1][:, None] * binary_list[1] * mask), dim=1)
        new_alpha1 = 1. / (torch.sum(binary_list[1] * mask * binary_list[1] * mask, dim=1) + 1e-8) * torch.sum(binary_list[1] * mask * (new_matrix - refine_mean[:, None] * mask - new_alpha0[:, None] * binary_list[0] * mask), dim=1)

        # 3. Fix mean, alpha1, and alpha2, update B1 and B2
        new_matrix_expanded = (new_matrix - refine_mean[:, None] * mask).unsqueeze(-1)
        v = torch.stack([-new_alpha0 - new_alpha1, -new_alpha0 + new_alpha1, 
                    new_alpha0 - new_alpha1, new_alpha0 + new_alpha1], dim=1)
        v_expanded = v.unsqueeze(1)

        differences = torch.abs(new_matrix_expanded - v_expanded)
        min_indices = torch.argmin(differences, dim=-1)

        new_binary0 = torch.ones_like(min_indices)
        new_binary0[(min_indices == 0) | (min_indices == 1)] = -1
        new_binary1 = torch.ones_like(min_indices)
        new_binary1[(min_indices == 0) | (min_indices == 2)] = -1 

        sum_order_alternating = torch.zeros_like(x) + (new_alpha0[:, None] * new_binary0 + new_alpha1[:, None] * new_binary1 + refine_mean[:, None]) * mask
        
        # update parameters
        alpha_list[0] = new_alpha0.clone()
        alpha_list[1] = new_alpha1.clone()
        binary_list[0] = new_binary0.clone()
        binary_list[1] = new_binary1.clone()

    return sum_order, sum_order_alternating


loss0 = 0
loss = 0
loss1 = 0
loss2 = 0
loss3 = 0
loss4 = 0
loss5 = 0
for seed in range(400):
    torch.manual_seed(seed)

    x = torch.randn(8,9) + 4.567


    # mask = torch.tensor([
    #     [1,1,1],
    #     [0,0,0]
    # ],dtype=torch.bool)

    mask = torch.randint(1, 2, (8,9)).bool()
    mask[:, [1,2,5,6,7]] = 0
    # print(mask)

    # sum_order1, sum_order, sum_order_refine, sum_order_refine2, sum_order_alternating = high_order_residual_refine(x, mask, order=2)
    # err1 = error_computing(x, sum_order1)
    # err = error_computing(x, sum_order)
    # refine_err = error_computing(x, sum_order_refine)
    # refine_err2 = error_computing(x, sum_order_refine2)
    # alternating_err = error_computing(x, sum_order_alternating)



    sum_order1, sum_order_alternating2 = high_order_residual_alternating_mean(x, mask, order=2)
    sum_order2, sum_order_alternating3 = high_order_residual_alternating_order2_rc(x, mask, order=2)
    sum_order3, sum_order_alternating4 = high_order_residual_alternating_order2_rcmean(x, mask, order=2)

    sum_order4, sum_order_alternating5 = high_order_residual_alternating_order1_x_rc_nomean(x, mask, order=1)
    # sum_order2, sum_order_alternating3 = high_order_residual_alternating_order1_x_rc2(x, mask, order=1, iter2=0)
    # sum_order2, sum_order_alternating3 = high_order_residual_alternating_order1_x_rc2_split(x, mask, order=1)
    # sum_order3, sum_order_alternating4 = high_order_residual_alternating_order1_x_rc2_split(x, mask, order=1)
    err = error_computing(x, sum_order1)
    err2 = error_computing(x, sum_order2)
    err3 = error_computing(x, sum_order3)
    err4 = error_computing(x, sum_order4)
    # alternating_err = error_computing(x, sum_order_alternating)
    alternating_err2 = error_computing(x, sum_order_alternating2)
    alternating_err3 = error_computing(x, sum_order_alternating3)
    alternating_err4 = error_computing(x, sum_order_alternating4)
    alternating_err5 = error_computing(x, sum_order_alternating5)

    # print(err1)
    # print(err)
    # print(refine_err)
    # print(refine_err2)
    # print(alternating_err)
    # print(alternating_err2)
    # # accelerate.append(alternating_err2)
    
    # print('========')
    # loss0 += err - alternating_err
    loss0 += err
    loss1 += err2
    loss4 += err4
    loss2 += alternating_err2
    loss3 += alternating_err3
    loss5 += alternating_err5
    if err4 < alternating_err5:
        print('wrong')
    # if alternating_err2 < alternating_err4:
    #     print('wrong')
    else:
        # loss += alternating_err - alternating_err2
        loss += alternating_err2

print(loss0)
# print(loss)
print(loss1)
print(loss2)
print(loss3)
print(loss4)
print(loss5)

