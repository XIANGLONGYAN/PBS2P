import torch

error_N = 2048*4096*128

def error_computing_x_all_accelerate(origin_matrix, quantized_matrix, mask, S):
    # inps shape [128, 2048, 128]
    R = ((origin_matrix - quantized_matrix) * mask).T
    P = torch.einsum('ik,jk->ij', R, R)
    return torch.sum(P * S) / error_N

index = 0

def high_order_residual_alternating_mean_x(x, mask, order=2, S=None, num_iters=15, iter2=15):
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

    MM = mask[:, :, None] * mask[:, None, :]
    refine_mean_den = torch.sum(S * MM, dim=(1,2)) + 1e-10
    masked_B0 = binary_list[0] * mask
    BB0 = masked_B0[:, :, None] * masked_B0[:, None, :]
    new_alpha0_den = torch.sum(S * BB0, dim=(1,2)) + 1e-10
    masked_B1 = binary_list[1] * mask
    BB1 = masked_B1[:, :, None] * masked_B1[:, None, :]
    new_alpha1_den = torch.sum(S * BB1, dim=(1,2)) + 1e-10
    for kk in range(iter2):
        # X error update mean
        R = new_matrix - new_alpha0[:, None] * new_binary0 * mask - new_alpha1[:, None] * new_binary1 * mask
        refine_mean = torch.sum(S * R[:, :, None] * MM, dim=(1,2)) / refine_mean_den

        # X error update alpha
        masked_W_mu = new_matrix - refine_mean[:, None] * mask
        BW0 = masked_B0[:, :, None] * masked_W_mu[:, None, :]
        BB01 = masked_B0[:, :, None] * (new_alpha1[:, None] * masked_B1)[:, None, :]
        new_alpha0 = torch.sum(S * (BW0-BB01), dim=(1,2)) / new_alpha0_den

        BW1 = masked_B1[:, :, None] * masked_W_mu[:, None, :]
        BB10 = masked_B1[:, :, None] * (new_alpha0[:, None] * masked_B0)[:, None, :]
        new_alpha1 = torch.sum(S * (BW1-BB10), dim=(1,2)) / new_alpha1_den


    sum_order_alternating2 = torch.zeros_like(x) + (new_alpha0[:, None] * binary_list[0] + new_alpha1[:, None] * binary_list[1] + refine_mean[:, None]) * mask

    return sum_order_alternating, sum_order_alternating2

for seed in range(200):
    torch.manual_seed(seed)

    x = torch.randn(8,9) + 4.567


    # mask = torch.tensor([
    #     [1,1,1],
    #     [0,0,0]
    # ],dtype=torch.bool)

    mask = torch.randint(0, 2, (8, 9)).bool()
    # mask[:, [2,4,5]] = False
    # print(mask)

    # sum_order1, sum_order, sum_order_refine, sum_order_refine2, sum_order_alternating = high_order_residual_refine(x, mask, order=2)
    # err1 = error_computing(x, sum_order1)
    # err = error_computing(x, sum_order)
    # refine_err = error_computing(x, sum_order_refine)
    # refine_err2 = error_computing(x, sum_order_refine2)
    # alternating_err = error_computing(x, sum_order_alternating)

    inp = torch.randn(3,10,9)
    S = torch.einsum('bki,bkj->ij', inp, inp)

    sum_order_alternating, sum_order_alternating2 = high_order_residual_alternating_mean_x(x, mask, order=2, S=S)
    # sum_order_alternating3, sum_order_alternating4 = high_order_residual_alternating_order1_x(x, ~mask, order=1, S=S)
    # err = error_computing(x, sum_order)
    alternating_err = error_computing_x_all_accelerate(x, sum_order_alternating, mask, S)
    alternating_err2 = error_computing_x_all_accelerate(x, sum_order_alternating2, mask, S)
    # alternating_err3 = error_computing_x_all_accelerate(x, sum_order_alternating3, mask, S)
    # alternating_err4 = error_computing_x_all_accelerate(x, sum_order_alternating4, mask, S)
    # alternating_err2 = error_computing_x_all_accelerate(x, sum_order_alternating2+sum_order_alternating4, S)

    # print(err1)
    # print(err)
    # print(refine_err)
    # print(refine_err2)
    print(alternating_err)
    print(alternating_err2)
    # print(alternating_err3)
    # print(alternating_err4)
    # # # accelerate.append(alternating_err2)
    
    print('========')
    # if alternating_err < alternating_err2:
    #     print('wrong')
    # if alternating_err < alternating_err4:
    #     # if dLoss >= 0:
    #     #     print('wrong')
    #     print(alternating_err)
    #     # print(alternating_err2)
    #     print(alternating_err4)
    #     # print(dLoss)
    # elif alternating_err > alternating_err4:
    #     if dLoss <= 0:
    #         print('wrong')
