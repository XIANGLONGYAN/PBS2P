import torch
import time

error_N = 2048*4096*128

def error_computing_x_all_accelerate(origin_matrix, quantized_matrix, mask, S):
    # inps shape [128, 2048, 128]
    R = ((origin_matrix - quantized_matrix) * mask).T
    P = torch.einsum('ik,jk->ij', R, R)
    return torch.sum(P * S) / error_N

index = 0

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


    MM = mask[:, :, None] * mask[:, None, :]
    refine_mean_den = 2*torch.sum(S * MM, dim=(1,2)) + 1e-10
    masked_B = new_binary * mask
    BB = masked_B[:, :, None] * masked_B[:, None, :]
    new_alpha_den = 2*torch.sum(S * BB, dim=(1,2)) + 1e-10
    diag_S = torch.diag(S)
    for kk in range(iter2):
        # X error update mean
        R = new_matrix - new_alpha[:, None] * new_binary * mask
        WB = R[:, :, None] + R[:, None, :]
        refine_mean = torch.sum(S * WB * MM, dim=(1,2)) / refine_mean_den

        # X error update alpha
        masked_W_mu = new_matrix - refine_mean[:, None] * mask
        BW = masked_B[:, :, None] * masked_W_mu[:, None, :]
        BW = BW + BW.transpose(1,2)
        new_alpha = torch.sum(S * BW, dim=(1,2)) / new_alpha_den


        # # prev loss
        # alpha_B = (new_alpha[:, None] * new_binary) * mask
        # alpha_BW = torch.einsum('ij,ik->jk', alpha_B, masked_W_mu)
        # alpha_BW = alpha_BW + alpha_BW.T
        # double_alpha_B = torch.einsum('ij,ik->jk', alpha_B, alpha_B)
        # prevLoss = torch.sum((double_alpha_B-alpha_BW) * S)

        # # X error update B (approx.)
        # new_binary1 = torch.sign(
        #     (torch.einsum('ij,ki->kj', S.to(torch.float), masked_W_mu) / new_alpha[:,None] \
        #     - torch.einsum('ij,ki->kj', S.to(torch.float), new_binary) \
        #     + diag_S * new_binary) / diag_S
        # )

        # # new loss
        # alpha_B = (new_alpha[:, None] * new_binary1) * mask
        # alpha_BW = torch.einsum('ij,ik->jk', alpha_B, masked_W_mu)
        # alpha_BW = alpha_BW + alpha_BW.T
        # double_alpha_B = torch.einsum('ij,ik->jk', alpha_B, alpha_B)
        # newLoss = torch.sum((double_alpha_B-alpha_BW) * S)
        # if newLoss < prevLoss:
        #     new_binary = new_binary1
        #     masked_B = new_binary * mask
        #     BB = masked_B[:, :, None] * masked_B[:, None, :]
        #     new_alpha_den = 2*torch.sum(S * BB, dim=(1,2)) + 1e-10

    sum_order_alternating2 = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating, sum_order_alternating2

def high_order_residual_alternating_order1_x_rc(x, mask, order=2, S=None, iter=15, iter2=15):
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
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
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

    MM = mask[:, :, None] * mask[:, None, :]
    refine_mean_den = torch.sum(S * MM, dim=(1,2)) + 1e-10
    masked_B = new_alpha_c[None, :] * new_binary * mask
    BB = masked_B[:, :, None] * masked_B[:, None, :]
    new_alpha_den = torch.sum(S * BB, dim=(1,2)) + 1e-10
    diag_S = torch.diag(S)
    gamma = 1
    for kk in range(iter2):
        # X error update mean
        R = new_matrix - new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary * mask
        refine_mean = torch.sum(S * R[:, :, None] * MM, dim=(1,2)) / refine_mean_den

        # X error update alpha row
        masked_W_mu = new_matrix - refine_mean[:, None] * mask
        BW = masked_B[:, :, None] * masked_W_mu[:, None, :]
        new_alpha_r = torch.sum(S * BW, dim=(1,2)) / new_alpha_den

    # # X error update alpha column (newton)
    # alpha_r_B = new_alpha_r[:, None] * new_binary * mask
    # new_alpha_c -= gamma * torch.sum(S * torch.sum(alpha_r_B[:, :, None] * (new_alpha_c[None, :] * alpha_r_B - masked_W_mu)[:, None, :], dim=0), dim=1) / (diag_S * torch.sum(alpha_r_B**2, dim=0) + 1e-10)
    # gamma *= 0.9

    sum_order_alternating2 = torch.zeros_like(x) + (new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating, sum_order_alternating2

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

    MM = mask[:, :, None] * mask[:, None, :]
    refine_mean_den = torch.sum(S * MM, dim=(1,2)) + 1e-10
    masked_B = new_alpha_c[None, :] * new_binary * mask
    BB = masked_B[:, :, None] * masked_B[:, None, :]
    new_alpha_den = torch.sum(S * BB, dim=(1,2)) + 1e-10
    # diag_S = torch.diag(S)
    # gamma = 1
    for kk in range(iter2):
        # X error update mean row
        R = new_matrix - refine_mean_c[None, :] - new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary * mask
        refine_mean_r = torch.sum(S * R[:, :, None] * MM, dim=(1,2)) / refine_mean_den

        # X error update alpha row
        masked_W_mu = new_matrix - (refine_mean_r[:, None] + refine_mean_c[None,:]) * mask
        BW = masked_B[:, :, None] * masked_W_mu[:, None, :]
        new_alpha_r = torch.sum(S * BW, dim=(1,2)) / new_alpha_den

    # # X error update alpha column (newton)
    # alpha_r_B = new_alpha_r[:, None] * new_binary * mask
    # new_alpha_c -= gamma * torch.sum(S * torch.sum(alpha_r_B[:, :, None] * (new_alpha_c[None, :] * alpha_r_B - masked_W_mu)[:, None, :], dim=0), dim=1) / (diag_S * torch.sum(alpha_r_B**2, dim=0) + 1e-10)
    # gamma *= 0.9

    sum_order_alternating2 = torch.zeros_like(x) + (new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary + refine_mean_r[:, None] + refine_mean_c[None, :]) * mask

    return sum_order_alternating, sum_order_alternating2

total_loss = 0
loss1 = 0
loss2 = 0
loss3 = 0
loss4 = 0
loss5 = 0
loss0 = 0
for seed in range(200):
    torch.manual_seed(seed)

    x = torch.randn(8,9) + 4.567


    # mask = torch.tensor([
    #     [1,1,1],
    #     [0,0,0]
    # ],dtype=torch.bool)

    mask = torch.randint(0, 2, (8, 9)).bool()
    # print(mask)

    # sum_order1, sum_order, sum_order_refine, sum_order_refine2, sum_order_alternating = high_order_residual_refine(x, mask, order=2)
    # err1 = error_computing(x, sum_order1)
    # err = error_computing(x, sum_order)
    # refine_err = error_computing(x, sum_order_refine)
    # refine_err2 = error_computing(x, sum_order_refine2)
    # alternating_err = error_computing(x, sum_order_alternating)

    inp = torch.randn(3,10,9)
    S = torch.einsum('bki,bkj->ij', inp, inp)

    sum_order_alternating, sum_order_alternating1 = high_order_residual_alternating_order1_x(x, mask, order=1, S=S)
    sum_order_alternating2, sum_order_alternating3 = high_order_residual_alternating_order1_x_rc(x, mask, order=1, S=S)
    sum_order_alternating4, sum_order_alternating5 = high_order_residual_alternating_order1_x_rc2(x, mask, order=1, S=S)
    # sum_order_alternating3, sum_order_alternating4 = high_order_residual_alternating_order1_x(x, ~mask, order=1, S=S)
    # err = error_computing(x, sum_order)
    alternating_err = error_computing_x_all_accelerate(x, sum_order_alternating, mask, S)
    alternating_err1 = error_computing_x_all_accelerate(x, sum_order_alternating1, mask, S)
    alternating_err2 = error_computing_x_all_accelerate(x, sum_order_alternating2, mask, S)
    alternating_err3 = error_computing_x_all_accelerate(x, sum_order_alternating3, mask, S)
    alternating_err4 = error_computing_x_all_accelerate(x, sum_order_alternating4, mask, S)
    alternating_err5 = error_computing_x_all_accelerate(x, sum_order_alternating5, mask, S)
    # alternating_err2 = error_computing_x_all_accelerate(x, sum_order_alternating2+sum_order_alternating4, S)

    # print(err1)
    # print(err)
    # print(refine_err)
    # print(refine_err2)
    # print(alternating_err)
    # print(alternating_err2)
    # print(alternating_err3)
    # total_loss += alternating_err3
    # # # print(alternating_err4)
    # # # # # # accelerate.append(alternating_err2)
    
    # print('========')
    # if alternating_err2 < alternating_err:
    #     print('true')
    #     print(alternating_err)
    #     print(alternating_err2)
    loss0 += alternating_err
    loss1 += alternating_err1
    loss2 += alternating_err2
    loss3 += alternating_err3
    loss4 += alternating_err4
    loss5 += alternating_err5
    if alternating_err1 < alternating_err5:
        torch.set_printoptions(precision=10)
        print('wrong')
    #     # if dLoss >= 0:
    #     #     print('wrong')
    #     print(alternating_err)
    #     # print(alternating_err2)
    #     print(alternating_err2)
    #     # print(dLoss)
    # # elif alternating_err > alternating_err4:
    # #     if dLoss <= 0:
    # #         print('wrong')

# print(f'total loss: {total_loss}')
print(loss0)
print(loss1)
print(loss2)
print(loss3)
print(loss4)
print(loss5)