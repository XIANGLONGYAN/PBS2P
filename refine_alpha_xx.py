import torch
import time

# error_N = 2048*4096*128
error_N = 3*10*9

def error_computing_x_all_accelerate(origin_matrix, quantized_matrix, mask, S):
    # inps shape [128, 2048, 128]
    R = ((origin_matrix - quantized_matrix) * mask).T
    P = torch.einsum('ik,jk->ij', R, R)
    return torch.sum(P * S) / error_N

def error_computing_xx_all(origin_matrix, quantized_matrix, mask, inp, binp):
    # inps shape [128, 2048, 128]
    origin_matrix *= mask
    quantized_matrix *= mask
    mse = torch.mean((inp.to(torch.float) @ origin_matrix.T - binp.to(torch.float) @ quantized_matrix.T) ** 2)
    return mse

def error_computing_xx_all_accelerate(origin_matrix, quantized_matrix, mask, S_hat, S_hathat):
    # inps shape [128, 2048, 128]
    origin_matrix *= mask
    quantized_matrix *= mask
    WW_hat = torch.einsum('jk,jl->kl', quantized_matrix, origin_matrix)
    WW_hathat = torch.einsum('jk,jl->kl', quantized_matrix, quantized_matrix)
    return torch.sum(S_hathat * WW_hathat - 2 * S_hat * WW_hat) / error_N

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

    sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating

def high_order_residual_alternating_order1_xx(x, mask, order=2, S_hat=None, S_hathat=None, iter=15, iter2=15):
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
    refine_mean_den = 2*torch.sum(S_hathat * MM, dim=(1,2)) + 1e-10
    S_hat2 = 2*torch.sum(S_hat * MM, dim=1)
    masked_B = new_binary * mask
    new_alpha_den = 2*torch.sum(S_hathat * (masked_B[:, :, None] * masked_B[:, None, :]), dim=(1,2)) + 1e-10
    S_hathat = S_hathat * MM * (masked_B[:, :, None] + masked_B[:, None, :])
    S_hat = S_hat * (2 * masked_B[:, :, None] * new_matrix[:, None, :])
    for kk in range(iter2):
        # X error update mean
        refine_mean = (torch.sum(S_hat2 * new_matrix, dim=1) - torch.sum(S_hathat * new_alpha[:, None, None], dim=(1,2))) / refine_mean_den

        # X error update alpha
        new_alpha = torch.sum(S_hat - S_hathat * refine_mean[:, None, None], dim=(1,2)) / new_alpha_den

    sum_order_alternating2 = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating, sum_order_alternating2

total_loss = 0
num_layer = 2
for seed in range(200):
    torch.manual_seed(seed)

    inp = torch.randn(3,10,9)
    # S = torch.einsum('bki,bkj->ij', inp, inp)
    binp = inp

    for kkk in range(num_layer):
        x = torch.randn(9,9) + 4.567


        # mask = torch.tensor([
        #     [1,1,1],
        #     [0,0,0]
        # ],dtype=torch.bool)

        mask = torch.randint(0, 2, (9, 9)).bool()
        # print(mask)

        # sum_order1, sum_order, sum_order_refine, sum_order_refine2, sum_order_alternating = high_order_residual_refine(x, mask, order=2)
        # err1 = error_computing(x, sum_order1)
        # err = error_computing(x, sum_order)
        # refine_err = error_computing(x, sum_order_refine)
        # refine_err2 = error_computing(x, sum_order_refine2)
        # alternating_err = error_computing(x, sum_order_alternating)

        # inp = torch.randn(3,10,9)
        # # S = torch.einsum('bki,bkj->ij', inp, inp)
        # binp = inp

        S_old = torch.einsum('bki,bkj->ij', binp, binp)
        S_hat = torch.einsum('bki,bkj->ij', binp, inp)
        S_hathat = torch.einsum('bki,bkj->ij', binp, binp)


        sum_order_alternating = high_order_residual_alternating_order1_x(x, mask, order=1, S=S_old)
        sum_order_alternating2, sum_order_alternating3 = high_order_residual_alternating_order1_xx(x, mask, order=1, S_hat=S_hat, S_hathat=S_hathat)
        # sum_order_alternating3, sum_order_alternating4 = high_order_residual_alternating_order1_x(x, ~mask, order=1, S=S)
        # err = error_computing(x, sum_order)
        # alternating_err = error_computing_xx_all(x, sum_order_alternating, mask, inp, binp)
        # alternating_err2 = error_computing_xx_all(x, sum_order_alternating2, mask, inp, binp)
        # alternating_err3 = error_computing_xx_all(x, sum_order_alternating3, mask, inp, binp)
        alternating_err = error_computing_xx_all_accelerate(x, sum_order_alternating, mask, S_hat, S_hathat)
        alternating_err2 = error_computing_xx_all_accelerate(x, sum_order_alternating2, mask, S_hat, S_hathat)
        alternating_err3 = error_computing_xx_all_accelerate(x, sum_order_alternating3, mask, S_hat, S_hathat)
        # alternating_err4 = error_computing_x_all_accelerate(x, sum_order_alternating4, mask, S)
        # alternating_err2 = error_computing_x_all_accelerate(x, sum_order_alternating2+sum_order_alternating4, S)

        # print(err1)
        # print(err)
        # print(refine_err)
        # print(refine_err2)
        # print(alternating_err)
        # print(alternating_err2)
        # print(alternating_err3)
        # # total_loss += alternating_err3
        # # # print(alternating_err4)
        # # # # # # accelerate.append(alternating_err2)
        
        # print('========')
        # if alternating_err2 < alternating_err:
        #     print('true')
        #     print(alternating_err)
        #     print(alternating_err2)
        if kkk==1:
            if alternating_err < alternating_err3:
                torch.set_printoptions(precision=10)
                print('wrong')
            #     # if dLoss >= 0:
            #     #     print('wrong')
                print(alternating_err)
                # print(alternating_err2)
                print(alternating_err3)
                # total_loss += alternating_err3 - alternating_err
            else:
                # pass
                # print(alternating_err)
                # # print(alternating_err2)
                # print(alternating_err3)
                total_loss += (alternating_err - alternating_err3)
        #     # print(dLoss)
        # # elif alternating_err > alternating_err4:
        # #     if dLoss <= 0:
        # #         print('wrong')

        inp = inp @ (x*mask).T
        binp = binp @ (sum_order_alternating*mask).T

print(f'total loss: {total_loss}')
