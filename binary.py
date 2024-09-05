from numpy import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
index = 0
@torch.no_grad()
def part_mean(tensor, op='-'):
    non_zero = tensor*(tensor!=0)

    mean_val = non_zero.mean(-1).view(-1, 1)

    return mean_val

@torch.no_grad()
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

@torch.no_grad()
def high_order_residual_refine_mean(x, mask, order=2):
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

    # refine mean
    residual = new_matrix - sum_order
    masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
    mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
    mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
    sum_order += mean_tensor_all[:, None] * mask
    
    return sum_order

@torch.no_grad()
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
        new_binary = torch.sign(new_matrix - refine_mean[:, None] * mask)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask


    return sum_order_alternating

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

    # prev loss
    alpha_B = (new_alpha[:, None] * new_binary) * mask
    alpha_BW = torch.einsum('ij,ik->jk', alpha_B, masked_W_mu)
    alpha_BW = alpha_BW + alpha_BW.T
    double_alpha_B = torch.einsum('ij,ik->jk', alpha_B, alpha_B)
    prevLoss = torch.sum((double_alpha_B-alpha_BW) * S)

    # X error update B (approx.)
    new_binary1 = torch.sign(
        (torch.einsum('ij,ki->kj', S.to(torch.float), masked_W_mu) / new_alpha[:,None] \
        - torch.einsum('ij,ki->kj', S.to(torch.float), new_binary) \
        + diag_S * new_binary) / diag_S
    )

    # new loss
    alpha_B = (new_alpha[:, None] * new_binary1) * mask
    alpha_BW = torch.einsum('ij,ik->jk', alpha_B, masked_W_mu)
    alpha_BW = alpha_BW + alpha_BW.T
    double_alpha_B = torch.einsum('ij,ik->jk', alpha_B, alpha_B)
    newLoss = torch.sum((double_alpha_B-alpha_BW) * S)
    if newLoss < prevLoss:
        new_binary = new_binary1

    # X error update mean
    R = new_matrix - new_alpha[:, None] * new_binary * mask
    WB = R[:, :, None] + R[:, None, :]
    refine_mean = torch.sum(S * WB * MM, dim=(1,2)) / refine_mean_den

    # X error update alpha
    masked_W_mu = new_matrix - refine_mean[:, None] * mask
    BW = masked_B[:, :, None] * masked_W_mu[:, None, :]
    BW = BW + BW.transpose(1,2)
    new_alpha = torch.sum(S * BW, dim=(1,2)) / new_alpha_den

    sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_order1_xB(x, mask, order=2, S=None, iter=15, iter2=15):
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

    # print(new_binary.shape[1])   # [blocksize]
    SA = diag_S * new_alpha[:, None]
    iter3 = 1
    for _ in range(iter3):
        for t in range(new_binary.shape[1]):
            r = masked_W_mu - new_alpha[:, None] * new_binary * mask
            new_binary[:, t] = torch.sign(new_alpha * (r @ S[t].to(torch.float) + SA[:,t] * new_binary[:,t]))

    # X error update mean
    R = new_matrix - new_alpha[:, None] * new_binary * mask
    WB = R[:, :, None] + R[:, None, :]
    refine_mean = torch.sum(S * WB * MM, dim=(1,2)) / refine_mean_den

    sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_order1_onlyx(x, mask, order=2, S=None, iter=0, iter2=15):
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
    SM = S * MM
    for kk in range(iter2):
        # X error update mean
        R = new_matrix - new_alpha[:, None] * new_binary * mask
        WB = R[:, :, None] + R[:, None, :]
        refine_mean = torch.sum(SM * WB, dim=(1,2)) / refine_mean_den

        # X error update alpha
        masked_W_mu = new_matrix - refine_mean[:, None] * mask
        BW = masked_B[:, :, None] * masked_W_mu[:, None, :]
        BW = BW + BW.transpose(1,2)
        new_alpha = torch.sum(S * BW, dim=(1,2)) / new_alpha_den


        # prev loss
        alpha_B = (new_alpha[:, None] * new_binary) * mask
        alpha_BW = torch.einsum('ij,ik->jk', alpha_B, masked_W_mu)
        alpha_BW = alpha_BW + alpha_BW.T
        double_alpha_B = torch.einsum('ij,ik->jk', alpha_B, alpha_B)
        prevLoss = torch.sum((double_alpha_B-alpha_BW) * S)

        # X error update B (approx.)
        new_binary1 = torch.sign(
            (torch.einsum('ij,ki->kj', S.to(torch.float), masked_W_mu) / new_alpha[:,None] \
            - torch.einsum('ij,ki->kj', S.to(torch.float), new_binary) \
            + diag_S * new_binary) / diag_S
        )

        # new loss
        alpha_B = (new_alpha[:, None] * new_binary1) * mask
        alpha_BW = torch.einsum('ij,ik->jk', alpha_B, masked_W_mu)
        alpha_BW = alpha_BW + alpha_BW.T
        double_alpha_B = torch.einsum('ij,ik->jk', alpha_B, alpha_B)
        newLoss = torch.sum((double_alpha_B-alpha_BW) * S)
        if newLoss < prevLoss:
            new_binary = new_binary1
            masked_B = new_binary * mask
            BB = masked_B[:, :, None] * masked_B[:, None, :]
            new_alpha_den = 2*torch.sum(S * BB, dim=(1,2)) + 1e-10


    sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating



@torch.no_grad()
def high_order_residual_alternating_order1_x_prev(x, mask, order=2, S=None, prev_W=None, iter=15):
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
        new_binary = torch.sign(new_matrix - refine_mean[:, None] * mask)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    # X error update
    masked_B = new_binary * mask
    BB = masked_B[:, :, None] * masked_B[:, None, :]

    masked_W_mu = new_matrix - refine_mean[:, None] * mask
    BW = masked_B[:, :, None] * masked_W_mu[:, None, :]
    BW = BW + BW.transpose(1,2)

    if prev_W != None:
        pBW = masked_B[:, :, None] * prev_W[:, None, :]
        BW = BW + pBW + pBW.transpose(1,2)

    new_alpha = torch.sum(S * BW, dim=(1,2)) / (2*torch.sum(S * BB, dim=(1,2)) + 1e-10)
    sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating

@torch.no_grad()
def db_llm(x, mask):
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    masked_x_tensor = torch.where(mask, new_matrix, torch.tensor(float('nan')))
    mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
    mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
    masked_x_tensor -= mean_tensor_all[:, None]

    new_masked_x_tensor = torch.where(torch.isnan(masked_x_tensor), torch.zeros_like(masked_x_tensor), masked_x_tensor)
    scale = torch.max(torch.abs(new_masked_x_tensor), dim=1, keepdim=True).values
    scale /= (2**2-1)
    alpha1, alpha2 = 2*scale, -scale

    x_1 = torch.sign(masked_x_tensor - 0.5*(alpha1+alpha2)).clamp(min=0)
    x_2 = torch.sign(-(masked_x_tensor - alpha1*x_1 - 0.5*alpha2)).clamp(min=0)

    binary = alpha1*x_1 + alpha2*x_2
    binary += mean_tensor_all[:, None]
    binary *= mask

    return binary

@torch.no_grad()
def high_order_residual_search_clip(x, mask, order=2, k=0):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    K = 100
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        # adjust clip
        new_masked_x_tensor = torch.where(torch.isnan(masked_x_tensor), torch.zeros_like(masked_x_tensor), masked_x_tensor)
        scale = torch.max(torch.abs(new_masked_x_tensor), dim=1).values
        step = scale / K
        print(step / scale_tensor_all)
        scale_tensor_all += k*step
        # print(f'scale shape: {scale_tensor_all.shape}')   # [4096]
        # print(f'step shape: {step.shape}')   # [4096]

        binary= torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask
    
    return sum_order

@torch.no_grad()
def high_order_residual_refine(x, mask, order=2):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list = []
    mean_list = []
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        mean_list.append(mean_tensor_all.clone())
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)
        alpha_list.append(scale_tensor_all.clone())

        binary = torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    # refine one step
    sum_order_refine = torch.zeros_like(x)
    new_matrix = x.clone() * mask
    new_matrix -= (mean_list[0][:, None] + mean_list[1][:, None])
    new_alpha0 = 1. / (torch.sum(binary_list[0] * mask * binary_list[0] * mask, dim=1) + 1e-8) * torch.sum(binary_list[0] * mask * (new_matrix - alpha_list[1][:, None] * binary_list[1] * mask), dim=1)
    sum_order_refine = sum_order_refine + (new_alpha0[:, None] * binary_list[0] + mean_list[0][:, None]) * mask

    new_alpha1 = 1. / (torch.sum(binary_list[1] * mask * binary_list[1] * mask, dim=1) + 1e-8) * torch.sum(binary_list[1] * mask * (new_matrix - alpha_list[0][:, None] * binary_list[0] * mask), dim=1)
    sum_order_refine = sum_order_refine + (new_alpha1[:, None] * binary_list[1] + mean_list[1][:, None]) * mask

    # print(f'debug 000000')
    return sum_order_refine


@torch.no_grad()
def high_order_residual_refine_accelerate(x, mask, order=2):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list = []
    mean_list = []
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        mean_list.append(mean_tensor_all.clone())
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)
        alpha_list.append(scale_tensor_all.clone())

        binary = torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    # refine one step (accelerate way)
    sum_order_refine_accelerate = torch.zeros_like(x)
    new_matrix2 = x.clone() * mask
    new_matrix2 -= (mean_list[0][:, None] + mean_list[1][:, None])
    new_alpha0 = 1. / (torch.sum(binary_list[0] * mask * binary_list[0] * mask, dim=1) + 1e-8) * torch.sum(binary_list[0] * mask * (new_matrix2 - alpha_list[1][:, None] * binary_list[1] * mask), dim=1)
    sum_order_refine_accelerate = sum_order_refine_accelerate + (new_alpha0[:, None] * binary_list[0] + mean_list[0][:, None]) * mask

    new_alpha1 = 1. / (torch.sum(binary_list[1] * mask * binary_list[1] * mask, dim=1) + 1e-8) * torch.sum(binary_list[1] * mask * (new_matrix2 - new_alpha0[:, None] * binary_list[0] * mask), dim=1)
    sum_order_refine_accelerate = sum_order_refine_accelerate + (new_alpha1[:, None] * binary_list[1] + mean_list[1][:, None]) * mask


    return sum_order_refine_accelerate

@torch.no_grad()
def high_order_residual_alternating(x, mask, order=2, num_iters=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list = []
    mean_list = []
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        mean_list.append(mean_tensor_all.clone())
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
    new_matrix -= (mean_list[0][:, None] + mean_list[1][:, None])
    new_matrix_expanded = new_matrix.unsqueeze(-1)
    for k in range(num_iters):
        # refine one step
        new_alpha0 = 1. / (torch.sum(binary_list[0] * mask * binary_list[0] * mask, dim=1) + 1e-8) * torch.sum(binary_list[0] * mask * (new_matrix - alpha_list[1][:, None] * binary_list[1] * mask), dim=1)
        new_alpha1 = 1. / (torch.sum(binary_list[1] * mask * binary_list[1] * mask, dim=1) + 1e-8) * torch.sum(binary_list[1] * mask * (new_matrix - new_alpha0[:, None] * binary_list[0] * mask), dim=1)

        # alternating
        v = torch.stack([-new_alpha0 - new_alpha1, -new_alpha0 + new_alpha1, 
                    new_alpha0 - new_alpha1, new_alpha0 + new_alpha1], dim=1)
        v_expanded = v.unsqueeze(1)

        differences = torch.abs(new_matrix_expanded - v_expanded)
        min_indices = torch.argmin(differences, dim=-1)

        new_binary0 = torch.ones_like(min_indices)
        new_binary0[(min_indices == 0) | (min_indices == 1)] = -1
        new_binary1 = torch.ones_like(min_indices)
        new_binary1[(min_indices == 0) | (min_indices == 2)] = -1 

        sum_order_alternating = torch.zeros_like(x) + (new_alpha0[:, None] * new_binary0 + new_alpha1[:, None] * new_binary1 + mean_list[0][:, None] + mean_list[1][:, None]) * mask

        # update parameters
        alpha_list[0] = new_alpha0.clone()
        alpha_list[1] = new_alpha1.clone()
        binary_list[0] = new_binary0.clone()
        binary_list[1] = new_binary1.clone()


    return sum_order_alternating

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

    return sum_order_alternating

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
    refine_mean_den = 2*torch.sum(S * MM, dim=(1,2)) + 1e-10
    masked_B0 = binary_list[0] * mask
    BB0 = masked_B0[:, :, None] * masked_B0[:, None, :]
    new_alpha0_den = 2*torch.sum(S * BB0, dim=(1,2)) + 1e-10
    masked_B1 = binary_list[1] * mask
    BB1 = masked_B1[:, :, None] * masked_B1[:, None, :]
    new_alpha1_den = 2*torch.sum(S * BB1, dim=(1,2)) + 1e-10
    for kk in range(iter2):
        # X error update mean
        R = new_matrix - (new_alpha0[:, None] * new_binary0 - new_alpha1[:, None] * new_binary1) * mask
        WB = R[:, :, None] + R[:, None, :]
        refine_mean = torch.sum(S * WB * MM, dim=(1,2)) / refine_mean_den

        # X error update alpha
        masked_W_mu = new_matrix - refine_mean[:, None] * mask
        BW0 = masked_B0[:, :, None] * masked_W_mu[:, None, :]
        BW0 = BW0 + BW0.transpose(1,2)
        BB01 = masked_B0[:, :, None] * (new_alpha1[:, None] * masked_B1)[:, None, :]
        BB01 = BB01 + BB01.transpose(1,2)
        new_alpha0 = torch.sum(S * (BW0-BB01), dim=(1,2)) / new_alpha0_den

        BW1 = masked_B1[:, :, None] * masked_W_mu[:, None, :]
        BW1 = BW1 + BW1.transpose(1,2)
        BB10 = masked_B1[:, :, None] * (new_alpha0[:, None] * masked_B0)[:, None, :]
        BB10 = BB10 + BB10.transpose(1,2)
        new_alpha1 = torch.sum(S * (BW1-BB10), dim=(1,2)) / new_alpha1_den


    sum_order_alternating = torch.zeros_like(x) + (new_alpha0[:, None] * binary_list[0] + new_alpha1[:, None] * binary_list[1] + refine_mean[:, None]) * mask

    return sum_order_alternating

@torch.no_grad()
def normal_quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Binarization(nn.Module):
    def __init__(self, weight, method="2bit", groupsize=-1):
        super().__init__()
        oc,ic=weight.shape
        if groupsize==-1:
            groupsize=ic
        self.groupsize=groupsize
        self.n_groups=math.ceil(ic/groupsize)
        self.method=method
        self.mean = 0

    def quantize(self, w, mask, order=2, groupi=0, S=None, prev_W=None):
        if self.method=="xnor":
            w_mean = self.mean[groupi]
            w = w - w_mean  # oc, ic
            w = w.sign()
            w = w * self.scale[groupi]
            w+=w_mean
        elif self.method=="braq": # The method used in paper
            # original
            # w = high_order_residual(w, mask, order=order)  

            # use DB-LLM
            # if order == 2:
            #     w = db_llm(w, mask)
            # else:
            #     w = high_order_residual(w, mask, order=order)  

            # # search clip
            # if order == 2:
            #     w = high_order_residual(w, mask, order=order)  
            # else:
            #     w = high_order_residual_search_clip(w, mask, order=order, k=k)

            # # refine 2nd
            # if order == 2:
            #     w = high_order_residual_refine(w, mask, order=order)  
            # else:
            #     w = high_order_residual(w, mask, order=order)  

            # # refine 2nd accelerate
            # if order == 2:
            #     w = high_order_residual_refine_accelerate(w, mask, order=order)  
            # else:
            #     w = high_order_residual(w, mask, order=order)  

            # # alternating 2nd
            # if order == 2:
            #     w = high_order_residual_alternating(w, mask, order=order)  
            # else:
            #     w = high_order_residual(w, mask, order=order)  

            # # alternating 2nd + refine mean 1st
            # if order == 2:
            #     w = high_order_residual_alternating(w, mask, order=order)  
            # else:
            #     w = high_order_residual_refine_mean(w, mask, order=order)  

            # # alternating 2nd + 1st
            # if order == 2:
            #     w = high_order_residual_alternating(w, mask, order=order)  
            # else:
            #     w = high_order_residual_alternating_order1(w, mask, order=order)  

            # # alternating 2nd + 1st
            # if order == 2:
            #     w = high_order_residual_alternating_mean(w, mask, order=order)  
            # else:
            #     w = high_order_residual_alternating_order1(w, mask, order=order)  

            # # alternating 2nd + 1st + x error alpha
            # if order == 2:
            #     w = high_order_residual_alternating_mean(w, mask, order=order)  
            # else:
            #     w = high_order_residual_alternating_order1_x(w, mask, order=order, S=S)  

            # alternating 2nd + 1st + x error alpha (only x error)
            if order == 2:
                w = high_order_residual_alternating_mean(w, mask, order=order)  
            else:
                w = high_order_residual_alternating_order1_xB(w, mask, order=order, S=S)  

            # # alternating 2nd + 1st + x error alpha
            # if order == 2:
            #     w = high_order_residual_alternating_mean(w, mask, order=order)  
            # else:
            #     w = high_order_residual_alternating_order1_x_prev(w, mask, order=order, S=S, prev_W=prev_W) 

        elif self.method=="sign":
            w=(w>0).float()
            w*=self.scale[groupi]
        elif self.method=="rtn":
            w=F.relu(w)
            w_int=(w/self.scale[groupi]).round().clamp(0,1)
            w=w_int*self.scale[groupi]
        elif self.method in ['2bit','4bit']:

            bits = int(self.method[0])
            perchannel = True
            weight = True
            dev = w.device
            maxq = torch.tensor(2 ** bits - 1)
            scale = torch.zeros(1)
            zero = torch.zeros(1)

            if dev != scale.device:
                scale=scale.to(dev)
                zero=zero.to(dev)
                maxq=maxq.to(dev)

            x = w.clone()
            shape = x.shape

            if perchannel:
                if weight:
                    x = x.flatten(1)
                else:
                    if len(shape) == 4:
                        x = x.permute([1, 0, 2, 3])
                        x = x.flatten(1)
                    if len(shape) == 3:
                        x = x.reshape((-1, shape[-1])).t()
                    if len(shape) == 2:
                        x = x.t()
            else:
                x = x.flatten().unsqueeze(0)
            tmp = torch.zeros(x.shape[0], device=dev)
            xmin = torch.minimum(x.min(1)[0], tmp)
            xmax = torch.maximum(x.max(1)[0], tmp)

            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)
            if not perchannel:
                if weight:
                    tmp = shape[0]
                else:
                    tmp = shape[1] if len(shape) != 3 else shape[2]
                scale = scale.repeat(tmp)
                zero = zero.repeat(tmp)

            if weight:
                shape = [-1] + [1] * (len(shape) - 1)
                scale = scale.reshape(shape)
                zero = zero.reshape(shape)
            w = normal_quantize(w, scale, zero, maxq)

        elif self.method=="prune":
            return torch.zeros_like(w)
        return w
