import torch

# error_N = 2048*4096*128
error_N = 3*10*9

def error_computing_x_all_accelerate(origin_matrix, quantized_matrix, mask, S):
    # inps shape [128, 2048, 128]
    R = ((origin_matrix - quantized_matrix) * mask).T
    P = torch.einsum('ik,jk->ij', R, R)
    return torch.sum(P * S) / error_N

index = 0
cnt = 0
loss = 0

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

    iter3 = 3
    SA = diag_S * new_alpha[:, None]
    masked_W_mu = new_matrix - refine_mean[:, None] * mask
    for _ in range(iter3):
        for t in range(new_binary.shape[1]):
            r = masked_W_mu - new_alpha[:, None] * new_binary * mask
            new_binary[:, t] = torch.sign(new_alpha * (r @ S[t] + SA[:,t] * new_binary[:,t]))

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
        # # print(newLoss)
        # # print(prevLoss)
        # # if newLoss > prevLoss:
        # #     print(222)
        # if newLoss < prevLoss:
        #     # print(111)
        #     new_binary = new_binary1
        #     masked_B = new_binary * mask
        #     BB = masked_B[:, :, None] * masked_B[:, None, :]
        #     new_alpha_den = 2*torch.sum(S * BB, dim=(1,2)) + 1e-10

    sum_order_alternating0 = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    # # prev loss
    # alpha_B = (new_alpha[:, None] * new_binary) * mask
    # alpha_BW = torch.einsum('ij,ik->jk', alpha_B, masked_W_mu)
    # alpha_BW = alpha_BW + alpha_BW.T
    # double_alpha_B = torch.einsum('ij,ik->jk', alpha_B, alpha_B)
    # prevLoss = torch.sum((double_alpha_B-alpha_BW) * S)

    R = (new_matrix - (refine_mean[:, None] + new_alpha[:, None] * new_binary) * mask).T
    P = torch.einsum('ik,jk->ij', R, R)
    prevLoss = torch.sum(P * S)
    # print(prevLoss)
    # print(prevLoss2)
    # print('==========')

    # X error update B (approx.)
    new_binary2 = torch.sign(
        (torch.einsum('ij,ki->kj', S.to(torch.float), masked_W_mu) / new_alpha[:,None] \
        - torch.einsum('ij,ki->kj', S.to(torch.float), new_binary) \
        + diag_S * new_binary) / diag_S
    )

    # Loss(B=-1) - Loss(B=+1)
    # r = masked_W_mu - new_alpha[:, None] * new_binary * mask
    new_binary1 = new_binary.clone()
    # dLoss = 4*(new_alpha[:, None] * (torch.einsum('ij,kj->ki', S, r) \
    #                             + diag_S * new_alpha[:, None] * new_binary))[0,0]
    # new_binary1[0,0] = torch.sign(new_alpha[:, None] * (torch.einsum('ij,kj->ki', S, r) \
    #                             + diag_S * new_alpha[:, None] * new_binary))[0,0]
    # new_binary1 = torch.sign(new_alpha[:, None] * (torch.einsum('ij,kj->ki', S, r) \
    #                             + diag_S * new_alpha[:, None] * new_binary))
    iter3 = 3
    SA = diag_S * new_alpha[:, None]
    masked_W_mu = new_matrix - refine_mean[:, None] * mask
    for _ in range(iter3):
        for t in range(new_binary.shape[1]):
            r = masked_W_mu - new_alpha[:, None] * new_binary1 * mask
            new_binary1[:, t] = torch.sign(new_alpha * (r @ S[t] + SA[:,t] * new_binary1[:,t]))
    
    # iter3 = 1
    # SA = diag_S * new_alpha[:, None]
    # for _ in range(iter3):
    #     for t in range(new_binary.shape[1]):
    #         r = masked_W_mu - new_alpha[:, None] * new_binary1 * mask
    #         new_binary1[:, t] = torch.sign(new_alpha * (r @ S[t] + SA[:,t] * new_binary1[:,t]))

    R = (new_matrix - (refine_mean[:, None] + new_alpha[:, None] * new_binary1) * mask).T
    P = torch.einsum('ik,jk->ij', R, R)
    newLoss = torch.sum(P * S)

    # if newLoss < prevLoss:

    # X error update mean
    R = new_matrix - new_alpha[:, None] * new_binary1 * mask
    WB = R[:, :, None] + R[:, None, :]
    refine_mean = torch.sum(S * WB * MM, dim=(1,2)) / refine_mean_den

    # X error update alpha
    masked_W_mu = new_matrix - refine_mean[:, None] * mask
    masked_B = new_binary1 * mask
    BB = masked_B[:, :, None] * masked_B[:, None, :]
    new_alpha_den = 2*torch.sum(S * BB, dim=(1,2)) + 1e-10
    BW = masked_B[:, :, None] * masked_W_mu[:, None, :]
    BW = BW + BW.transpose(1,2)
    new_alpha = torch.sum(S * BW, dim=(1,2)) / new_alpha_den

    # if dLoss < 0:
    #     new_binary1[0,0] = -1
    # else:
    #     new_binary1[0,0] = 1
    
    # new_binary1[:,0] = torch.sign(new_alpha * (torch.einsum('ij,kj->ki', S[0].unsqueeze(0), r).squeeze(1) - S[0,0] * new_alpha))
    # print(c.shape)

    # new loss
    # # masked_W_mu = new_matrix - refine_mean[:, None] * mask
    # alpha_B = (new_alpha[:, None] * new_binary1) * mask
    # alpha_BW = torch.einsum('ij,ik->jk', alpha_B, masked_W_mu)
    # alpha_BW = alpha_BW + alpha_BW.T
    # double_alpha_B = torch.einsum('ij,ik->jk', alpha_B, alpha_B)
    # newLoss = torch.sum((double_alpha_B-alpha_BW) * S)

    R = (new_matrix - (refine_mean[:, None] + new_alpha[:, None] * new_binary1) * mask).T
    P = torch.einsum('ik,jk->ij', R, R)
    newLoss = torch.sum(P * S)

    # print(newLoss)
    # print(prevLoss)
    # if newLoss > prevLoss:
    #     print(222)
    
    sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary1 + refine_mean[:, None]) * mask
    if newLoss < prevLoss:
        pass
        # print(newLoss-prevLoss)
        # print(dLoss)
        # print(111)
        # sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary1 + refine_mean[:, None]) * mask
    elif newLoss > prevLoss:
        # print(newLoss)
        # print(prevLoss)
        # print(new_binary1[0,0])
        # print(new_binary[0,0])
        print(newLoss-prevLoss)
        # print(dLoss)
        print(222)
        # sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    # # Loss(B=-1) - Loss(B=+1)
    # r = masked_W_mu - new_alpha[:, None] * new_binary * mask
    # new_binary1 = torch.sign(new_alpha[:, None] * torch.einsum('ij,kj->ki', S, r))
    

    # new loss
    alpha_B = (new_alpha[:, None] * new_binary2) * mask
    alpha_BW = torch.einsum('ij,ik->jk', alpha_B, masked_W_mu)
    alpha_BW = alpha_BW + alpha_BW.T
    double_alpha_B = torch.einsum('ij,ik->jk', alpha_B, alpha_B)
    newLoss = torch.sum((double_alpha_B-alpha_BW) * S)
    # print(newLoss)
    # print(prevLoss)
    # if newLoss > prevLoss:
    #     print(222)
    if newLoss < prevLoss:
        # print(111)
        sum_order_alternating2 = torch.zeros_like(x) + (new_alpha[:, None] * new_binary2 + refine_mean[:, None]) * mask
    else:
        sum_order_alternating2 = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    # # Loss(B=-1) - Loss(B=+1)
    # r = masked_W_mu - new_alpha[:, None] * new_binary * mask
    # new_binary1 = torch.sign(new_alpha[:, None] * torch.einsum('ij,kj->ki', S, r))
    


    
    # # X error update mean
    # R = new_matrix - new_alpha[:, None] * new_binary * mask
    # WB = R[:, :, None] + R[:, None, :]
    # refine_mean = torch.sum(S * WB * MM, dim=(1,2)) / refine_mean_den

    # # X error update alpha
    # masked_W_mu = new_matrix - refine_mean[:, None] * mask
    # BW = masked_B[:, :, None] * masked_W_mu[:, None, :]
    # BW = BW + BW.transpose(1,2)
    # new_alpha = torch.sum(S * BW, dim=(1,2)) / new_alpha_den

    # sum_order_alternating2 = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating0, sum_order_alternating, sum_order_alternating2

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

    # sum_order_alternating = high_order_residual_alternating_order1_x(x, mask, order=1, S=S)
    sum_order_alternating, sum_order_alternating2, sum_order_alternating3 = high_order_residual_alternating_order1_xB(x, mask, order=1, S=S)
    # sum_order_alternating3, sum_order_alternating4 = high_order_residual_alternating_order1_x(x, ~mask, order=1, S=S)
    # err = error_computing(x, sum_order)
    alternating_err = error_computing_x_all_accelerate(x, sum_order_alternating, mask, S)
    alternating_err2 = error_computing_x_all_accelerate(x, sum_order_alternating2, mask, S)
    alternating_err3 = error_computing_x_all_accelerate(x, sum_order_alternating3, mask, S)
    # alternating_err4 = error_computing_x_all_accelerate(x, sum_order_alternating4, mask, S)
    # alternating_err2 = error_computing_x_all_accelerate(x, sum_order_alternating2+sum_order_alternating4, S)

    # print(err1)
    # print(err)
    # print(refine_err)
    # print(refine_err2)
    # print(alternating_err)
    # print(alternating_err2)
    # print(alternating_err3)
    # # # print(alternating_err4)
    # # # # # # accelerate.append(alternating_err2)
    
    # print('========')
    loss += alternating_err2
    if alternating_err3 < alternating_err2:
        print('wrong')
    # if alternating_err2 < alternating_err3:
    #     cnt += 1
    #     print('true')
    #     print(alternating_err2)
    #     print(alternating_err3)
    #     loss += alternating_err3-alternating_err2
    # if alternating_err < alternating_err2:
    #     torch.set_printoptions(precision=10)
    #     print('wrong')
    #     # if dLoss >= 0:
    #     #     print('wrong')
    #     print(alternating_err)
    #     # print(alternating_err2)
    #     print(alternating_err2)
    #     # print(dLoss)
    # # elif alternating_err > alternating_err4:
    # #     if dLoss <= 0:
    # #         print('wrong')

print(cnt)
print(loss)