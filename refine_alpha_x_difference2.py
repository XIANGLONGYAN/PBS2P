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

        # # X error update B
        # SA = diag_S * new_alpha[:, None]
        # iter3 = 1
        # for _ in range(iter3):
        #     for t in range(new_binary.shape[1]):
        #         r = masked_W_mu - new_alpha[:, None] * new_binary * mask
        #         new_binary[:, t] = torch.sign(new_alpha * (r @ S[t].to(torch.float) + SA[:,t] * new_binary[:,t]))

    sum_order_alternating2 = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating, sum_order_alternating2

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

        # X error update B
        SA = diag_S * new_alpha[:, None]
        iter3 = 1
        for _ in range(iter3):
            for t in range(new_binary.shape[1]):
                r = masked_W_mu - new_alpha[:, None] * new_binary * mask
                new_binary[:, t] = torch.sign(new_alpha * (r @ S[t].to(torch.float) + SA[:,t] * new_binary[:,t]))

    sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating

loss0 = 0
loss1 = 0
loss2 = 0
for seed in range(200):
    torch.manual_seed(seed)

    x = torch.randn(8,9) * 3.2 + 4.567


    # mask = torch.tensor([
    #     [1,1,1],
    #     [0,0,0]
    # ],dtype=torch.bool)

    mask = torch.randint(1, 2, (8, 9)).bool()
    mask[:, [0,3,4]] = 0
    # print(mask)

    # sum_order1, sum_order, sum_order_refine, sum_order_refine2, sum_order_alternating = high_order_residual_refine(x, mask, order=2)
    # err1 = error_computing(x, sum_order1)
    # err = error_computing(x, sum_order)
    # refine_err = error_computing(x, sum_order_refine)
    # refine_err2 = error_computing(x, sum_order_refine2)
    # alternating_err = error_computing(x, sum_order_alternating)

    inp = torch.randn(3,10,9)
    S = torch.einsum('bki,bkj->ij', inp, inp)

    sum_order, sum_order_alternating = high_order_residual_alternating_order1_x(x, mask, order=1, S=S)
    sum_order_alternating2 = high_order_residual_alternating_order1_xB(x, mask, order=1, S=S)

    alternating_err0 = error_computing_x_all_accelerate(x, sum_order, mask, S)
    alternating_err1 = error_computing_x_all_accelerate(x, sum_order_alternating, mask, S)
    alternating_err2 = error_computing_x_all_accelerate(x, sum_order_alternating2, mask, S)
    loss0 += alternating_err0
    loss1 += alternating_err1
    loss2 += alternating_err2

    # print(alternating_err0)
    # print(alternating_err1)
    # print(alternating_err2)
    # print('===========')

    if alternating_err1 < alternating_err2:
        print('wrong')

print(loss0)
print(loss1)
print(loss2)