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

@torch.no_grad()
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

    # return sum_order
    return sum_order_alternating

loss0 = 0
loss = 0
for seed in range(400):
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



    sum_order, sum_order_alternating = high_order_residual_alternating_order1_x(x, mask, order=1)
    sum_order_alternating2 = high_order_residual_alternating_order1_x_rc(x, mask, order=1)
    err = error_computing(x, sum_order)
    alternating_err = error_computing(x, sum_order_alternating)
    alternating_err2 = error_computing(x, sum_order_alternating2)

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
    if alternating_err < alternating_err2:
        print('wrong')
    else:
        # loss += alternating_err - alternating_err2
        loss += alternating_err2

print(loss0)
print(loss)

