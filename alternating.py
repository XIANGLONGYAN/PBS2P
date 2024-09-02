import torch

def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    return mse

index = 0
def high_order_residual_refine(x, mask, order=2):
    sum_order = torch.zeros_like(x)
    sum_order1 = torch.zeros_like(x)
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

        if od == 0:
            sum_order1 += sum_order

    # # refine one step [!BUG]
    # print(alpha_list[0].shape)
    # sum_order_refine = torch.zeros_like(x)
    # new_matrix = x.clone() * mask
    # new_matrix -= mean_list[0][:, None]
    # new_alpha0 = torch.linalg.inv(binary_list[0].T @ binary_list[0]) @ binary_list[0].T @ new_matrix
    # sum_order_refine = sum_order_refine + new_alpha0[:, None] * binary_list[0] + mean_list[0][:, None]

    # new_matrix -= alpha_list[0][:, None] * binary_list[0] * mask
    # new_matrix -= mean_list[1][:, None] 
    # new_alpha1 = torch.linalg.inv(binary_list[1].T @ binary_list[1]) @ binary_list[1].T @ new_matrix
    # sum_order_refine = sum_order_refine + new_alpha1[:, None] * binary_list[1] + mean_list[1][:, None]

    # refine one step
    sum_order_refine = torch.zeros_like(x)
    new_matrix = x.clone() * mask
    new_matrix -= (mean_list[0][:, None] + mean_list[1][:, None])
    new_alpha0 = 1. / (torch.sum(binary_list[0] * binary_list[0], dim=1) + 1e-8) * torch.sum(binary_list[0] * (new_matrix - alpha_list[1][:, None] * binary_list[1] * mask), dim=1)
    sum_order_refine = sum_order_refine + (new_alpha0[:, None] * binary_list[0] + mean_list[0][:, None]) * mask

    new_alpha1 = 1. / (torch.sum(binary_list[1] * binary_list[1], dim=1) + 1e-8) * torch.sum(binary_list[1] * (new_matrix - alpha_list[0][:, None] * binary_list[0] * mask), dim=1)
    sum_order_refine = sum_order_refine + (new_alpha1[:, None] * binary_list[1] + mean_list[1][:, None]) * mask

    # refine one step (accelerate way)
    sum_order_refine2 = torch.zeros_like(x)
    new_matrix2 = x.clone() * mask
    new_matrix2 -= (mean_list[0][:, None] + mean_list[1][:, None])
    new_alpha0 = 1. / (torch.sum(binary_list[0] * binary_list[0], dim=1) + 1e-8) * torch.sum(binary_list[0] * (new_matrix2 - alpha_list[1][:, None] * binary_list[1] * mask), dim=1)
    sum_order_refine2 = sum_order_refine2 + (new_alpha0[:, None] * binary_list[0] + mean_list[0][:, None]) * mask

    new_alpha1 = 1. / (torch.sum(binary_list[1] * binary_list[1], dim=1) + 1e-8) * torch.sum(binary_list[1] * (new_matrix2 - new_alpha0[:, None] * binary_list[0] * mask), dim=1)
    sum_order_refine2 = sum_order_refine2 + (new_alpha1[:, None] * binary_list[1] + mean_list[1][:, None]) * mask

    # alternating
    sum_order_alternating = torch.zeros_like(x)
    new_matrix_expanded = new_matrix.unsqueeze(-1)
    v = torch.stack([-new_alpha0 - new_alpha1, -new_alpha0 + new_alpha1, 
                 new_alpha0 - new_alpha1, new_alpha0 + new_alpha1], dim=1)
    v_expanded = v.unsqueeze(1)

    differences = torch.abs(new_matrix_expanded - v_expanded)
    min_indices = torch.argmin(differences, dim=-1)

    new_binary0 = torch.ones_like(min_indices)
    new_binary0[(min_indices == 0) | (min_indices == 1)] = -1
    new_binary1 = torch.ones_like(min_indices)
    new_binary1[(min_indices == 0) | (min_indices == 2)] = -1 

    sum_order_alternating = sum_order_alternating + (new_alpha0[:, None] * new_binary0 + new_alpha1[:, None] * new_binary1 + mean_list[0][:, None] + mean_list[1][:, None]) * mask


    return sum_order1, sum_order, sum_order_refine, sum_order_refine2, sum_order_alternating


# def high_order_residual_alternating(x, mask, order=2, num_iters=5):
#     sum_order = torch.zeros_like(x)
#     new_matrix = x.clone()
#     new_matrix = new_matrix * mask
#     global index
#     index += 1
#     binary_list = []
#     alpha_list = []
#     mean_list = []
#     for od in range(order):
#         residual = new_matrix - sum_order
#         masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

#         mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
#         mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
#         mean_list.append(mean_tensor_all.clone())
#         masked_x_tensor -= mean_tensor_all[:, None]
#         scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
#         scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)
#         alpha_list.append(scale_tensor_all.clone())

#         binary = torch.sign(masked_x_tensor)
#         binary_list.append(binary.clone())
#         binary *= scale_tensor_all[:, None]
#         binary += mean_tensor_all[:, None]
#         sum_order = sum_order + binary*mask

#     new_matrix = x.clone() * mask
#     new_matrix -= (mean_list[0][:, None] + mean_list[1][:, None])
#     new_matrix_expanded = new_matrix.unsqueeze(-1)
#     for k in range(num_iters):
#         # refine one step
#         new_alpha0 = 1. / (torch.sum(binary_list[0] * mask * binary_list[0] * mask, dim=1) + 1e-8) * torch.sum(binary_list[0] * mask * (new_matrix - alpha_list[1][:, None] * binary_list[1] * mask), dim=1)
#         new_alpha1 = 1. / (torch.sum(binary_list[1] * mask * binary_list[1] * mask, dim=1) + 1e-8) * torch.sum(binary_list[1] * mask * (new_matrix - alpha_list[0][:, None] * binary_list[0] * mask), dim=1)

#         # alternating
#         v = torch.stack([-new_alpha0 - new_alpha1, -new_alpha0 + new_alpha1, 
#                     new_alpha0 - new_alpha1, new_alpha0 + new_alpha1], dim=1)
#         v_expanded = v.unsqueeze(1)

#         differences = torch.abs(new_matrix_expanded - v_expanded)
#         min_indices = torch.argmin(differences, dim=-1)

#         new_binary0 = torch.ones_like(min_indices)
#         new_binary0[(min_indices == 0) | (min_indices == 1)] = -1
#         new_binary1 = torch.ones_like(min_indices)
#         new_binary1[(min_indices == 0) | (min_indices == 2)] = -1 

#         sum_order_alternating = torch.zeros_like(x) + (new_alpha0[:, None] * new_binary0 + new_alpha1[:, None] * new_binary1 + mean_list[0][:, None] + mean_list[1][:, None]) * mask

#         # update parameters
#         alpha_list[0] = new_alpha0.clone()
#         alpha_list[1] = new_alpha1.clone()
#         binary_list[0] = new_binary0.clone()
#         binary_list[1] = new_binary1.clone()


#     return sum_order, sum_order_alternating


def high_order_residual_alternating_mean(x, mask, order=2, num_iters=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list = []
    refine_mean = torch.zeros(x.shape[0])
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

# x = torch.tensor([
#     [1,2,-4],
#     [7,-0.0001,0.05]
# ])

naive = []

for seed in range(40):
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

    
    accelerate = []


    sum_order, sum_order_alternating = high_order_residual_alternating_mean(x, mask, order=2)
    err = error_computing(x, sum_order)
    alternating_err = error_computing(x, sum_order_alternating)

    # print(err1)
    print(err)
    # print(refine_err)
    # print(refine_err2)
    print(alternating_err)
    naive.append(alternating_err)
    # accelerate.append(alternating_err2)
    
    print('========')
    # if refine_err2 < alternating_err:
    #     print('wrong')

# print(naive)