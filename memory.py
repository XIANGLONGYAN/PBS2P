from run_arb import get_model

# fp16
save_file = '/data/pretrained/meta-llama/llama-2-7b-hf'

model = get_model(save_file)

# # 计算模型的总参数量
# def count_parameters(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     return total_params

# # 计算模型参数占用的存储空间（以字节为单位）
# def calculate_storage_space(model):
#     total_bytes = 0
#     for param in model.parameters():
#         total_bytes += param.numel() * param.element_size()  # param.element_size() 返回每个元素的字节数
#     return total_bytes


# 计算每个模块的参数数量和存储空间
quant_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
def module_params_and_storage(model, method='billm', ratio=0.09, order2group=False):
    total_bytes = 0
    for name, param in model.named_parameters():
        param_count = param.numel()  # 参数个数
        # param_size = param.numel() * param.element_size()  # 参数占用的字节数
        # print(f"Module: {name}")
        # print(f" - Parameters: {param_count}")
        # print(f" - Storage: {param_size / 1024:.2f} KB\n")
        # print(param.element_size())   # 2 Bytes

        flag = False
        for module in quant_modules:
            if module in name:
                total_bytes += param.numel() / 8 * (1-ratio)
                n, m = param.shape
                if method == 'billm' or method == 'arb' or 'method' == 'arb-x':
                    total_bytes += param.numel() / 8             # group bitmap
                    total_bytes += m / 8                         # salient column bitmap
                    total_bytes += (m // 128) * n * 2 * 2 * 2    # 2 groups x 1st order 
                    total_bytes += (m // 128) * n * 2 * 3        # 1 group x 2nd order
                    total_bytes += m * n * 2 * ratio / 8         # ratio 2 bit
                    if order2group:
                        total_bytes += (m // 128) * n * 2 * 3        # 1 group x 2nd order
                elif method == 'pbllm':
                    total_bytes += param.numel() / 8             # group bitmap
                    total_bytes += (m // 128) * n * 2 * 2        # 1st order
                    total_bytes += m * n * ratio                 # ratio 8 bit
                elif method == 'arb-rc':
                    total_bytes += param.numel() / 8             # group bitmap
                    total_bytes += m / 8                         # salient column bitmap
                    total_bytes += (m // 128) * n * 2 * 2        # 2 groups x 1st order (row)
                    total_bytes += (m // 128) * n * 2 * 2        # 1 group x 2nd order (row)
                    total_bytes += m * 2 * 2 * (1-ratio)         # 2 groups x 1st order (column)
                    total_bytes += m * 2 * 2 * ratio             # 1 group x 2nd order (column)
                    total_bytes += m * n * 2 * ratio / 8         # ratio 2 bit
                    if order2group:
                        total_bytes += (m // 128) * n * 2 * 2    # 1 group x 2nd order (row)
                        total_bytes += m * 2 * 2 * ratio         # 1 group x 2nd order (column)

                flag = True
                break

        if not flag:
            total_bytes += param.numel() * param.element_size()

    return total_bytes


print(module_params_and_storage(model, method='billm', ratio=0.08896, order2group=True))
