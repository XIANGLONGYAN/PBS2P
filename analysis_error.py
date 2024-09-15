import matplotlib.pyplot as plt

# w_errors_1 = [[0 for _ in range(7)] for _ in range(32)]
# x_errors_1 = [[0 for _ in range(7)] for _ in range(32)]
# w_errors_2 = [[0 for _ in range(7)] for _ in range(32)]
# x_errors_2 = [[0 for _ in range(7)] for _ in range(32)]

w_errors_1 = [0 for _ in range(7*32)]
w_errors_2 = [0 for _ in range(7*32)]
x_errors_1 = [0 for _ in range(7*32)]
x_errors_2 = [0 for _ in range(7*32)]


logfile1 = '/data/user/zhitengli/BiLLM/log/_data_pretrained_meta-llama_llama-3-8b_c4_braq_128_hessian_aternating_accelerate_iter15_#p1_x_group1and2_rc1x_fix.log'
logfile2 = '/data/user/zhitengli/BiLLM/log/_data_pretrained_meta-llama_llama-3-8b_c4_braq_128_hessian_aternating_accelerate_iter15_#p1_x_group1and2_rc1wox_fix.log'
module = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']


cnt = 0
with open(logfile1, 'r') as file:
    for line in file:
        if '- error' in line:
            w_errors_1[cnt] = float(line.split('error ')[1])
        elif 'x error' in line:
            x_errors_1[cnt] = float(line.split('x error ')[1])
            cnt += 1


cnt = 0
with open(logfile2, 'r') as file:
    for line in file:
        if '- error' in line:
            w_errors_2[cnt] = float(line.split('error ')[1])
        elif 'x error' in line:
            x_errors_2[cnt] = float(line.split('x error ')[1])
            cnt += 1

# plt.figure(figsize=(10, 6))
# timestamps = list(range(len(errors)))
# plt.plot(timestamps, errors, marker='.', linestyle='-')
# plt.xlabel('Layer')
# plt.ylabel('Error')
# plt.title('Quantization Error Per Module (Llama-3-8b, #p=3)')
# plt.grid(True)
# # plt.legend()
# plt.tight_layout()
# plt.xticks(rotation=45)
# plt.savefig('./visualization/llama-3-8b_error_p3.png')

timestamps = list(range(len(w_errors_1)//7))
for i in range(7):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps[4:], w_errors_1[i::7][4:], marker='.', linestyle='-', label='w/ x')
    plt.plot(timestamps[4:], w_errors_2[i::7][4:], marker='.', linestyle='-', label='w/o x')
    plt.xlabel('Layer')
    plt.ylabel('Error')
    plt.title(f'Quantization W Error {module[i]} (Llama-3-8b, #p=1)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(f'./visualization/llama-3-8b_error_{module[i]}.png')

timestamps = list(range(len(w_errors_1)//7))
for i in range(7):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, x_errors_1[i::7], marker='.', linestyle='-', label='w/ x')
    plt.plot(timestamps, x_errors_2[i::7], marker='.', linestyle='-', label='w/o x')
    plt.xlabel('Layer')
    plt.ylabel('Error')
    plt.title(f'Quantization X Error {module[i]} (Llama-3-8b, #p=1)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(f'./visualization/llama-3-8b_xerror_{module[i]}.png')


# total_error = [0 for _ in range(32)]
# plt.figure(figsize=(10, 6))
# for i in range(7):
#     plt.plot(timestamps, errors[i::7], marker='.', linestyle='-', label=module[i])
#     for j in range(32):
#         total_error[j] += errors[i::7][j]
# plt.plot(timestamps, total_error, marker='.', linestyle='-', label='Total')
# plt.xlabel('Layer')
# plt.ylabel('Error')
# plt.title(f'Quantization Error Per Module (Llama-3-8b, #p=3)')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.xticks(rotation=45)
# plt.savefig(f'./visualization/llama-3-8b_error_p3_per_module.png')