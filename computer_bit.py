masks = [0 for _ in range(5)]
cnt = 0
logfile0 = '/data/user/zhitengli/BiLLM/log/_data_pretrained_meta-llama_llama-3-8b_c4_braq_128_hessian.log'
logfile1 = '/data/user/zhitengli/BiLLM/log/_data_pretrained_meta-llama_llama-3-8b_c4_braq_128_hessian_aternating_accelerate_iter15_alternating_mean_order1_accelerate_group.log'
logfile2 = '/data/user/zhitengli/BiLLM/log/_data_pretrained_meta-llama_llama-3-8b_c4_braq_128_hessian_aternating_accelerate_iter15_alternating_mean_order1_group_iter3.log'
with open(logfile0, 'r') as file:
    for line in file:
        if 'mask' in line:
            for i in range(5):
                masks[i] += float(line.split(f'mask{i+1} ratio: ')[1].split(',')[0])
                cnt += 1

cnt /= 5
for i in range(5):
    masks[i] /= cnt
print(masks)

bit = masks[0] * 2 + 1 - masks[0]
print(bit)


