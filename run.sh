# python3 run.py /data/pretrained/meta-llama/llama-3-8b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:1" --save --num_p 3 --load_quantized --tasks arc_easy,arc_challenge

# python3 run.py /data/pretrained/meta-llama/llama-3-8b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:1" --save --num_p 3 --experiment aternating_accelerate_iter15_alternating_mean_order1_accelerate_group_percentile_1e-10

python3 run_x.py /data/pretrained/meta-llama/llama-3-8b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:7" --save --num_p 1 --experiment test

# python3 run_x.py facebook/opt-6.7b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:6" --save --num_p 3 --experiment debug

# python3 run_xx.py /data/pretrained/meta-llama/llama-3-8b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:5" --save --num_p 3 --experiment aternating_accelerate_iter15_x0.1_alpha_mean_iter15_xx2_wogptq

# python3 run_refine_lm_head.py /data/pretrained/meta-llama/llama-3-8b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:7" --num_p 3 --experiment aternating_accelerate_iter15_x0.1_alpha_mean_iter15 --load_quantized