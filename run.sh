# python3 run.py /data/pretrained/meta-llama/llama-3-8b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:1" --save --num_p 3 --load_quantized --tasks arc_easy,arc_challenge

# python3 run.py /data/pretrained/meta-llama/llama-3-8b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:1" --save --num_p 3 --experiment aternating_accelerate_iter15_alternating_mean_order1_accelerate_group_percentile_1e-10

python3 run_x.py /data/pretrained/meta-llama/llama-3-8b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:4" --save --num_p 3 --experiment aternating_accelerate_iter15_x0.1_alpha_mean_iter15