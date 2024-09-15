# arb-rc
# python3 run_arb.py /data/pretrained/meta-llama/llama-3-8b c4 arb-rc --blocksize 128 --salient_metric hessian --device "cuda:7" --save --num_p 1 --order2_group
# python3 run_arb.py /data/pretrained/meta-llama/llama-3-8b c4 arb-rc --blocksize 128 --salient_metric hessian --device "cuda:7" --save --num_p 1

# arb-x
# python3 run_arb.py /data/pretrained/meta-llama/llama-3-8b c4 arb-x --blocksize 128 --salient_metric hessian --device "cuda:7" --save --num_p 1 --order2_group
# python3 run_arb.py /data/pretrained/meta-llama/llama-3-8b c4 arb-x --blocksize 128 --salient_metric hessian --device "cuda:7" --save --num_p 1

# arb
# python3 run_arb.py /data/pretrained/meta-llama/llama-3-8b c4 arb --blocksize 128 --salient_metric hessian --device "cuda:7" --save --num_p 1 --order2_group
python3 run_arb.py /data/pretrained/meta-llama/llama-3-8b c4 arb --blocksize 128 --salient_metric hessian --device "cuda:7" --save --num_p 1

# BiLLM
# python3 run_arb.py /data/pretrained/meta-llama/llama-3-8b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:7" --save --num_p 1