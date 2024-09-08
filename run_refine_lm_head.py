import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time

import torch
import torch.nn as nn

from bigptq_x import BRAGPTQ
# from bigptq_x_refine import BRAGPTQ
from binary import Binarization
from modelutils import find_layers
from datautils import get_tokenizer
from lm import EvalLM

import logging
from lm_eval import evaluator

def get_model(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if "opt" in model:
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = 2048
    return model

def error_computing_xx_all(origin_matrix, quantized_matrix, inp, binp):
    # inps shape [128, 2048, 128]
    # origin_matrix *= mask
    # quantized_matrix *= mask
    mse = torch.mean((inp.to(torch.float) @ origin_matrix.T - binp.to(torch.float) @ quantized_matrix.T) ** 2)
    return mse

def error_computing_xx_all_accelerate(origin_matrix, quantized_matrix, S_hat, S_hathat):
    # inps shape [128, 2048, 128]
    WW_hat = torch.einsum('jk,jl->kl', quantized_matrix, origin_matrix)
    WW_hathat = torch.einsum('jk,jl->kl', quantized_matrix, quantized_matrix)
    return torch.sum(S_hathat * WW_hathat - 2 * S_hat * WW_hat)

def refine_lm_head_W(W, S_hat, S_hathat, num_iter=1):
    diag_S_hathat = torch.diag(S_hathat)
    S_hat_W = torch.einsum('kl,jl->kj', S_hat, W)
    W_hat = W.clone()
    for _ in range(num_iter):
        S_hathat_W_hat = torch.einsum('kl,jl->kj', S_hathat, W_hat)
        # W_hat = (S_hat_W - S_hathat_W_hat) / diag_S_hathat + W_hat
        W_hat = W_hat - 4e-9 * (S_hathat_W_hat - S_hat_W).T

    return W_hat

'''
The function is employed to calibrate the quantized models' lm head.
'''
@torch.no_grad()
def refine_lm_head(model, qmodel, dataloader, dev):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

        lm_head = model.lm_head
    layers[0] = layers[0].to(dev)
    lm_head = lm_head.to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    if "llama" in args.model:
        qlayers = qmodel.model.layers
        qmodel.model.embed_tokens = qmodel.model.embed_tokens.to(dev)
        qmodel.model.norm = qmodel.model.norm.to(dev)

        # lm_head = qmodel.lm_head
    qlayers[0] = qlayers[0].to(dev)
    # lm_head = lm_head.to(dev)

    dtype = next(iter(qmodel.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, qmodel.seqlen, qmodel.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    qlayers[0] = Catcher(qlayers[0])
    for batch in dataloader:
        try:
            qmodel(batch[0].to(dev))
        except ValueError:
            pass
    qlayers[0] = qlayers[0].module

    qlayers[0] = qlayers[0].cpu()
    if "llama" in args.model:
        qmodel.model.embed_tokens = qmodel.model.embed_tokens.cpu()
        qmodel.model.norm = qmodel.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    binps = inps.clone()
    bouts = torch.zeros_like(binps)
    attention_mask = cache["attention_mask"]

    print("Ready.")
    
    # quantized model
    for i in range(len(qlayers)):
        qlayer = qlayers[i].to(dev)

        for j in range(args.nsamples):
            bouts[j] = qlayer(binps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        qlayers[i] = qlayer.cpu()
        del qlayer
        torch.cuda.empty_cache()

        binps, bouts = bouts, binps

    print(11111111)

    # fp16 model
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    print(2222222)

    # self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
    #         self.layer.weight.data.dtype
    #     )
    W = lm_head.weight.data.clone()

    # S_hat = torch.einsum('bki,bkj->ij', binps[:,i*block_size:(i+1)*block_size], inps[:,i*block_size:(i+1)*block_size])
    # S_hathat = torch.einsum('bki,bkj->ij', binps[:,i*block_size:(i+1)*block_size], binps[:,i*block_size:(i+1)*block_size])

    # llama-3-8b
    block_size = 128256
    num_block = 128256 // block_size
    W_hat = torch.zeros_like(W, device=W.device)
    loss = 0
    total_err1 = 0
    total_err2 = 0
    print(W.shape)
    print(inps.shape)
    print(f'inps nan: {torch.isnan(inps).any()}')
    print(f'binps nan: {torch.isnan(binps).any()}')
    print(f'W nan: {torch.isnan(W).any()}')
    for i in range(num_block):
        S_hat = torch.einsum('bki,bkj->ij', binps[:,:,i*block_size:(i+1)*block_size], inps[:,:,i*block_size:(i+1)*block_size])
        S_hathat = torch.einsum('bki,bkj->ij', binps[:,:,i*block_size:(i+1)*block_size], binps[:,:,i*block_size:(i+1)*block_size])
        # print(f'S_hat nan: {torch.isnan(S_hat).any()}')
        # print(f'S_hathat nan: {torch.isnan(S_hathat).any()}')

        W_hat[:,i*block_size:(i+1)*block_size] = refine_lm_head_W(W[:,i*block_size:(i+1)*block_size], S_hat, S_hathat)
        err1 = error_computing_xx_all_accelerate(W[:,i*block_size:(i+1)*block_size], W[:,i*block_size:(i+1)*block_size], S_hat, S_hathat)
        err2 = error_computing_xx_all_accelerate(W[:,i*block_size:(i+1)*block_size], W_hat[:,i*block_size:(i+1)*block_size], S_hat, S_hathat)
        total_err1 += err1
        total_err2 += err2
        loss += err1 - err2

    print(f'total err1: {total_err1}')
    print(f'total err2: {total_err2}')
    print(f'loss: {loss}')
    qmodel.lm_head.weight.data = W_hat.to(
        qmodel.lm_head.weight.data.dtype
    )

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=["xnor", "sign", "no", "2bit", "4bit", "prune", "braq"],
        help="quantization method; `xnor` is the method using XNOR to adapt hardware calculation; `prune` is the method used in sparseGPTQ; braq is the method used in BiLLM",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--num_p",
        type=int,
        default=2,
        help="Number of division for non-salient weights",
    )
    parser.add_argument(
        "--salient_metric",
        type=str,
        default="magnitude",
        choices=["magnitude", "hessian"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization.",
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="",
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)

    args = parser.parse_args()
    groupsize = args.blocksize

    device = args.device
    save_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}"
    save_file = "./output/" + save_title.replace("/", "_") + f"_{args.experiment}" + ".pt"
    if args.load_quantized:
        qmodel = get_model(save_file)
        qmodel.eval()
        model = get_model('/data/pretrained/meta-llama/llama-3-8b')
        model.eval()
        tick = time.time()
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=qmodel.seqlen,
        )
        # print(model)
        
        refine_lm_head(model, qmodel, dataloader, device)
        print("refine lm head time:", time.time() - tick, "s")
    else:
        print('Require quantized model!')

    # if args.save:
    #     save_path = os.path.dirname(save_file)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     model.save_pretrained(save_file)

    # 7 zero-shot QA
    results = {}
    if args.tasks != "":
        tokenizer = get_tokenizer(args.model)
        lm = EvalLM(model, tokenizer)
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        print(results)

    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
        )
        print(dataset)
        if "opt" in args.model:
            from eval_ppl_utils import opt_eval

            opt_eval(qmodel, testloader, device, dataset, args.log_wandb)
        elif "llama" in args.model:
            from eval_ppl_utils import llama_eval

            llama_eval(qmodel, testloader, device, dataset, args.log_wandb)


    print(f'Experiment: {args.experiment}')
