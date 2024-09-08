import torch

def error_computing_xx_all(origin_matrix, quantized_matrix, inp, binp):
    # inps shape [128, 2048, 128]
    # origin_matrix *= mask
    # quantized_matrix *= mask
    mse = torch.mean((inp.to(torch.float) @ origin_matrix.T - binp.to(torch.float) @ quantized_matrix.T) ** 2)
    return mse

def refine_lm_head(W, S_hat, S_hathat, num_iter=5):
    diag_S_hathat = torch.diag(S_hathat)
    S_hat_W = torch.einsum('kl,jl->kj', S_hat, W)
    W_hat = W.clone()
    for _ in range(num_iter):
        S_hathat_W_hat = torch.einsum('kl,jl->kj', S_hathat, W_hat)
        # W_hat = (S_hat_W - S_hathat_W_hat).T / diag_S_hathat + W_hat
        W_hat = W_hat - 0.01 * (S_hathat_W_hat - S_hat_W).T

    return W_hat


loss = 0
for seed in range(200):
    torch.manual_seed(seed)

    inp = torch.randn(3,10,9)
    W = torch.randn(8,9) + 4.567

    binp = torch.randn(3,10,9)

    S_hat = torch.einsum('bki,bkj->ij', binp, inp)
    S_hathat = torch.einsum('bki,bkj->ij', binp, binp)

    err1 = error_computing_xx_all(W, W, inp=inp, binp=binp)

    W_hat = refine_lm_head(W, S_hat, S_hathat)

    err2 = error_computing_xx_all(W, W_hat, inp=inp, binp=binp)

    loss += err1 - err2

    if err2 > err1:
        print('wrong')
        # print(err1)
        # print(err2)
    else:
        # print('true')
        pass

print(loss)