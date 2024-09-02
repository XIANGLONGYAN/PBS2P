import torch

torch.manual_seed(42)

x = torch.randn(32, 2048, 4096)

w = torch.randn(4096, 4096)

q = torch.zeros(4096, 4096).float()


def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    return mse

def error_computing_x_all(origin_matrix, quantized_matrix, inp):
    # inps shape [128, 2048, 128]
    mse = torch.mean((inp.to(torch.float) @ (origin_matrix - quantized_matrix).T) ** 2)
    return mse

def error_computing_x_all_accelerate(origin_matrix, quantized_matrix, S):
    # inps shape [128, 2048, 128]
    R = (origin_matrix - quantized_matrix).T
    P = torch.einsum('ik,jk->ij', R, R)
    # print(P.shape)   # [128, 128]
    return torch.sum(P * S) / (2048*4096*32)
    # mse = torch.mean((inp.to(torch.float) @ ) ** 2)
    # return mse


# A = torch.tensor([
#     [1., 2.],
#     [3., 4.],
#     [5., 6.]
# ])

# C = torch.einsum('ik,jk->ij', A, A)
# print(C)

inp = x[:, :, :128]
S = torch.einsum('bki,bkj->ij', inp, inp)

print(S.shape)

ori = error_computing_x_all_accelerate(w[:, :128], q[:, :128], S)
print(ori)