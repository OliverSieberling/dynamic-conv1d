import torch
import torch.nn.functional as F


def headwise_dynamic_convolution_ref(x, weight, static_w=None, residual=False):
    B, T, D = x.shape
    _, _, H, W = weight.shape
    assert D % H == 0, f"D={D} must be divisible by H={H}"
    head_size = D // H

    y = torch.zeros_like(x, dtype=torch.float32)
    for k in range(W):
        x_shift = F.pad(x, (0, 0, k, 0))[:, :T, :].float()
        w_tap = weight[:, :, :, k].repeat_interleave(head_size, dim=2).float()
        if static_w is not None:
            w_tap = w_tap + static_w[k].float()
        y = y + w_tap * x_shift
    if residual:
        y = y + x.float()
    return y.to(x.dtype)


def lowrank_dynamic_convolution_ref(x, z, U, static_w=None, residual=False):
    B, T, D = x.shape
    R, WD = U.shape
    assert WD % D == 0, f"Last dim. of U ({WD}) must be divisible by D={D}"
    W = WD // D

    K = (z.float() @ U.float()).view(B, T, W, D)
    if static_w is not None:
        K = K + static_w.float()

    y = torch.zeros_like(x, dtype=torch.float32)
    for k in range(W):
        x_shift = F.pad(x, (0, 0, k, 0))[:, :T, :].float()
        y = y + K[:, :, k, :] * x_shift
    if residual:
        y = y + x.float()
    return y.to(x.dtype)
