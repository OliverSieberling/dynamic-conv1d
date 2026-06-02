import torch

from headwise import headwise_dynamic_convolution as hw
from lowrank import lowrank_dynamic_convolution as lr
from reference import headwise_dynamic_convolution_ref as hw_ref
from reference import lowrank_dynamic_convolution_ref as lr_ref


DTYPES = [torch.bfloat16, torch.float32]
WIDTHS = [2, 4]  # convolution width
HEAD_SIZES = [1, 16]  # head size for headwise kernel
RANKS = [16]  # rank for lowrank kernel (will use R-1 when static convolution is fused)
RESIDUAL = [False, True]  # whether to fuse in residual connection
STATIC = [False, True]  # whether to fuse in static convolution
B, T, D = 2, 512, 256

TOL = {torch.float32: (1e-4, 1e-3), torch.bfloat16: (4e-1, 5e-1)}


def _check(kernel, ref, src, dy, dtype, residual):
    atol, rtol = TOL[dtype]
    refs = [t.clone().requires_grad_() if t is not None else None for t in src]
    krns = [t.to(dtype).detach().requires_grad_() if t is not None else None for t in src]
    yr = ref(*refs, residual=residual)
    yr.backward(dy)
    yk = kernel(*krns, residual=residual)
    yk.backward(dy.to(dtype))
    pairs = [(yk, yr)] + [(k.grad, r.grad) for k, r in zip(krns, refs) if k is not None]
    for a, b in pairs:
        torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)


def main():
    assert torch.cuda.is_available()
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(B, T, D, device="cuda", generator=g)
    dy = torch.randn(B, T, D, device="cuda", generator=g)
    n = 0
    for dtype in DTYPES:
        for residual in RESIDUAL:
            for W in WIDTHS:
                sw_full = torch.randn(W, D, device="cuda", generator=g) * 0.1
                for hs in HEAD_SIZES:
                    w = torch.randn(B, T, D // hs, W, device="cuda", generator=g) * 0.1
                    for has_static in STATIC:
                        sw = sw_full if has_static else None
                        _check(hw, hw_ref, (x, w, sw), dy, dtype, residual)
                        n += 1
                for R in RANKS:
                    z_full = torch.randn(B, T, R, device="cuda", generator=g) * (0.1 / R**0.5)
                    U_full = torch.randn(R, W * D, device="cuda", generator=g)
                    for has_static in STATIC:
                        z = z_full[..., :-1].contiguous() if has_static else z_full
                        U = U_full[:-1, :].contiguous() if has_static else U_full
                        sw = sw_full if has_static else None
                        _check(lr, lr_ref, (x, z, U, sw), dy, dtype, residual)
                        n += 1
    print(f"All {n} configs passed")


if __name__ == "__main__":
    main()
