import torch
import triton

from headwise import headwise_dynamic_convolution as hw
from lowrank import lowrank_dynamic_convolution as lr

DTYPE = torch.bfloat16
HEAD_SIZES = [1, 4, 16]  # head size for headwise kernel
RANKS = [16]  # rank for lowrank kernel (will use R-1 when static convolution is fused)
RESIDUAL = [False]  # whether to fuse in residual connection
STATIC = [False]  # whether to fuse in static convolution
B, T, D, W = 4, 4096, 2048, 4


def _bench(fn, args, dy, residual=False, trials=5, warmup_ms=500, rep_ms=3000):
    for _ in range(15):
        for a in args:
            a.grad = None
        y = fn(*args, residual=residual)
        y.backward(dy)
    torch.cuda.synchronize()

    def fwd():
        for a in args:
            a.grad = None
        fn(*args, residual=residual)

    def fwdbwd():
        for a in args:
            a.grad = None
        y = fn(*args, residual=residual)
        y.backward(dy)

    f = min(triton.testing.do_bench(fwd, warmup=warmup_ms, rep=rep_ms, return_mode="median") for _ in range(trials))
    fb = min(triton.testing.do_bench(fwdbwd, warmup=warmup_ms, rep=rep_ms, return_mode="median") for _ in range(trials))
    return f, fb


def main():
    assert torch.cuda.is_available()
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(B, T, D, device="cuda", generator=g)
    sw = torch.randn(W, D, device="cuda", generator=g) * 0.1
    dy = torch.randn(B, T, D, device="cuda", generator=g).to(DTYPE)

    def make_inputs(tensors):
        """Cast each tensor to DTYPE and make a fresh grad-enabled leaf copy."""
        return [t.to(DTYPE).detach().clone().requires_grad_() for t in tensors]

    print(f"{torch.cuda.get_device_name(0)}  B={B} T={T} D={D} W={W} {DTYPE}\n")

    for hs in HEAD_SIZES:
        w = torch.randn(B, T, D // hs, W, device="cuda", generator=g) * 0.1
        for has_static in STATIC:
            for residual in RESIDUAL:
                tag = (" +static" if has_static else "") + (" +res" if residual else "")
                args = make_inputs([x, w] + ([sw] if has_static else []))
                f, fb = _bench(hw, args, dy, residual=residual)
                print(f"headwise hs={hs}{tag}  fwd {f * 1000:.1f}us  bwd {(fb - f) * 1000:.1f}us  fb {fb * 1000:.1f}us")

    for R in RANKS:
        z_full = torch.randn(B, T, R, device="cuda", generator=g) * (0.1 / R**0.5)
        U_full = torch.randn(R, W * D, device="cuda", generator=g)
        for has_static in STATIC:
            z = z_full[..., :-1].contiguous() if has_static else z_full
            U = U_full[:-1, :].contiguous() if has_static else U_full
            for residual in RESIDUAL:
                tag = (" +static" if has_static else "") + (" +res" if residual else "")
                args = make_inputs([x, z, U] + ([sw] if has_static else []))
                f, fb = _bench(lr, args, dy, residual=residual)
                print(f"lowrank R={R}{tag}  fwd {f * 1000:.1f}us  bwd {(fb - f) * 1000:.1f}us  fb {fb * 1000:.1f}us")


if __name__ == "__main__":
    main()
