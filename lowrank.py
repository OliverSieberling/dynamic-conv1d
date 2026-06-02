# Copyright (c) 2026, Oliver Sieberling

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.heuristics(
    {
        "USE_BF16": lambda args: args["x"].dtype == torch.bfloat16,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BD": BD, "BT": BT}, num_warps=nw, num_stages=ns)
        for BD in [32, 64]
        for BT in [32]
        for nw in [1]
        for ns in [3, 4, 5, 6]
    ],
    key=["D", "W", "R", "NB"],
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    z,
    U,
    y,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    NB: tl.constexpr,
    USE_BF16: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = (i_b * T).to(tl.int64)

    p_z = tl.make_block_ptr(z + bos * R, (T, R), (R, 1), (i_t * BT, 0), (BT, R), (1, 0))
    b_z = tl.load(p_z)

    # Do all loads first
    b_Us = ()
    b_xs = ()
    for i_w in tl.static_range(-W + 1, 1):
        p_U = tl.make_block_ptr(U + (-i_w) * D, (R, D), (W * D, 1), (0, i_d * BD), (R, BD), (1, 0))
        b_Us = b_Us + (tl.load(p_U),)
        p_xi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
        b_xs = b_xs + (tl.load(p_xi, boundary_check=(0,), padding_option="zero"),)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    for k in tl.static_range(0, W):
        if USE_BF16:
            b_w_k = tl.dot(b_z, b_Us[k], out_dtype=tl.float32)
        else:
            b_w_k = tl.dot(b_z, b_Us[k].to(tl.float32), input_precision="ieee")
        b_y += b_xs[k].to(tl.float32) * b_w_k
        if RESIDUAL and k == W - 1:
            b_y += b_xs[k].to(tl.float32)

    p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding="rtne"))


@triton.heuristics(
    {
        "USE_BF16": lambda args: args["x"].dtype == torch.bfloat16,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BD": BD, "BT": BT}, num_warps=nw, num_stages=ns)
        for BD in [32, 64, 128]
        for BT in [32, 64, 128]
        for nw in [2, 4]
        for ns in [3]
    ],
    key=["D", "W", "R", "NB"],
    reset_to_zero=["dU"],
)
@triton.jit
def causal_conv1d_bwd_kernel_dU_dz(
    x,
    z,
    U,
    dy,
    dU,
    dz,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    NB: tl.constexpr,
    USE_BF16: tl.constexpr,
):
    i_t, i_b = tl.program_id(0), tl.program_id(1)
    bos = (i_b * T).to(tl.int64)

    p_z = tl.make_block_ptr(z + bos * R, (T, R), (R, 1), (i_t * BT, 0), (BT, R), (1, 0))
    b_z = tl.load(p_z)

    o_r = tl.arange(0, R)
    b_dz = tl.zeros((BT, R), dtype=tl.float32)

    for i_d in range(0, tl.cdiv(D, BD)):
        p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_dy = tl.load(p_dy, boundary_check=(0, 1))

        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D

        for i_w in tl.static_range(0, W):
            p_U = tl.make_block_ptr(U + i_w * D, (R, D), (W * D, 1), (0, i_d * BD), (R, BD), (1, 0))
            b_U = tl.load(p_U, boundary_check=(1,))

            p_xi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT - i_w, i_d * BD), (BT, BD), (1, 0))
            b_xi = tl.load(p_xi, boundary_check=(0, 1))

            if USE_BF16:
                b_dyx_lo = b_dy * b_xi
                b_dU_partial = tl.dot(tl.trans(b_z), b_dyx_lo, out_dtype=tl.float32)
                b_dz += tl.dot(b_dyx_lo, tl.trans(b_U), out_dtype=tl.float32)
            else:
                b_dyx = b_dy.to(tl.float32) * b_xi.to(tl.float32)
                b_dU_partial = tl.dot(tl.trans(b_z), b_dyx, input_precision="ieee")
                b_dz += tl.dot(b_dyx, tl.trans(b_U).to(tl.float32), input_precision="ieee")

            ptr_dU = dU + o_r[:, None] * (W * D) + i_w * D + o_d[None, :]
            tl.atomic_add(ptr_dU, b_dU_partial, mask=m_d[None, :], sem="relaxed")

    p_dz = tl.make_block_ptr(dz + bos * R, (T, R), (R, 1), (i_t * BT, 0), (BT, R), (1, 0))
    tl.store(p_dz, tl.cast(b_dz, dtype=p_dz.dtype.element_ty, fp_downcast_rounding="rtne"))


@triton.heuristics(
    {
        "USE_BF16": lambda args: args["dy"].dtype == torch.bfloat16,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BD": BD, "BT": BT}, num_warps=nw, num_stages=ns)
        for BD in [64, 128]
        for BT in [32, 64]
        for nw in [1, 2]
        for ns in [3, 4]
    ],
    key=["D", "W", "R", "NB"],
)
@triton.jit
def causal_conv1d_bwd_kernel_dx(
    z,
    U,
    dy,
    dx,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    NB: tl.constexpr,
    USE_BF16: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = (i_b * T).to(tl.int64)

    p_U = tl.make_block_ptr(U, (R, W * D), (W * D, 1), (0, i_d * BD), (R, BD), (1, 0))
    p_dy_fwd = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    p_z_fwd = tl.make_block_ptr(z + bos * R, (T, R), (R, 1), (i_t * BT, 0), (BT, R), (1, 0))

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)

    for _ in range(0, W):
        b_U = tl.load(p_U)
        b_dy_fwd = tl.load(p_dy_fwd, boundary_check=(0,), padding_option="zero").to(tl.float32)
        b_z_fwd = tl.load(p_z_fwd, boundary_check=(0,), padding_option="zero")

        if USE_BF16:
            b_weight = tl.dot(b_z_fwd, b_U, out_dtype=tl.float32)
        else:
            b_weight = tl.dot(b_z_fwd, b_U.to(tl.float32), input_precision="ieee")

        b_dx += b_dy_fwd * b_weight

        p_U = tl.advance(p_U, (0, D))
        p_dy_fwd = tl.advance(p_dy_fwd, (1, 0))
        p_z_fwd = tl.advance(p_z_fwd, (1, 0))

    if RESIDUAL:
        p_dy_resid = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_dx += tl.load(p_dy_resid, boundary_check=(0,), padding_option="zero").to(tl.float32)

    p_dx = tl.make_block_ptr(dx + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding="rtne"))


def _get_params(x):
    B, T, D = x.shape
    NB = triton.cdiv(B * T, 1024)
    return B, T, D, NB


class _LowrankDynamicConvolutionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, U, static_w, residual):
        B, T, D, NB = _get_params(x)
        has_static = static_w is not None
        R_dyn = z.shape[-1]
        W = U.shape[-1] // D

        # Use one additional rank to fuse in static convolution
        if has_static:
            z = F.pad(z, (0, 1), value=1.0)
            U = torch.cat([U, static_w.reshape(1, W * D)], dim=0)
        R = z.shape[-1]

        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(D, meta["BD"]), triton.cdiv(T, meta["BT"]), B)
        causal_conv1d_fwd_kernel[grid](
            x=x,
            z=z,
            U=U,
            y=y,
            B=B,
            T=T,
            D=D,
            W=W,
            R=R,
            NB=NB,
            RESIDUAL=residual,
        )

        ctx.save_for_backward(x, z, U)
        ctx.has_static = has_static
        ctx.R_dyn = R_dyn
        ctx.W = W
        ctx.residual = residual
        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        x, z, U = ctx.saved_tensors
        B, T, D, NB = _get_params(x)
        R = z.shape[-1]
        W = ctx.W

        dx = torch.empty_like(x)
        dz = torch.empty_like(z)
        dU_fp32 = torch.zeros(U.shape, dtype=torch.float32, device=U.device)

        grid_no_d = lambda meta: (triton.cdiv(T, meta["BT"]), B)
        grid_d = lambda meta: (triton.cdiv(D, meta["BD"]), triton.cdiv(T, meta["BT"]), B)

        causal_conv1d_bwd_kernel_dU_dz[grid_no_d](
            x=x,
            z=z,
            U=U,
            dy=dy,
            dU=dU_fp32,
            dz=dz,
            B=B,
            T=T,
            D=D,
            W=W,
            R=R,
            NB=NB,
        )
        causal_conv1d_bwd_kernel_dx[grid_d](
            z=z,
            U=U,
            dy=dy,
            dx=dx,
            B=B,
            T=T,
            D=D,
            W=W,
            R=R,
            NB=NB,
            RESIDUAL=ctx.residual,
        )

        dU = dU_fp32.to(U.dtype)
        if ctx.has_static:
            R_dyn = ctx.R_dyn
            dstatic_w = dU[R_dyn, :].view(W, D)
            return dx, dz[..., :R_dyn], dU[:R_dyn, :], dstatic_w, None
        return dx, dz, dU, None, None


def lowrank_dynamic_convolution(
    x: torch.Tensor,
    z: torch.Tensor,
    U: torch.Tensor,
    static_w: torch.Tensor | None = None,
    residual: bool = False,
) -> torch.Tensor:
    """Lowrank dynamic short convolution, with optional fused (per-channel) static short convolution.

    Instead of taking the full dynamic convolution weights as input, this kernel takes in the low-rank
    inputs z and the second projection U and produces the dynamic convolution weights z @ U on-chip.

    Args:
        x:          [B, T, D] input activations.
        z:          [B, T, R] hidden states after the first projection of low-rank factorization.
        U:          [R, W*D] second projection of low-rank factorization, viewed as (R, W, D).
        static_w:   [W, D] optional per-channel static bias added to the dynamic weight at every (b, t).
        Equivalent to a static convolution filter. Since the second projection of the low-rank factorization
        can be interpreted as R static filters, one can use one of the ranks to perform a static convolution.
        Through this fusion, a rank-R dynamic convolution + static convolution has roughly the same speed as
        a rank-(R+1) dynamic convolution.

    Returns:
        y: [B, T, D] with y[b, t, d] = sum_w (sum_r z[b, t, r] * U[r, w*D + d]) * x[b, t-w, d]
        (for static_w = None, residual = False)
    """
    assert x.is_cuda and x.is_contiguous()
    assert z.is_cuda and z.is_contiguous()
    assert U.is_cuda and U.is_contiguous()
    R = z.shape[-1]
    R_kernel = R + 1 if static_w is not None else R  # static_w is fused into dynamic convolution using one rank
    assert R_kernel & (R_kernel - 1) == 0, "rank R must be a power of 2. If static_w is set R+1 must be a power of 2."
    if static_w is not None:
        assert static_w.is_cuda and static_w.is_contiguous()
        assert static_w.dtype == z.dtype == U.dtype
        assert static_w.shape == (U.shape[-1] // x.shape[-1], x.shape[-1])
    return _LowrankDynamicConvolutionFn.apply(x, z, U, static_w, residual)
