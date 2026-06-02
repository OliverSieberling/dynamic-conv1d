# Copyright (c) 2026, Oliver Sieberling

import torch
import triton
import triton.language as tl


NUM_WARPS_AUTOTUNE = [2, 4, 8]
BT_AUTOTUNE = [4, 8, 16, 32]
BD_AUTOTUNE = [32, 64, 128]
NUM_STAGES_AUTOTUNE = [3]


@triton.autotune(
    configs=[
        triton.Config({"BT": BT, "BD": BD}, num_warps=num_warps, num_stages=num_stages)
        for BT in BT_AUTOTUNE
        for BD in BD_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in NUM_STAGES_AUTOTUNE
    ],
    key=["D", "H", "W", "HAS_STATIC"],
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    static_w,
    B,
    T,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    HAS_STATIC: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    tl.static_assert(BD % HEAD_SIZE == 0, "BD must be divisible by HEAD_SIZE")
    BH: tl.constexpr = BD // HEAD_SIZE

    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = (i_b * T).to(tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    o_h = i_d * BH + tl.arange(0, BH)
    m_d = o_d < D
    m_h = o_h < H

    o_w = tl.arange(0, BW) + W - BW
    m_w = o_w >= 0

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_w = tl.load(
        weight + i_b * T * H * W + o_t[:, None, None] * (H * W) + o_h[None, :, None] * W + o_w[None, None, :],
        mask=m_t[:, None, None] & m_h[None, :, None] & m_w[None, None, :],
        other=0,
    ).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    for i_w in tl.static_range(-W + 1, 1):
        p_xi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
        b_xi = tl.load(p_xi, boundary_check=(0, 1)).to(tl.float32)

        # Extract tap and replicate weights across the head
        w_tap = tl.sum(b_w * (o_w == -i_w)[None, None, :], 2)
        w_tap_bd = tl.reshape(
            tl.broadcast_to(w_tap[:, :, None], (BT, BH, HEAD_SIZE)),
            (BT, BD),
        )

        if HAS_STATIC:
            b_sw = tl.load(static_w + (-i_w) * D + o_d, mask=m_d, other=0.0).to(tl.float32)
            b_y += b_xi * (w_tap_bd + b_sw[None, :])
        else:
            b_y += b_xi * w_tap_bd

        if RESIDUAL and i_w == 0:
            b_y += b_xi

    p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(
        p_y,
        tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.autotune(
    configs=[
        triton.Config({"BT": BT, "BD": BD}, num_warps=num_warps, num_stages=num_stages)
        for BT in BT_AUTOTUNE
        for BD in BD_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in NUM_STAGES_AUTOTUNE
    ],
    key=["D", "H", "W", "HAS_STATIC"],
    reset_to_zero=["dstatic_w"],
)
@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    weight,
    static_w,
    dy,
    dx,
    dw,
    dstatic_w,
    B,
    T,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    HAS_STATIC: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    tl.static_assert(BD % HEAD_SIZE == 0, "BD must be divisible by HEAD_SIZE")
    BH: tl.constexpr = BD // HEAD_SIZE

    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = (i_b * T).to(tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    o_h = i_d * BH + tl.arange(0, BH)
    m_d = o_d < D
    m_h = o_h < H

    p_dy_t = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    b_dy_t = tl.load(p_dy_t, boundary_check=(0, 1)).to(tl.float32)

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)

    b_dw_all = tl.zeros((BT, BH, BW), dtype=tl.float32)
    o_w_inner = tl.arange(0, BW)

    for i_w in tl.static_range(0, W):
        p_xi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT - i_w, i_d * BD), (BT, BD), (1, 0))
        b_xi = tl.load(p_xi, boundary_check=(0, 1)).to(tl.float32)

        b_dw_k = tl.sum(
            tl.reshape(b_dy_t, (BT, BH, HEAD_SIZE)) * tl.reshape(b_xi, (BT, BH, HEAD_SIZE)),
            axis=2,
        )
        b_dw_all += b_dw_k[:, :, None] * (o_w_inner == i_w).to(tl.float32)[None, None, :]

        if HAS_STATIC:
            b_dstatic = tl.sum(b_dy_t * b_xi, axis=0)
            tl.atomic_add(
                dstatic_w + i_w * D + o_d, b_dstatic, mask=m_d, sem="relaxed"
            )  # TODO: Improve with split-k reduction

        if i_w == 0:
            b_dy_shift = b_dy_t
        else:
            p_dy_shift = tl.make_block_ptr(
                dy + bos * D,
                (T, D),
                (D, 1),
                (i_t * BT + i_w, i_d * BD),
                (BT, BD),
                (1, 0),
            )
            b_dy_shift = tl.load(p_dy_shift, boundary_check=(0, 1)).to(tl.float32)

        o_ti = i_t * BT + i_w + tl.arange(0, BT)
        m_ti = o_ti < T
        b_w_col = tl.load(
            weight + i_b * T * H * W + o_ti[:, None] * (H * W) + o_h[None, :] * W + i_w,
            mask=m_ti[:, None] & m_h[None, :],
            other=0.0,
        ).to(tl.float32)
        b_w_col_bd = tl.reshape(
            tl.broadcast_to(b_w_col[:, :, None], (BT, BH, HEAD_SIZE)),
            (BT, BD),
        )

        if HAS_STATIC:
            b_sw = tl.load(static_w + i_w * D + o_d, mask=m_d, other=0.0).to(tl.float32)
            b_dx += b_dy_shift * (b_w_col_bd + b_sw[None, :])
        else:
            b_dx += b_dy_shift * b_w_col_bd

    if RESIDUAL:
        b_dx += b_dy_t

    b_dw_flat = tl.reshape(b_dw_all, (BT, BH * BW))
    p_dw = tl.make_block_ptr(
        dw + i_b * T * H * BW,
        (T, H * BW),
        (H * BW, 1),
        (i_t * BT, i_d * BH * BW),
        (BT, BH * BW),
        (1, 0),
    )
    tl.store(
        p_dw,
        tl.cast(b_dw_flat, dtype=p_dw.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )

    p_dx = tl.make_block_ptr(dx + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(
        p_dx,
        tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


class _HeadwiseDynamicConvolutionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, static_w, residual):
        B, T, D = x.shape
        H = weight.shape[2]
        W = weight.shape[-1]
        assert D % H == 0, f"D={D} must be divisible by H={H}"
        head_size = D // H
        BW = triton.next_power_of_2(W)
        has_static = static_w is not None

        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(D, meta["BD"]), triton.cdiv(T, meta["BT"]), B)

        static_arg = static_w if has_static else torch.empty(1, device=x.device, dtype=x.dtype)

        causal_conv1d_fwd_kernel[grid](
            x=x,
            y=y,
            weight=weight,
            static_w=static_arg,
            B=B,
            T=T,
            D=D,
            H=H,
            W=W,
            HEAD_SIZE=head_size,
            BW=BW,
            HAS_STATIC=has_static,
            RESIDUAL=residual,
        )

        ctx.save_for_backward(x, weight, static_w if has_static else None)
        ctx.head_size = head_size
        ctx.has_static = has_static
        ctx.residual = residual
        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        x, weight, static_w = ctx.saved_tensors
        has_static = ctx.has_static
        B, T, D = x.shape
        H = weight.shape[2]
        W = weight.shape[-1]
        head_size = ctx.head_size
        BW = triton.next_power_of_2(W)

        dx = torch.empty_like(x)
        dw_buf = torch.empty(B, T, H, BW, device=weight.device, dtype=weight.dtype)

        if has_static:
            if static_w.dtype == torch.float32:
                dstatic_accum = torch.empty_like(static_w)
            else:
                dstatic_accum = torch.empty((W, D), device=static_w.device, dtype=torch.float32)
            dstatic_accum.zero_()
            static_arg = static_w
            dstatic_arg = dstatic_accum
        else:
            static_arg = torch.empty(1, device=x.device, dtype=x.dtype)
            dstatic_arg = torch.empty(1, device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(D, meta["BD"]), triton.cdiv(T, meta["BT"]), B)
        causal_conv1d_bwd_kernel[grid](
            x=x,
            weight=weight,
            static_w=static_arg,
            dy=dy,
            dx=dx,
            dw=dw_buf,
            dstatic_w=dstatic_arg,
            B=B,
            T=T,
            D=D,
            H=H,
            W=W,
            HEAD_SIZE=head_size,
            BW=BW,
            HAS_STATIC=has_static,
            RESIDUAL=ctx.residual,
        )

        dw = dw_buf if W == BW else dw_buf[..., :W].contiguous()

        if has_static:
            dstatic_w = dstatic_accum if dstatic_accum.dtype == static_w.dtype else dstatic_accum.to(static_w.dtype)
        else:
            dstatic_w = None

        return dx, dw, dstatic_w, None


def headwise_dynamic_convolution(
    x: torch.Tensor,
    weight: torch.Tensor,
    static_w: torch.Tensor | None = None,
    residual: bool = False,
) -> torch.Tensor:
    """Headwise dynamic causal short convolution, with optional fused (per-channel) static short convolution.

    Args:
        x:          [B, T, D] input activations.
        weight:     [B, T, H, W] per-token (dynamic) convolution weights. D must be divisible by H.
        Each head spans D // H consecutive channels.
        static_w:   [W, D] optional per-channel static bias added to the dynamic weight at every (b, t).
        Equivalenet to a static convolution filter.

    Returns:
        y: [B, T, D] with y[b, t, d] = sum_w weight[b, t, d // head_size, w] * x[b, t-w, d]
        (for static_w = None, residual = False)
    """

    assert x.is_cuda and x.is_contiguous()
    if static_w is not None:
        assert static_w.is_cuda and static_w.is_contiguous()
        assert static_w.dtype == x.dtype
        assert static_w.shape == (weight.shape[-1], x.shape[-1])
    return _HeadwiseDynamicConvolutionFn.apply(x, weight, static_w, residual)
