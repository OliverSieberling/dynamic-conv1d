# Dynamic Short Convolutions (Triton)
Triton kernels for causal depthwise separable convolutions with per-token (dynamic) convolution weights.

Two variants:
- `headwise.py`: general dynamic convolution kernel that takes in `[B, T, D]` activations `x` and `[B, T, H, W]` dynamic convolution weights `weights`. Each convolution weight is shared across a head of dimension `D/H`.
- `lowrank.py`: low-rank dynamic convolution kernel that takes in `[B, T, D]` activations `x`, the  `[B, T, R]` low-rank hidden states `z` and the `[R, W*D]` second projection of a low-rank factorization `U`. Here, generating the dynamic convolution weights `weights = z @ U` is fused into the kernel.

Each variant optionally fuses a residual connection and optionally fuses a `[W, D]` static convolution kernel `static_w`.

## Math

```
headwise: y[b, t, d] = sum_w (weight[b, t, d//head_size, w] + static_w[w, d]) * x[b, t-w, d]
lowrank: y[b, t, d] = sum_w ((z @ U).view(B,T,W,D)[b,t,w,d] + static_w[w, d]) * x[b, t-w, d]
```

`+ x[b, t, d]` is added if `residual=True`.

## Usage
`git clone` this repo and import the files directly.

```python
import torch
from headwise import headwise_dynamic_convolution
from lowrank import lowrank_dynamic_convolution

B, T, D, W = 4, 4096, 2048, 4
x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
static_w = torch.randn(W, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

# General dynamic short convolution (headsize = 1)
H = D
weights = torch.randn(B, T, H, W, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = headwise_dynamic_convolution(x, weights, static_w=static_w, residual=True)

# Headwise dynamic short convolution (headsize = 32)
H = 64
weights = torch.randn(B, T, H, W, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = headwise_dynamic_convolution(x, weights, static_w=static_w, residual=True)

# Low-rank dynamic short convolution (rank = 16)
R = 16
z = torch.randn(B, T, R, device="cuda", dtype=torch.bfloat16, requires_grad=True)
U = torch.randn(R, W*D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = lowrank_dynamic_convolution(x, z, U, static_w=None, residual=True)

y.sum().backward()
```

For `lowrank_dynamic_convolution`: `static_w` is fused as an extra rank, so `R+1` must be a power of 2 (e.g. pass `z` of rank 15 + `static_w`). For `static_w=None` `R` must be a power of 2.

## Files

- `headwise.py`   --    triton kernels for head-wise dynamic convolutions
- `lowrank.py`     --    triton kernels for low-rank dynamic convolutions
- `reference.py`   --   PyTorch reference implementations
- `test.py`        --    Correctness check (python test.py)
- `bench.py`       --     Kernel timings (python bench.py)

## Citation
```bibtex
@misc{sieberling2026dynamicshortconvolutionsimprove,
      title={Dynamic Short Convolutions Improve Transformers}, 
      author={Oliver Sieberling and Bharat Runwal and Rameswar Panda and Yoon Kim},
      year={2026},
      eprint={2606.03825},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.03825}, 
}
```

## License
MIT.
