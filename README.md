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

## Example: Adding Low-Rank Dynamic Short Convolutions to QKV
The following example shows how to add low-rank dynamic short convolutions to the queries, keys, and values of a Transformer attention layer.
```python
from lowrank import lowrank_dynamic_convolution

class LowRankDynamicShortConvolution(nn.Module):
      """Low-rank dynamic short convolution with bias term as in the paper."""

      def __init__(self, d_model: int, R: int, W: int):
            super().__init()

            # Fused bias term is an additional rank
            if ((R+1) & R) != 0: raise ValueError("R+1 must be a power of two")

            # Low-rank factorization
            self.proj_in = nn.Linear(d_model, R, bias=False)
            self.proj_out = nn.Parameter(torch.zeros(R, W * d_model))
            self.proj_out_bias = nn.Parameter(torch.empty(W, d_model))
      
            # Initialize bias Kaiming-uniform
            nn.init.uniform_(self.proj_out_bias, -1.0 / math.sqrt(W), +1.0 / math.sqrt(W))

      def forward(self, x, pre_z: torch.Tensor) -> torch.Tensor:
            z = self.proj_in(pre_z)

            return lowrank_dynamic_convolution(
                  x=x.contiguous(),
                  z=z.contiguous(),
                  U=self.proj_out,
                  static_w=self.proj_out_bias,
                  residual=True,
            )
```

Create separate modules for the query, key, and value convolutions:
```python
D = 2048

dynamic_conv_q = LowRankDynamicShortConvolution(D, R=15, W=4)
dynamic_conv_k = LowRankDynamicShortConvolution(D, R=15, W=4)
dynamic_conv_v = LowRankDynamicShortConvolution(D, R=15, W=4)
```

Inside a pre-norm attention layer, apply the convolutions after the QKV projection and before RoPE:
```python
x_norm = norm(x)

# Standard QKV projection
q, k, v = c_attn(x_norm).split(D, dim=-1)

# Apply dynamic convolutions
q = dynamic_conv_q(q, x_norm)
k = dynamic_conv_k(k, x_norm)
v = dynamic_conv_v(v, x_norm)

# Reshape into heads
q = q.view(B, T, H, head_dim).transpose(1, 2)
k = k.view(B, T, H, head_dim).transpose(1, 2)
v = v.view(B, T, H, head_dim).transpose(1, 2)

# Apply RoPE
q, k = apply_rope(q, k)

# Apply attention
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
out = out.transpose(1, 2).contiguous().view(B, T, D)
```
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
