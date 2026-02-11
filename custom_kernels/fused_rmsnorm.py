"""
Fused RMSNorm Triton kernel for AMD MI300X.

Standard RMSNorm in PyTorch launches multiple kernels:
  1. x.pow(2).mean(-1) → variance (reduction)
  2. x * rsqrt(variance + eps) → normalize
  3. output * weight → scale

This fused kernel does all three in a single GPU launch, eliminating
intermediate global memory reads/writes.

Reference: LlamaRMSNorm from transformers
    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
"""

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _rms_norm_fwd_kernel(
        X,        # Input tensor pointer
        W,        # Weight tensor pointer
        Y,        # Output tensor pointer
        stride,   # Row stride
        N,        # Number of columns (hidden_size)
        eps,      # Epsilon for numerical stability
        BLOCK_SIZE: tl.constexpr,
    ):
        """Forward pass of fused RMSNorm."""
        # Each program instance handles one row
        row_idx = tl.program_id(0)

        # Compute offsets for this row
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        # Load input row
        x_ptrs = X + row_idx * stride + col_offsets
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Compute variance: mean(x^2)
        x_sq = x * x
        variance = tl.sum(x_sq, axis=0) / N

        # Compute rsqrt(variance + eps)
        rstd = 1.0 / tl.sqrt(variance + eps)

        # Normalize
        x_norm = x * rstd

        # Load weight and scale
        w = tl.load(W + col_offsets, mask=mask, other=1.0).to(tl.float32)
        y = x_norm * w

        # Store result
        y_ptrs = Y + row_idx * stride + col_offsets
        tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _rms_norm_bwd_kernel(
        DY,       # Grad output pointer
        X,        # Input tensor pointer
        W,        # Weight tensor pointer
        DX,       # Grad input pointer
        DW_partial,  # Partial grad weight pointer (per-row)
        stride,   # Row stride
        N,        # Number of columns
        eps,      # Epsilon
        BLOCK_SIZE: tl.constexpr,
    ):
        """Backward pass of fused RMSNorm."""
        row_idx = tl.program_id(0)

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        # Load values
        x_ptrs = X + row_idx * stride + col_offsets
        dy_ptrs = DY + row_idx * stride + col_offsets
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + col_offsets, mask=mask, other=1.0).to(tl.float32)

        # Recompute forward quantities
        x_sq = x * x
        variance = tl.sum(x_sq, axis=0) / N
        rstd = 1.0 / tl.sqrt(variance + eps)
        x_norm = x * rstd

        # Grad weight (partial, per-row contribution)
        dw_partial = dy * x_norm
        dw_ptrs = DW_partial + row_idx * stride + col_offsets
        tl.store(dw_ptrs, dw_partial.to(tl.float32), mask=mask)

        # Grad input
        dy_w = dy * w
        # d(x_norm)/dx = rstd * (I - x * x^T / (N * variance))
        # Simplified: dx = rstd * (dy_w - x_norm * mean(dy_w * x_norm))
        mean_dy_xnorm = tl.sum(dy_w * x_norm, axis=0) / N
        dx = rstd * (dy_w - x_norm * mean_dy_xnorm)

        dx_ptrs = DX + row_idx * stride + col_offsets
        tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)


class FusedRMSNormFunction(torch.autograd.Function):
    """Autograd function wrapping the Triton RMSNorm kernels."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        # Flatten to 2D for kernel
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        num_rows, N = x_2d.shape

        # Output tensor
        y = torch.empty_like(x_2d)

        # Determine block size (next power of 2 >= N)
        BLOCK_SIZE = triton.next_power_of_2(N)

        # Launch kernel
        _rms_norm_fwd_kernel[(num_rows,)](
            x_2d, weight, y,
            stride=x_2d.stride(0),
            N=N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x_2d, weight)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_rows = num_rows
        ctx.N = N

        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight = ctx.saved_tensors
        dy_2d = dy.reshape(-1, ctx.N)

        # Grad input
        dx = torch.empty_like(x_2d)
        # Partial grad weight (one row per input row, will be summed)
        dw_partial = torch.empty(ctx.num_rows, ctx.N, dtype=torch.float32, device=x_2d.device)

        _rms_norm_bwd_kernel[(ctx.num_rows,)](
            dy_2d, x_2d, weight, dx, dw_partial,
            stride=x_2d.stride(0),
            N=ctx.N,
            eps=ctx.eps,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )

        # Sum partial weight gradients across rows
        dw = dw_partial.sum(dim=0).to(weight.dtype)

        return dx.reshape(dy.shape), dw, None


class FusedRMSNorm(nn.Module):
    """Drop-in replacement for LlamaRMSNorm using fused Triton kernel."""

    def __init__(self, hidden_size, eps=1e-6, weight=None):
        super().__init__()
        if weight is not None:
            self.weight = weight
        else:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return FusedRMSNormFunction.apply(
            hidden_states, self.weight, self.variance_epsilon
        )

    def extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


def patch_rmsnorm(model):
    """
    Replace all LlamaRMSNorm modules in the model with FusedRMSNorm.
    Preserves the learned weights.
    """
    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ == 'LlamaRMSNorm':
            # Get the parent module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)

            # Create fused replacement with existing weights
            fused = FusedRMSNorm(
                hidden_size=module.weight.shape[0],
                eps=module.variance_epsilon,
                weight=module.weight,
            )

            setattr(parent, parts[-1], fused)
            count += 1

    print(f"  Replaced {count} LlamaRMSNorm → FusedRMSNorm")
    return model
