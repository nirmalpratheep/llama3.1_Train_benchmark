"""
Fused Rotary Positional Embedding (RoPE) Triton kernel for AMD MI300X.

Standard RoPE in PyTorch:
  1. Compute cos/sin from position frequencies → materializes cos/sin tensors
  2. Split x into (x1, x2) halves
  3. result = cat(x1*cos - x2*sin, x1*sin + x2*cos)

This fused kernel avoids materializing the full cos/sin tensors in global
memory and applies the rotation in a single kernel pass.

Supports Llama 3.1 RoPE config (theta=500000, non-interleaved).
"""

import torch
import torch.nn as nn
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _rope_fwd_kernel(
        Q,          # Input query/key tensor [batch, seq, heads, head_dim]
        OUT,        # Output tensor
        COS,        # Precomputed cos cache [seq, head_dim//2]
        SIN,        # Precomputed sin cache [seq, head_dim//2]
        stride_qb,  # Batch stride
        stride_qs,  # Sequence stride
        stride_qh,  # Head stride
        stride_qd,  # Dim stride
        seq_len,    # Sequence length
        n_heads,    # Number of heads
        HEAD_DIM: tl.constexpr,      # Head dimension
        HALF_DIM: tl.constexpr,      # Half of head dimension
        BLOCK_SIZE: tl.constexpr,
    ):
        """Apply rotary embeddings in a single fused kernel."""
        # Program ID maps to (batch, seq_pos, head)
        pid = tl.program_id(0)
        # Decode linear index
        head_idx = pid % n_heads
        remaining = pid // n_heads
        seq_idx = remaining % seq_len
        batch_idx = remaining // seq_len

        # Load the half dimensions
        dim_offsets = tl.arange(0, HALF_DIM)
        dim_mask = dim_offsets < HALF_DIM

        # Base pointer for this (batch, seq, head)
        base_offset = (batch_idx * stride_qb +
                       seq_idx * stride_qs +
                       head_idx * stride_qh)

        # Load first half and second half of the head dimension
        x1_ptrs = Q + base_offset + dim_offsets * stride_qd
        x2_ptrs = Q + base_offset + (dim_offsets + HALF_DIM) * stride_qd

        x1 = tl.load(x1_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptrs, mask=dim_mask, other=0.0).to(tl.float32)

        # Load cos and sin for this position
        cos_ptrs = COS + seq_idx * HALF_DIM + dim_offsets
        sin_ptrs = SIN + seq_idx * HALF_DIM + dim_offsets

        cos_val = tl.load(cos_ptrs, mask=dim_mask, other=1.0).to(tl.float32)
        sin_val = tl.load(sin_ptrs, mask=dim_mask, other=0.0).to(tl.float32)

        # Apply rotation
        out1 = x1 * cos_val - x2 * sin_val
        out2 = x1 * sin_val + x2 * cos_val

        # Store results
        out1_ptrs = OUT + base_offset + dim_offsets * stride_qd
        out2_ptrs = OUT + base_offset + (dim_offsets + HALF_DIM) * stride_qd

        tl.store(out1_ptrs, out1.to(tl.bfloat16), mask=dim_mask)
        tl.store(out2_ptrs, out2.to(tl.bfloat16), mask=dim_mask)

    @triton.jit
    def _rope_bwd_kernel(
        DOUT,       # Grad output
        DX,         # Grad input
        COS,        # Precomputed cos cache
        SIN,        # Precomputed sin cache
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        seq_len,
        n_heads,
        HEAD_DIM: tl.constexpr,
        HALF_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Backward pass for rotary embeddings."""
        pid = tl.program_id(0)
        head_idx = pid % n_heads
        remaining = pid // n_heads
        seq_idx = remaining % seq_len
        batch_idx = remaining // seq_len

        dim_offsets = tl.arange(0, HALF_DIM)
        dim_mask = dim_offsets < HALF_DIM

        base_offset = (batch_idx * stride_qb +
                       seq_idx * stride_qs +
                       head_idx * stride_qh)

        # Load grad output halves
        dy1 = tl.load(DOUT + base_offset + dim_offsets * stride_qd,
                       mask=dim_mask, other=0.0).to(tl.float32)
        dy2 = tl.load(DOUT + base_offset + (dim_offsets + HALF_DIM) * stride_qd,
                       mask=dim_mask, other=0.0).to(tl.float32)

        # Load cos/sin
        cos_val = tl.load(COS + seq_idx * HALF_DIM + dim_offsets,
                          mask=dim_mask, other=1.0).to(tl.float32)
        sin_val = tl.load(SIN + seq_idx * HALF_DIM + dim_offsets,
                          mask=dim_mask, other=0.0).to(tl.float32)

        # Backward of rotation: transpose of rotation matrix
        # out1 = x1*cos - x2*sin → dx1 += dy1*cos, dx2 += -dy1*sin
        # out2 = x1*sin + x2*cos → dx1 += dy2*sin, dx2 += dy2*cos
        dx1 = dy1 * cos_val + dy2 * sin_val
        dx2 = -dy1 * sin_val + dy2 * cos_val

        tl.store(DX + base_offset + dim_offsets * stride_qd,
                 dx1.to(tl.bfloat16), mask=dim_mask)
        tl.store(DX + base_offset + (dim_offsets + HALF_DIM) * stride_qd,
                 dx2.to(tl.bfloat16), mask=dim_mask)


class FusedRoPEFunction(torch.autograd.Function):
    """Autograd function for fused rotary positional embeddings."""

    @staticmethod
    def forward(ctx, x, cos_cache, sin_cache):
        """
        Args:
            x: [batch, seq_len, n_heads, head_dim]
            cos_cache: [seq_len, head_dim//2]
            sin_cache: [seq_len, head_dim//2]
        """
        batch, seq_len, n_heads, head_dim = x.shape
        half_dim = head_dim // 2

        out = torch.empty_like(x)

        # Total work items: one per (batch, seq, head)
        grid = (batch * seq_len * n_heads,)

        _rope_fwd_kernel[grid](
            x, out, cos_cache, sin_cache,
            stride_qb=x.stride(0),
            stride_qs=x.stride(1),
            stride_qh=x.stride(2),
            stride_qd=x.stride(3),
            seq_len=seq_len,
            n_heads=n_heads,
            HEAD_DIM=head_dim,
            HALF_DIM=half_dim,
            BLOCK_SIZE=half_dim,
        )

        ctx.save_for_backward(cos_cache, sin_cache)
        ctx.shape = x.shape

        return out

    @staticmethod
    def backward(ctx, dout):
        cos_cache, sin_cache = ctx.saved_tensors
        batch, seq_len, n_heads, head_dim = ctx.shape
        half_dim = head_dim // 2

        dx = torch.empty_like(dout)

        grid = (batch * seq_len * n_heads,)

        _rope_bwd_kernel[grid](
            dout, dx, cos_cache, sin_cache,
            stride_qb=dout.stride(0),
            stride_qs=dout.stride(1),
            stride_qh=dout.stride(2),
            stride_qd=dout.stride(3),
            seq_len=seq_len,
            n_heads=n_heads,
            HEAD_DIM=head_dim,
            HALF_DIM=half_dim,
            BLOCK_SIZE=half_dim,
        )

        return dx, None, None


def fused_apply_rotary_emb(x, cos, sin):
    """
    Apply fused rotary embeddings.

    Args:
        x: tensor of shape [batch, seq_len, n_heads, head_dim]
        cos: tensor of shape [seq_len, head_dim//2] or broadcastable
        sin: tensor of shape [seq_len, head_dim//2] or broadcastable

    Returns:
        Rotated tensor of same shape as x.
    """
    # Ensure cos/sin are 2D [seq_len, half_dim]
    if cos.dim() > 2:
        cos = cos.squeeze(0).squeeze(0)
    if sin.dim() > 2:
        sin = sin.squeeze(0).squeeze(0)

    # Ensure contiguous
    cos = cos.contiguous()
    sin = sin.contiguous()

    return FusedRoPEFunction.apply(x.contiguous(), cos, sin)


def patch_rope(model):
    """
    Patch the model's rotary embedding application to use the fused kernel.
    This patches the LlamaAttention.forward to intercept the rotary embedding step.
    """
    import types
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        rotate_half,
    )

    count = 0

    def fused_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Fused replacement for apply_rotary_pos_emb."""
        # Standard HuggingFace shape: cos/sin are [batch, 1, seq, dim] or [1, 1, seq, dim]
        # We need [seq, half_dim] for our kernel

        # Expand cos/sin for the sequence
        cos_2d = cos.squeeze(0).squeeze(0)  # [seq_len, dim]
        sin_2d = sin.squeeze(0).squeeze(0)

        # q, k shapes: [batch, n_heads, seq_len, head_dim]
        # Transpose to [batch, seq_len, n_heads, head_dim] for our kernel
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()

        q_out = fused_apply_rotary_emb(q_t, cos_2d, sin_2d)
        k_out = fused_apply_rotary_emb(k_t, cos_2d, sin_2d)

        # Transpose back to [batch, n_heads, seq_len, head_dim]
        return q_out.transpose(1, 2), k_out.transpose(1, 2)

    # Patch the module-level function used by LlamaAttention
    import transformers.models.llama.modeling_llama as llama_module
    llama_module.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    count = 1  # Module-level patch

    print(f"  Patched apply_rotary_pos_emb → fused Triton RoPE")
    return model
