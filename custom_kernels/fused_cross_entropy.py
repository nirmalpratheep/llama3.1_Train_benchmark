"""
Fused Cross-Entropy Loss Triton kernel for AMD MI300X.

Standard cross-entropy in PyTorch:
  1. Computes full softmax over vocab dimension (128K for Llama 3.1)
  2. Materializes the entire [batch*seq, vocab_size] probability tensor
  3. Takes log and gathers the target indices

This fused kernel computes log-softmax and NLL loss in a single pass
over the logits, using chunked reduction to handle the large vocabulary
without materializing the full softmax tensor.

For Llama 3.1 with vocab_size=128256:
  Standard: ~2GB intermediate tensor per batch element at bf16
  Fused: O(chunk_size) intermediate memory
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
    def _cross_entropy_fwd_kernel(
        LOGITS,     # [N, V] logits
        TARGETS,    # [N] target indices
        LOSSES,     # [N] output per-sample losses
        MAX_LOGITS, # [N] buffer for max logits (for numerical stability)
        SUM_EXP,    # [N] buffer for sum of exp
        stride_n,   # Stride along N dimension
        V,          # Vocabulary size
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Compute cross-entropy loss for one row (one token position).
        Uses online softmax algorithm to avoid materializing full softmax.
        """
        row_idx = tl.program_id(0)
        target = tl.load(TARGETS + row_idx)

        # Skip padding tokens (target == -100)
        if target < 0:
            tl.store(LOSSES + row_idx, 0.0)
            return

        # Phase 1: Find max logit for numerical stability (online)
        max_val = float('-inf')
        for block_start in range(0, V, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < V
            logits = tl.load(
                LOGITS + row_idx * stride_n + offsets,
                mask=mask,
                other=float('-inf')
            ).to(tl.float32)
            block_max = tl.max(logits, axis=0)
            max_val = tl.maximum(max_val, block_max)

        # Phase 2: Compute sum(exp(logits - max)) (online)
        sum_exp = 0.0
        for block_start in range(0, V, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < V
            logits = tl.load(
                LOGITS + row_idx * stride_n + offsets,
                mask=mask,
                other=float('-inf')
            ).to(tl.float32)
            sum_exp += tl.sum(tl.exp(logits - max_val), axis=0)

        # Phase 3: Compute loss = -(logit[target] - max - log(sum_exp))
        target_logit = tl.load(LOGITS + row_idx * stride_n + target).to(tl.float32)
        log_sum_exp = tl.log(sum_exp)
        loss = -(target_logit - max_val - log_sum_exp)

        tl.store(LOSSES + row_idx, loss)
        tl.store(MAX_LOGITS + row_idx, max_val)
        tl.store(SUM_EXP + row_idx, sum_exp)

    @triton.jit
    def _cross_entropy_bwd_kernel(
        LOGITS,     # [N, V] logits
        TARGETS,    # [N] target indices
        GRAD_LOGITS,  # [N, V] output grad
        MAX_LOGITS, # [N] stored max logits from forward
        SUM_EXP,    # [N] stored sum_exp from forward
        GRAD_LOSS,  # [N] incoming gradient (usually 1/N for mean reduction)
        stride_n,
        V,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Backward pass: grad_logits = grad_loss * (softmax(logits) - one_hot(target))
        """
        row_idx = tl.program_id(0)
        target = tl.load(TARGETS + row_idx)

        if target < 0:
            # Zero gradient for padding
            for block_start in range(0, V, BLOCK_SIZE):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < V
                tl.store(
                    GRAD_LOGITS + row_idx * stride_n + offsets,
                    tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16),
                    mask=mask
                )
            return

        max_val = tl.load(MAX_LOGITS + row_idx).to(tl.float32)
        sum_exp_val = tl.load(SUM_EXP + row_idx).to(tl.float32)
        grad_loss_val = tl.load(GRAD_LOSS + row_idx).to(tl.float32)

        # Compute grad = grad_loss * (softmax - one_hot)
        for block_start in range(0, V, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < V

            logits = tl.load(
                LOGITS + row_idx * stride_n + offsets,
                mask=mask,
                other=float('-inf')
            ).to(tl.float32)

            # softmax = exp(logits - max) / sum_exp
            softmax_val = tl.exp(logits - max_val) / sum_exp_val

            # Subtract 1 at the target position
            is_target = (offsets == target).to(tl.float32)
            grad = grad_loss_val * (softmax_val - is_target)

            tl.store(
                GRAD_LOGITS + row_idx * stride_n + offsets,
                grad.to(tl.bfloat16),
                mask=mask
            )


class FusedCrossEntropyFunction(torch.autograd.Function):
    """Autograd function for fused cross-entropy loss."""

    @staticmethod
    def forward(ctx, logits, targets):
        """
        Args:
            logits: [N, V] float tensor of logits
            targets: [N] long tensor of target indices (-100 for ignore)
        Returns:
            loss: scalar mean loss
        """
        N, V = logits.shape

        # Allocate outputs
        losses = torch.empty(N, dtype=torch.float32, device=logits.device)
        max_logits = torch.empty(N, dtype=torch.float32, device=logits.device)
        sum_exp = torch.empty(N, dtype=torch.float32, device=logits.device)

        # Choose block size based on vocab size
        BLOCK_SIZE = min(triton.next_power_of_2(V), 4096)

        _cross_entropy_fwd_kernel[(N,)](
            logits, targets, losses, max_logits, sum_exp,
            stride_n=logits.stride(0),
            V=V,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Mean reduction (only count non-padding tokens)
        valid_mask = targets >= 0
        n_valid = valid_mask.sum().clamp(min=1)
        loss = losses.sum() / n_valid

        ctx.save_for_backward(logits, targets, max_logits, sum_exp)
        ctx.N = N
        ctx.V = V
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.n_valid = n_valid

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, max_logits, sum_exp = ctx.saved_tensors
        N, V = ctx.N, ctx.V

        grad_logits = torch.empty_like(logits)

        # grad_loss per sample = grad_output / n_valid (for mean reduction)
        grad_loss = torch.full(
            (N,), (grad_output.item() / ctx.n_valid.item()),
            dtype=torch.float32, device=logits.device
        )

        _cross_entropy_bwd_kernel[(N,)](
            logits, targets, grad_logits, max_logits, sum_exp, grad_loss,
            stride_n=logits.stride(0),
            V=V,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )

        return grad_logits, None


class FusedCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with fused Triton kernel.
    Computes log-softmax + NLL loss in a single pass over the logits.
    """

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, seq_len, vocab_size] or [N, vocab_size]
            targets: [batch_size, seq_len] or [N]
        """
        if logits.dim() == 3:
            # [batch, seq, vocab] → [batch*seq, vocab]
            B, S, V = logits.shape
            logits = logits.reshape(-1, V)
            targets = targets.reshape(-1)

        return FusedCrossEntropyFunction.apply(logits.contiguous(), targets.contiguous())
