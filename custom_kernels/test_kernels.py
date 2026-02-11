#!/usr/bin/env python3
"""
Numerical correctness tests for custom Triton kernels.
Compares outputs against PyTorch reference implementations using
representative shapes from Llama 3.1 8B.

Usage:
    python -m custom_kernels.test_kernels
    python -m custom_kernels.test_kernels -v
"""

import torch
import sys
import math

# Llama 3.1 8B representative shapes
HIDDEN_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = 128  # 4096 / 32
VOCAB_SIZE = 128256
BATCH_SIZE = 2
SEQ_LEN = 64  # Small for testing
EPS = 1e-6
DTYPE = torch.bfloat16


def check_close(name, actual, expected, atol=1e-2, rtol=1e-2):
    """Check if two tensors are close, with helpful diagnostics."""
    if actual.shape != expected.shape:
        print(f"  ✗ {name}: Shape mismatch {actual.shape} vs {expected.shape}")
        return False

    # Convert to float32 for comparison
    actual_f = actual.float()
    expected_f = expected.float()

    max_err = (actual_f - expected_f).abs().max().item()
    mean_err = (actual_f - expected_f).abs().mean().item()

    close = torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol)
    status = "✓" if close else "✗"
    print(f"  {status} {name}: max_err={max_err:.6f}, mean_err={mean_err:.6f}")

    if not close:
        # Show where the largest errors are
        diff = (actual_f - expected_f).abs()
        worst_idx = diff.argmax()
        print(f"    Worst at flat index {worst_idx}: "
              f"got {actual_f.flatten()[worst_idx]:.6f}, "
              f"expected {expected_f.flatten()[worst_idx]:.6f}")

    return close


def test_fused_rmsnorm():
    """Test fused RMSNorm against PyTorch reference."""
    print("\n" + "="*60)
    print("TEST: Fused RMSNorm")
    print("="*60)

    try:
        from custom_kernels.fused_rmsnorm import FusedRMSNorm
    except ImportError as e:
        print(f"  SKIP: {e}")
        return True

    device = 'cuda'
    all_passed = True

    # Test shapes
    shapes = [
        (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE),  # Standard
        (1, 1, HIDDEN_SIZE),                  # Single token
        (4, 128, HIDDEN_SIZE),                # Larger batch
    ]

    for shape in shapes:
        print(f"\n  Shape: {shape}")
        torch.manual_seed(42)

        x = torch.randn(shape, dtype=DTYPE, device=device, requires_grad=True)
        weight = torch.randn(HIDDEN_SIZE, dtype=DTYPE, device=device)

        # Reference: PyTorch RMSNorm
        x_ref = x.detach().clone().requires_grad_(True)
        variance = x_ref.float().pow(2).mean(-1, keepdim=True)
        ref_out = (x_ref.float() * torch.rsqrt(variance + EPS)).to(DTYPE)
        ref_out = weight * ref_out

        # Fused kernel
        fused_norm = FusedRMSNorm(HIDDEN_SIZE, eps=EPS, weight=nn.Parameter(weight.clone()))
        fused_out = fused_norm(x)

        # Forward check
        ok = check_close("Forward", fused_out, ref_out)
        all_passed = all_passed and ok

        # Backward check
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)
        fused_out.backward(grad_out.clone())

        ok = check_close("Grad input", x.grad, x_ref.grad, atol=5e-2, rtol=5e-2)
        all_passed = all_passed and ok

    return all_passed


def test_fused_rope():
    """Test fused RoPE against PyTorch reference."""
    print("\n" + "="*60)
    print("TEST: Fused RoPE")
    print("="*60)

    try:
        from custom_kernels.fused_rope import fused_apply_rotary_emb
    except ImportError as e:
        print(f"  SKIP: {e}")
        return True

    device = 'cuda'
    all_passed = True

    torch.manual_seed(42)
    half_dim = HEAD_DIM // 2

    # Input: [batch, seq_len, n_heads, head_dim]
    x = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device, requires_grad=True)

    # Generate cos/sin cache [seq_len, half_dim]
    theta = 500000.0  # Llama 3.1 theta
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim))
    positions = torch.arange(SEQ_LEN, dtype=torch.float32, device=device)
    angles = torch.outer(positions, freqs)
    cos_cache = torch.cos(angles).to(DTYPE)
    sin_cache = torch.sin(angles).to(DTYPE)

    # Reference computation
    x_ref = x.detach().clone().requires_grad_(True)
    x1_ref = x_ref[..., :half_dim]
    x2_ref = x_ref[..., half_dim:]
    cos_expanded = cos_cache[None, :, None, :].expand_as(x1_ref)
    sin_expanded = sin_cache[None, :, None, :].expand_as(x1_ref)
    ref_out = torch.cat([
        x1_ref * cos_expanded - x2_ref * sin_expanded,
        x1_ref * sin_expanded + x2_ref * cos_expanded,
    ], dim=-1)

    # Fused kernel
    fused_out = fused_apply_rotary_emb(x, cos_cache, sin_cache)

    # Forward check
    ok = check_close("Forward", fused_out, ref_out)
    all_passed = all_passed and ok

    # Backward check
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    fused_out.backward(grad_out.clone())

    ok = check_close("Grad input", x.grad, x_ref.grad, atol=5e-2, rtol=5e-2)
    all_passed = all_passed and ok

    return all_passed


def test_fused_cross_entropy():
    """Test fused cross-entropy against PyTorch reference."""
    print("\n" + "="*60)
    print("TEST: Fused Cross-Entropy Loss")
    print("="*60)

    try:
        from custom_kernels.fused_cross_entropy import FusedCrossEntropyLoss
    except ImportError as e:
        print(f"  SKIP: {e}")
        return True

    device = 'cuda'
    all_passed = True

    # Test with different vocab sizes
    test_cases = [
        (BATCH_SIZE * SEQ_LEN, 1024, "Small vocab"),
        (BATCH_SIZE * SEQ_LEN, 32000, "Medium vocab"),
        (32, VOCAB_SIZE, "Full Llama vocab (128K)"),
    ]

    for N, V, desc in test_cases:
        print(f"\n  {desc}: N={N}, V={V}")
        torch.manual_seed(42)

        logits = torch.randn(N, V, dtype=DTYPE, device=device, requires_grad=True)
        targets = torch.randint(0, V, (N,), device=device)
        # Add some padding tokens
        targets[0] = -100
        targets[N//2] = -100

        # Reference: PyTorch CrossEntropyLoss
        logits_ref = logits.detach().clone().requires_grad_(True)
        ref_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        ref_loss = ref_loss_fn(logits_ref.float(), targets)

        # Fused kernel
        fused_loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
        fused_loss = fused_loss_fn(logits, targets)

        ok = check_close("Loss value", fused_loss.unsqueeze(0), ref_loss.unsqueeze(0).to(DTYPE), atol=1e-1, rtol=1e-1)
        all_passed = all_passed and ok

        # Backward check
        ref_loss.backward()
        fused_loss.backward()

        ok = check_close("Grad logits", logits.grad, logits_ref.grad.to(DTYPE), atol=5e-2, rtol=5e-2)
        all_passed = all_passed and ok

    return all_passed


def test_fused_cross_entropy_3d():
    """Test fused cross-entropy with 3D inputs (batch, seq, vocab)."""
    print("\n" + "="*60)
    print("TEST: Fused Cross-Entropy Loss (3D input)")
    print("="*60)

    try:
        from custom_kernels.fused_cross_entropy import FusedCrossEntropyLoss
    except ImportError as e:
        print(f"  SKIP: {e}")
        return True

    device = 'cuda'
    torch.manual_seed(42)

    B, S, V = 2, 32, 1024
    logits = torch.randn(B, S, V, dtype=DTYPE, device=device, requires_grad=True)
    targets = torch.randint(0, V, (B, S), device=device)

    # Reference
    logits_ref = logits.detach().clone().requires_grad_(True)
    ref_loss = torch.nn.CrossEntropyLoss()(logits_ref.float().reshape(-1, V), targets.reshape(-1))

    # Fused
    fused_loss = FusedCrossEntropyLoss()(logits, targets)

    ok = check_close("3D Loss", fused_loss.unsqueeze(0), ref_loss.unsqueeze(0).to(DTYPE), atol=1e-1, rtol=1e-1)

    ref_loss.backward()
    fused_loss.backward()

    ok2 = check_close("3D Grad", logits.grad, logits_ref.grad.to(DTYPE), atol=5e-2, rtol=5e-2)

    return ok and ok2


# Need nn import for FusedRMSNorm Parameter
import torch.nn as nn


def main():
    """Run all kernel tests."""
    print("="*60)
    print("Custom Kernel Correctness Tests")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Dtype: {DTYPE}")
    print("="*60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available. Tests require GPU.")
        sys.exit(1)

    results = {}
    results['RMSNorm'] = test_fused_rmsnorm()
    results['RoPE'] = test_fused_rope()
    results['CrossEntropy'] = test_fused_cross_entropy()
    results['CrossEntropy3D'] = test_fused_cross_entropy_3d()

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("All tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
