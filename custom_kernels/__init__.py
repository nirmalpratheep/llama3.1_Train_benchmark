"""
Custom Triton kernels for optimizing Llama 3.1 training on AMD MI300X.

Provides fused implementations of:
- RMSNorm (fused variance + normalize + scale)
- RoPE (fused rotary positional embeddings)
- CrossEntropy (fused log-softmax + NLL loss)

Usage:
    from custom_kernels import patch_model, has_triton

    if has_triton():
        patch_model(model)  # Monkey-patches HuggingFace Llama model
"""

import torch
import importlib

_TRITON_AVAILABLE = None


def has_triton():
    """Check if Triton is available."""
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            import triton
            _TRITON_AVAILABLE = True
        except ImportError:
            _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


def patch_model(model, use_fused_rmsnorm=True, use_fused_rope=True, use_fused_cross_entropy=True):
    """
    Patch a HuggingFace Llama model with optimized custom kernels.

    Args:
        model: HuggingFace LlamaForCausalLM model
        use_fused_rmsnorm: Replace LlamaRMSNorm with fused Triton version
        use_fused_rope: Replace rotary embedding with fused Triton version
        use_fused_cross_entropy: Replace cross-entropy loss with fused version

    Returns:
        model: Patched model (modified in-place)
    """
    if not has_triton():
        print("WARNING: Triton not available. Custom kernels disabled.")
        return model

    patches_applied = []

    if use_fused_rmsnorm:
        from .fused_rmsnorm import patch_rmsnorm
        patch_rmsnorm(model)
        patches_applied.append("FusedRMSNorm")

    if use_fused_rope:
        from .fused_rope import patch_rope
        patch_rope(model)
        patches_applied.append("FusedRoPE")

    if use_fused_cross_entropy:
        from .fused_cross_entropy import FusedCrossEntropyLoss
        # Store for use during training
        model._fused_cross_entropy = FusedCrossEntropyLoss()
        patches_applied.append("FusedCrossEntropy")

    print(f"Custom kernels applied: {', '.join(patches_applied)}")
    return model


def get_kernel_info():
    """Return information about available custom kernels."""
    info = {
        'triton_available': has_triton(),
        'kernels': {
            'fused_rmsnorm': 'Fuses variance computation, normalization, and weight scaling into one kernel',
            'fused_rope': 'Fuses sin/cos computation with rotation into one kernel',
            'fused_cross_entropy': 'Computes log-softmax + NLL loss in a single pass',
        },
    }
    if has_triton():
        import triton
        info['triton_version'] = triton.__version__
    return info
