#!/usr/bin/env python3
"""
Analyze profiling output and generate an actionable optimization report.
Reads kernel_summary.json from profile_training.py output and categorizes
operations by type to identify optimization candidates.

Usage:
    python analyze_profile.py --input profiling_output
    python analyze_profile.py --input profiling_output --top 20
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict


# Kernel classification rules
KERNEL_CATEGORIES = {
    'GEMM': [
        'gemm', 'Gemm', 'GEMM', 'rocblas', 'hipblas', 'blas',
        'cutlass', 'matmul', 'addmm', 'mm_', 'linear',
        'CudnnBatchNorm',  # often fused with GEMM
    ],
    'Attention': [
        'attention', 'sdpa', 'flash', 'softmax', 'Softmax',
        'scaled_dot_product', 'fmha', 'FlashAttn',
    ],
    'Normalization': [
        'rmsnorm', 'RmsNorm', 'rms_norm', 'layer_norm', 'LayerNorm',
        'batch_norm', 'group_norm', 'instance_norm',
    ],
    'Positional Encoding': [
        'rope', 'rotary', 'Rotary', 'RoPE', 'cos_sin',
        'apply_rotary', 'freqs_cis',
    ],
    'Loss': [
        'cross_entropy', 'CrossEntropy', 'nll_loss', 'NllLoss',
        'log_softmax', 'LogSoftmax',
    ],
    'Activation': [
        'silu', 'SiLU', 'gelu', 'GELU', 'relu', 'ReLU',
        'swish', 'sigmoid', 'tanh', 'act_',
    ],
    'Elementwise': [
        'mul_', 'add_', 'div_', 'sub_', 'pow_', 'sqrt_',
        'fill_', 'copy_', 'cat_', 'where_', 'clamp',
        'Pointwise', 'elementwise', 'aten::mul', 'aten::add',
    ],
    'Memory': [
        'memcpy', 'memset', 'Memcpy', 'Memset',
        'copy_', 'contiguous', 'to_', 'reshape', 'view',
        'transpose', 'permute', 'embedding',
    ],
    'Reduction': [
        'sum', 'mean', 'reduce', 'Reduce', 'var', 'std',
        'norm', 'max_', 'min_', 'argmax', 'argmin',
    ],
    'Communication': [
        'nccl', 'allreduce', 'allgather', 'broadcast',
        'scatter', 'reduce_scatter',
    ],
}


def classify_kernel(name):
    """Classify a kernel name into a category."""
    name_lower = name.lower()
    for category, keywords in KERNEL_CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in name_lower:
                return category
    return 'Other'


def load_kernel_summary(input_dir):
    """Load kernel summary JSON from profiling output."""
    json_path = Path(input_dir) / "kernel_summary.json"
    if not json_path.exists():
        # Try finding it in subdirectories
        for p in Path(input_dir).rglob("kernel_summary.json"):
            json_path = p
            break

    if not json_path.exists():
        raise FileNotFoundError(
            f"kernel_summary.json not found in {input_dir}. "
            f"Run profile_training.py first."
        )

    with open(json_path, 'r') as f:
        return json.load(f)


def analyze(kernel_data, top_n=30):
    """Analyze kernel data and print report."""

    # Calculate total CUDA time
    total_cuda_time = sum(k['self_cuda_time_us'] for k in kernel_data if k['self_cuda_time_us'] > 0)

    if total_cuda_time == 0:
        print("WARNING: No CUDA time recorded. Was the model running on GPU?")
        return

    print(f"\n{'='*100}")
    print(f"KERNEL PROFILING ANALYSIS REPORT")
    print(f"{'='*100}")
    print(f"\nTotal GPU time: {total_cuda_time/1e6:.3f} seconds")

    # === Per-kernel ranking ===
    print(f"\n{'─'*100}")
    print(f"TOP {top_n} KERNELS BY SELF GPU TIME")
    print(f"{'─'*100}")
    print(f"{'Rank':<6}{'Category':<20}{'GPU Time (ms)':<16}{'% Total':<10}{'Count':<10}{'Avg (us)':<12}{'Kernel Name'}")
    print(f"{'─'*100}")

    sorted_kernels = sorted(kernel_data, key=lambda x: x['self_cuda_time_us'], reverse=True)

    for i, k in enumerate(sorted_kernels[:top_n]):
        category = classify_kernel(k['name'])
        pct = (k['self_cuda_time_us'] / total_cuda_time) * 100
        avg_us = k['self_cuda_time_us'] / max(k['count'], 1)
        name = k['name'][:60]
        print(f"{i+1:<6}{category:<20}{k['self_cuda_time_us']/1000:<16.3f}{pct:<10.1f}{k['count']:<10}{avg_us:<12.1f}{name}")

    # === Category summary ===
    print(f"\n{'─'*100}")
    print(f"GPU TIME BY CATEGORY")
    print(f"{'─'*100}")

    category_times = defaultdict(lambda: {'time_us': 0, 'count': 0, 'kernels': []})

    for k in kernel_data:
        if k['self_cuda_time_us'] > 0:
            cat = classify_kernel(k['name'])
            category_times[cat]['time_us'] += k['self_cuda_time_us']
            category_times[cat]['count'] += k['count']
            category_times[cat]['kernels'].append(k['name'])

    sorted_categories = sorted(category_times.items(), key=lambda x: x[1]['time_us'], reverse=True)

    print(f"{'Category':<25}{'GPU Time (ms)':<18}{'% Total':<12}{'Kernel Count':<15}{'Optimization'}")
    print(f"{'─'*100}")

    for cat, data in sorted_categories:
        pct = (data['time_us'] / total_cuda_time) * 100
        unique_kernels = len(set(data['kernels']))

        # Provide optimization recommendation
        if cat == 'GEMM':
            opt = "TunableOp auto-tuning"
        elif cat == 'Normalization':
            opt = "★ Fused Triton RMSNorm"
        elif cat == 'Positional Encoding':
            opt = "★ Fused Triton RoPE"
        elif cat == 'Loss':
            opt = "★ Fused Triton CrossEntropy"
        elif cat == 'Attention':
            opt = "torch SDPA / FlashAttention"
        elif cat == 'Activation':
            opt = "torch.compile fusion"
        elif cat == 'Elementwise':
            opt = "torch.compile fusion"
        elif cat == 'Memory':
            opt = "Layout optimization"
        else:
            opt = "-"

        print(f"{cat:<25}{data['time_us']/1000:<18.3f}{pct:<12.1f}{unique_kernels:<15}{opt}")

    # === Optimization candidates ===
    print(f"\n{'─'*100}")
    print(f"OPTIMIZATION CANDIDATES (non-GEMM kernels > 2% of total GPU time)")
    print(f"{'─'*100}")

    candidates = []
    for k in sorted_kernels:
        cat = classify_kernel(k['name'])
        pct = (k['self_cuda_time_us'] / total_cuda_time) * 100
        if cat != 'GEMM' and pct > 2.0:
            candidates.append({
                'name': k['name'],
                'category': cat,
                'pct': pct,
                'time_ms': k['self_cuda_time_us'] / 1000,
                'count': k['count'],
            })

    if candidates:
        for c in candidates:
            print(f"  [{c['category']}] {c['name'][:70]}")
            print(f"    Time: {c['time_ms']:.3f}ms ({c['pct']:.1f}% of total), Calls: {c['count']}")
            print()
    else:
        print("  No significant non-GEMM hotspots found (all < 2% of GPU time).")
        print("  GEMM operations dominate — consider TunableOp auto-tuning.")

    # === Actionable summary ===
    print(f"\n{'='*100}")
    print(f"RECOMMENDED OPTIMIZATIONS")
    print(f"{'='*100}")
    print("""
  1. CUSTOM KERNELS (implemented in custom_kernels/):
     - Fused RMSNorm:       Eliminates 3+ kernel launches per norm → 1 kernel
     - Fused RoPE:          Eliminates intermediate sin/cos materialization
     - Fused CrossEntropy:  Avoids full [batch*seq, 128K] softmax tensor

  2. GEMM AUTO-TUNING:
     export PYTORCH_TUNABLEOP_ENABLED=1
     export PYTORCH_TUNABLEOP_TUNING=1
     (Run once to auto-tune, then reuse with TUNING=0)

  3. TORCH.COMPILE:
     model = torch.compile(model, backend="inductor")
     Fuses elementwise/activation ops automatically

  4. ENVIRONMENT VARIABLES:
     export HIP_FORCE_DEV_KERNARG=1    # Faster kernel arg passing
     export HSA_ENABLE_SDMA=0          # Can help with memory transfer overlap

  Run optimized training:
     python train_optimized.py --config configs/llama_8b.yaml --use-custom-kernels --use-torch-compile
""")


def main():
    parser = argparse.ArgumentParser(description="Analyze profiling results")
    parser.add_argument("--input", type=str, default="profiling_output",
                        help="Directory containing profiling output")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of top kernels to show")
    args = parser.parse_args()

    kernel_data = load_kernel_summary(args.input)
    analyze(kernel_data, top_n=args.top)


if __name__ == "__main__":
    main()
