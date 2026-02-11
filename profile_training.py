#!/usr/bin/env python3
"""
Profiling script for Llama 3.1 training on AMD MI300X.
Uses PyTorch Profiler to capture GPU kernel-level traces and identify hotspots.

Usage:
    python profile_training.py --config configs/llama_8b.yaml --num-steps 10
    python profile_training.py --config configs/llama_8b.yaml --num-steps 5 --warmup-steps 2
"""

import argparse
import os
import time
import json
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import psutil
import yaml
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(tokenizer, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", max_length=8192):
    """Prepare a dataset for training."""
    print(f"Loading dataset {dataset_name}/{dataset_config}...")
    datasets = load_dataset(dataset_name, dataset_config)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )
    return tokenized_datasets


def create_batch(tokenized_dataset, batch_size, seq_length, device):
    """Create a single training batch from the dataset."""
    indices = list(range(min(batch_size, len(tokenized_dataset))))
    batch_input_ids = []
    batch_labels = []

    for idx in indices:
        item = tokenized_dataset[idx]
        input_ids = item['input_ids'][:seq_length]
        # Pad if necessary
        if len(input_ids) < seq_length:
            input_ids = input_ids + [0] * (seq_length - len(input_ids))
        batch_input_ids.append(input_ids)
        batch_labels.append(input_ids.copy())

    input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
    labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    attention_mask = (input_ids != 0).long()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def run_profiling(args):
    """Run profiling on the training loop."""
    cfg = load_config(args.config)
    model_cfg = cfg['model']
    training_cfg = cfg['training']
    dataset_cfg = cfg['dataset']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Memory: {props.total_memory / (1024**3):.1f} GB")

    # Load model
    model_name = model_cfg['name']
    print(f"\nLoading model config: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_config(config)
    model.to(device)

    dtype = torch.bfloat16 if training_cfg.get('bf16', True) else torch.float16
    model = model.to(dtype)
    print(f"Model loaded: {model.num_parameters():,} parameters ({dtype})")

    if training_cfg.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Prepare dataset
    tokenized_datasets = prepare_dataset(
        tokenizer,
        dataset_name=dataset_cfg['name'],
        dataset_config=dataset_cfg['config'],
        max_length=model_cfg['sequence_length'],
    )

    batch_size = training_cfg['per_device_train_batch_size']
    seq_length = model_cfg['sequence_length']
    batch = create_batch(tokenized_datasets['train'], batch_size, seq_length, device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg['learning_rate'],
        weight_decay=training_cfg.get('weight_decay', 0.1),
    )

    # Profiling output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = args.warmup_steps + args.num_steps
    print(f"\n{'='*80}")
    print(f"Profiling Configuration:")
    print(f"{'='*80}")
    print(f"  Warmup steps:   {args.warmup_steps}")
    print(f"  Profile steps:  {args.num_steps}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Seq length:     {seq_length}")
    print(f"  Tokens/step:    {batch_size * seq_length:,}")
    print(f"  Output dir:     {output_dir}")
    print(f"{'='*80}\n")

    # Run profiling
    model.train()

    print("Starting profiling run...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=0,
            warmup=args.warmup_steps,
            active=args.num_steps,
            repeat=1,
        ),
        # Note: not using on_trace_ready so we can export manually below
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for step in range(total_steps):
            step_start = time.time()

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            step_time = time.time() - step_start
            tokens_per_sec = (batch_size * seq_length) / step_time

            phase = "warmup" if step < args.warmup_steps else "profile"
            print(f"  Step {step+1}/{total_steps} [{phase}] - "
                  f"loss: {loss.item():.4f}, "
                  f"time: {step_time:.3f}s, "
                  f"tokens/s: {tokens_per_sec:.0f}")

            prof.step()

    # Export Chrome trace
    trace_path = output_dir / "chrome_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nChrome trace exported to: {trace_path}")

    # Print kernel summary
    print(f"\n{'='*80}")
    print("TOP 30 GPU KERNELS BY TOTAL CUDA TIME")
    print(f"{'='*80}")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
    ))

    # Print grouped summary
    print(f"\n{'='*80}")
    print("KERNEL TIME GROUPED BY INPUT SHAPE")
    print(f"{'='*80}")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total",
        row_limit=20,
    ))

    # Memory summary
    if torch.cuda.is_available():
        print(f"\n{'='*80}")
        print("GPU MEMORY SUMMARY")
        print(f"{'='*80}")
        print(f"  Peak allocated:  {torch.cuda.max_memory_allocated(0) / (1024**3):.2f} GB")
        print(f"  Peak reserved:   {torch.cuda.max_memory_reserved(0) / (1024**3):.2f} GB")

    # Export kernel data as JSON for analyze_profile.py
    kernel_data = []
    for evt in prof.key_averages():
        kernel_data.append({
            'name': evt.key,
            'count': evt.count,
            'cpu_time_total_us': evt.cpu_time_total,
            'cuda_time_total_us': evt.cuda_time_total,
            'cpu_time_avg_us': evt.cpu_time_total / max(evt.count, 1),
            'cuda_time_avg_us': evt.cuda_time_total / max(evt.count, 1),
            'self_cpu_time_us': evt.self_cpu_time_total,
            'self_cuda_time_us': evt.self_cuda_time_total,
            'flops': evt.flops if hasattr(evt, 'flops') else 0,
        })

    kernel_json_path = output_dir / "kernel_summary.json"
    with open(kernel_json_path, 'w') as f:
        json.dump(kernel_data, f, indent=2)
    print(f"\nKernel summary JSON exported to: {kernel_json_path}")

    print(f"\n{'='*80}")
    print("PROFILING COMPLETE")
    print(f"{'='*80}")
    print(f"\nTo view the trace visually:")
    print(f"  1. Open https://ui.perfetto.dev/ in your browser")
    print(f"  2. Load: {trace_path}")
    print(f"\nTo analyze results programmatically:")
    print(f"  python analyze_profile.py --input {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Profile Llama training on AMD MI300X")
    parser.add_argument("--config", type=str, default="configs/llama_8b.yaml",
                        help="Path to training config file")
    parser.add_argument("--num-steps", type=int, default=5,
                        help="Number of profiling steps (default: 5)")
    parser.add_argument("--warmup-steps", type=int, default=3,
                        help="Number of warmup steps before profiling (default: 3)")
    parser.add_argument("--output-dir", type=str, default="profiling_output",
                        help="Output directory for profiling results")
    args = parser.parse_args()

    run_profiling(args)


if __name__ == "__main__":
    main()
