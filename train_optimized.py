#!/usr/bin/env python3
"""
Optimized Llama 3.1 training with custom Triton kernels for AMD MI300X.

Builds on train_llama_from_scratch.py with:
- Fused RMSNorm Triton kernel
- Fused RoPE Triton kernel
- Fused Cross-Entropy loss
- TunableOp GEMM auto-tuning
- torch.compile integration

Usage:
    python train_optimized.py --config configs/llama_8b.yaml --use-custom-kernels
    python train_optimized.py --config configs/llama_8b.yaml --use-custom-kernels --use-torch-compile
"""

import argparse
import time
import os
import sys
import torch
import psutil
import yaml
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_dataset

# Import from existing training script
from train_llama_from_scratch import (
    load_config,
    MetricsCallback,
    prepare_dataset,
)


def setup_mi300x_env():
    """Set environment variables optimized for AMD MI300X."""
    env_vars = {
        'HIP_FORCE_DEV_KERNARG': '1',    # Faster kernel argument passing
        'HSA_ENABLE_SDMA': '0',           # Can help with memory transfers
    }

    for key, val in env_vars.items():
        if key not in os.environ:
            os.environ[key] = val
            print(f"  Set {key}={val}")


def setup_tunable_op(tune=False):
    """Setup PyTorch TunableOp for GEMM auto-tuning on MI300X."""
    os.environ['PYTORCH_TUNABLEOP_ENABLED'] = '1'

    if tune:
        os.environ['PYTORCH_TUNABLEOP_TUNING'] = '1'
        print("  TunableOp: TUNING mode (will auto-tune GEMMs)")
    else:
        os.environ['PYTORCH_TUNABLEOP_TUNING'] = '0'
        print("  TunableOp: ENABLED (using cached tuning results)")

    # Set tuning file path
    tuning_file = './output/tunableop_results.csv'
    os.environ['PYTORCH_TUNABLEOP_FILENAME'] = tuning_file
    print(f"  TunableOp file: {tuning_file}")


class OptimizedTrainer(Trainer):
    """Trainer subclass that uses fused cross-entropy loss."""

    def __init__(self, *args, fused_cross_entropy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fused_cross_entropy = fused_cross_entropy

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to use fused cross-entropy when available."""
        if self.fused_cross_entropy is not None:
            # Forward pass
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
            )

            logits = outputs.logits
            labels = inputs['labels']

            # Shift for causal LM (predict next token)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Use fused cross-entropy
            loss = self.fused_cross_entropy(shift_logits, shift_labels)

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


def main():
    """Main optimized training function."""
    parser = argparse.ArgumentParser(description="Optimized Llama training for MI300X")
    parser.add_argument("--config", type=str, default="configs/llama_8b.yaml",
                        help="Path to configuration file")
    parser.add_argument("--use-custom-kernels", action="store_true",
                        help="Enable custom Triton kernels (RMSNorm, RoPE, CrossEntropy)")
    parser.add_argument("--use-torch-compile", action="store_true",
                        help="Enable torch.compile for additional kernel fusion")
    parser.add_argument("--tune-gemm", action="store_true",
                        help="Enable TunableOp GEMM auto-tuning (slower first run)")
    parser.add_argument("--no-tunable-op", action="store_true",
                        help="Disable TunableOp entirely")
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    cfg = load_config(args.config)

    model_cfg = cfg['model']
    training_cfg = cfg['training']
    dataset_cfg = cfg['dataset']
    fsdp_cfg = cfg.get('fsdp', {})

    # Setup MI300X environment
    print("\n" + "="*80)
    print("MI300X OPTIMIZATION SETUP")
    print("="*80)

    setup_mi300x_env()

    if not args.no_tunable_op:
        setup_tunable_op(tune=args.tune_gemm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model_name = model_cfg['name']
    print(f"\nLoading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_config(config)

    if not fsdp_cfg.get('enabled', False):
        model.to(device)

    print(f"Model loaded: {model.num_parameters():,} parameters")

    # Apply custom kernels
    fused_ce = None
    if args.use_custom_kernels:
        print(f"\n{'='*80}")
        print("APPLYING CUSTOM KERNELS")
        print(f"{'='*80}")

        try:
            from custom_kernels import patch_model, has_triton
            if has_triton():
                patch_model(model, use_fused_cross_entropy=True)
                # Get the fused cross-entropy loss for the trainer
                from custom_kernels.fused_cross_entropy import FusedCrossEntropyLoss
                fused_ce = FusedCrossEntropyLoss()
                print("Custom kernels successfully applied!")
            else:
                print("WARNING: Triton not available. Falling back to standard kernels.")
        except Exception as e:
            print(f"WARNING: Failed to apply custom kernels: {e}")
            print("Falling back to standard kernels.")
            import traceback
            traceback.print_exc()

    # Apply torch.compile
    if args.use_torch_compile:
        print(f"\n{'='*80}")
        print("APPLYING TORCH.COMPILE")
        print(f"{'='*80}")
        try:
            model = torch.compile(model, backend="inductor", mode="reduce-overhead")
            print("torch.compile applied with backend='inductor', mode='reduce-overhead'")
        except Exception as e:
            print(f"WARNING: torch.compile failed: {e}")
            print("Continuing without torch.compile.")

    # Prepare dataset
    tokenized_datasets = prepare_dataset(
        tokenizer,
        dataset_name=dataset_cfg['name'],
        dataset_config=dataset_cfg['config'],
        max_length=model_cfg['sequence_length'],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Build FSDP config if enabled
    fsdp_config = None
    if fsdp_cfg.get('enabled', False):
        fsdp_config = {
            'fsdp_transformer_layer_cls_to_wrap': fsdp_cfg.get('fsdp_transformer_layer_cls_to_wrap', []),
            'fsdp_backward_prefetch': fsdp_cfg.get('fsdp_backward_prefetch', 'backward_pre'),
            'fsdp_auto_wrap_policy': fsdp_cfg.get('fsdp_auto_wrap_policy', 'transformer_based_wrap'),
            'fsdp_cpu_offload': fsdp_cfg.get('fsdp_cpu_offload', False),
            'fsdp_sync_module_states': fsdp_cfg.get('fsdp_sync_module_states', True),
            'fsdp_use_orig_params': fsdp_cfg.get('fsdp_use_orig_params', False),
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=training_cfg['output_dir'],
        num_train_epochs=training_cfg['num_train_epochs'],
        per_device_train_batch_size=training_cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=training_cfg['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_cfg['gradient_accumulation_steps'],
        learning_rate=training_cfg['learning_rate'],
        weight_decay=training_cfg['weight_decay'],
        warmup_steps=training_cfg['warmup_steps'],
        max_grad_norm=training_cfg.get('max_grad_norm', 1.0),
        logging_dir=training_cfg['logging_dir'],
        logging_steps=training_cfg['logging_steps'],
        save_steps=training_cfg['save_steps'],
        save_total_limit=training_cfg['save_total_limit'],
        eval_strategy=training_cfg.get('evaluation_strategy', 'steps'),
        eval_steps=training_cfg['eval_steps'],
        fp16=training_cfg.get('fp16', False),
        bf16=training_cfg.get('bf16', True),
        gradient_checkpointing=training_cfg['gradient_checkpointing'],
        report_to=training_cfg['report_to'],
        fsdp=fsdp_cfg.get('fsdp_strategy', '') if fsdp_cfg.get('enabled', False) else '',
        fsdp_config=fsdp_config if fsdp_cfg.get('enabled', False) else None,
    )

    # Metrics callback
    metrics_callback = MetricsCallback(
        sequence_length=model_cfg['sequence_length'],
        batch_size=training_cfg['per_device_train_batch_size'],
    )

    # Initialize optimized trainer
    trainer = OptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=[metrics_callback],
        fused_cross_entropy=fused_ce,
    )

    # Display optimization summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"  Custom Kernels:   {'ENABLED' if args.use_custom_kernels else 'DISABLED'}")
    print(f"  torch.compile:    {'ENABLED' if args.use_torch_compile else 'DISABLED'}")
    print(f"  TunableOp:        {'DISABLED' if args.no_tunable_op else ('TUNING' if args.tune_gemm else 'ENABLED')}")
    print(f"  bf16:             {training_cfg.get('bf16', True)}")
    print(f"  Grad checkpoint:  {training_cfg['gradient_checkpointing']}")
    print(f"{'='*80}\n")

    # Start training
    print(f"Starting optimized training...")
    tokens_per_step = (
        training_cfg['per_device_train_batch_size'] *
        model_cfg['sequence_length'] *
        training_cfg['gradient_accumulation_steps']
    )
    print(f"Tokens per step (per device): {tokens_per_step:,}\n")

    trainer.train()

    # Save model
    print("Saving model...")
    final_output_dir = os.path.join(training_cfg['output_dir'], "final_model_optimized")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print(f"Training complete! Model saved to {final_output_dir}")


if __name__ == "__main__":
    main()
