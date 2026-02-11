#!/usr/bin/env python3
"""
Script to load Llama 3.1 architecture from Hugging Face and train from scratch.
This initializes the model with random weights rather than using pretrained weights.
Includes detailed metrics logging: tokens/sec, loss, GPU util, mem util, CPU util, mem.
Supports FSDP for distributed training of large models (70B+).
"""

import argparse
import time
import torch
import psutil
import os
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


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class MetricsCallback(TrainerCallback):
    """Custom callback to log detailed metrics for each training step."""
    
    def __init__(self, sequence_length, batch_size):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.step_start_time = None
        self.process = psutil.Process(os.getpid())
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Record start time of step."""
        self.step_start_time = time.time()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log detailed metrics after each logging step."""
        if logs is None or self.step_start_time is None:
            return
            
        # Calculate step time
        step_time = time.time() - self.step_start_time
        
        # Calculate tokens/sec
        tokens_per_batch = self.sequence_length * self.batch_size * args.gradient_accumulation_steps
        tokens_per_sec = tokens_per_batch / step_time if step_time > 0 else 0
        
        # Get CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem_info = self.process.memory_info()
        mem_gb = mem_info.rss / (1024 ** 3)  # Convert to GB
        system_mem = psutil.virtual_memory()
        system_mem_percent = system_mem.percent
        
        # Get GPU metrics (works with both NVIDIA and AMD/ROCm)
        gpu_util = 0
        gpu_mem_used = 0
        gpu_mem_total = 0
        gpu_mem_percent = 0
        
        if torch.cuda.is_available():
            try:
                # Get GPU memory stats (works with both NVIDIA and AMD ROCm)
                gpu_mem_used = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)  # GB
                
                # Try to get total GPU memory
                gpu_properties = torch.cuda.get_device_properties(0)
                gpu_mem_total = gpu_properties.total_memory / (1024 ** 3)  # GB
                gpu_mem_percent = (torch.cuda.memory_allocated(0) / gpu_properties.total_memory) * 100
                
                # Note: GPU utilization is harder to get portably
                # For MI300X, this would require rocm-smi parsing
                # For now, we'll show memory stats which are more important
                gpu_util = "N/A"
            except Exception as e:
                pass
        
        # Print detailed metrics
        loss = logs.get('loss', 0.0)
        step = state.global_step
        
        # Get GPU device name
        gpu_name = "N/A"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        
        print(f"\n{'='*80}")
        print(f"Step {step} Metrics (GPU: {gpu_name}):")
        print(f"{'='*80}")
        print(f"  Loss:           {loss:.4f}")
        print(f"  Tokens/sec:     {tokens_per_sec:.2f}")
        print(f"  Step time:      {step_time:.3f}s")
        print(f"  GPU Util:       {gpu_util}")
        print(f"  GPU Mem:        {gpu_mem_used:.2f}GB / {gpu_mem_total:.2f}GB ({gpu_mem_percent:.1f}%)")
        print(f"  CPU Util:       {cpu_percent:.1f}%")
        print(f"  Process Mem:    {mem_gb:.2f}GB")
        print(f"  System Mem:     {system_mem_percent:.1f}%")
        print(f"{'='*80}\n")
        
        # Reset start time for next step
        self.step_start_time = time.time()


def load_model_from_scratch(model_name="meta-llama/Meta-Llama-3.1-8B"):
    """
    Load Llama 3.1 8B architecture with random initialization.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        model: Randomly initialized Llama model
        tokenizer: Tokenizer for the model
        config: Model configuration
    """
    print(f"Loading configuration for {model_name}...")
    
    # Load the model configuration (architecture only)
    config = AutoConfig.from_pretrained(model_name)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model from scratch (random weights)
    # Note: Setting use_cache=False is recommended for training
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config)
    
    print(f"Model initialized from scratch with {model.num_parameters():,} parameters")
    
    return model, tokenizer, config


def prepare_dataset(tokenizer, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", max_length=8192):
    """
    Prepare a dataset for training.
    
    Args:
        tokenizer: Model tokenizer
        dataset_name: Dataset name from HuggingFace
        dataset_config: Dataset configuration
        max_length: Maximum sequence length
        
    Returns:
        tokenized_datasets: Tokenized and formatted dataset
    """
    print(f"Loading dataset {dataset_name}/{dataset_config}...")
    
    # Load dataset
    datasets = load_dataset(dataset_name, dataset_config)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
    
    # Tokenize datasets
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )
    
    return tokenized_datasets


def main():
    """Main training function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Llama model from scratch")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/llama_8b.yaml",
        help="Path to configuration file (default: configs/llama_8b.yaml)"
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    cfg = load_config(args.config)
    
    # Extract config sections
    model_cfg = cfg['model']
    training_cfg = cfg['training']
    dataset_cfg = cfg['dataset']
    fsdp_cfg = cfg.get('fsdp', {})
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Display configuration
    print(f"\n{'='*80}")
    print(f"Training Configuration:")
    print(f"{'='*80}")
    if 'exp_name' in cfg:
        print(f"  Experiment:     {cfg.get('exp_name', 'N/A')}")
        print(f"  Work Group:     {cfg.get('work_group', 'N/A')}")
        print(f"  Workspace:      {cfg.get('workspace', 'N/A')}")
    print(f"  Model:          {model_cfg['name']}")
    print(f"  Sequence Len:   {model_cfg['sequence_length']}")
    if 'micro_batch_size' in training_cfg:
        print(f"  Micro Batch:    {training_cfg.get('micro_batch_size', training_cfg['per_device_train_batch_size'])}")
        print(f"  Global Batch:   {training_cfg.get('global_batch_size', 'N/A')}")
    print(f"  Per-Dev Batch:  {training_cfg['per_device_train_batch_size']}")
    print(f"  Grad Accum:     {training_cfg['gradient_accumulation_steps']}")
    print(f"  Learning Rate:  {training_cfg['learning_rate']}")
    print(f"  Weight Decay:   {training_cfg.get('weight_decay', 'N/A')}")
    print(f"  LR Schedule:    {training_cfg.get('lr_decay_style', 'N/A')}")
    print(f"  FSDP Enabled:   {fsdp_cfg.get('enabled', False)}")
    if fsdp_cfg.get('enabled', False):
        print(f"  FSDP Strategy:  {fsdp_cfg.get('fsdp_strategy', 'N/A')}")
    print(f"{'='*80}\n")
    
    # Load model from scratch
    model, tokenizer, config = load_model_from_scratch(model_cfg['name'])
    
    # Don't move to device if using FSDP - FSDP will handle device placement
    if not fsdp_cfg.get('enabled', False):
        model.to(device)
    
    # Prepare dataset with specified sequence length
    tokenized_datasets = prepare_dataset(
        tokenizer,
        dataset_name=dataset_cfg['name'],
        dataset_config=dataset_cfg['config'],
        max_length=model_cfg['sequence_length']
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
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
    
    # Training arguments from config
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
        # FSDP settings
        fsdp=fsdp_cfg.get('fsdp_strategy', '') if fsdp_cfg.get('enabled', False) else '',
        fsdp_config=fsdp_config if fsdp_cfg.get('enabled', False) else None,
    )
    
    # Initialize custom metrics callback
    metrics_callback = MetricsCallback(
        sequence_length=model_cfg['sequence_length'],
        batch_size=training_cfg['per_device_train_batch_size']
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=[metrics_callback],
    )
    
    # Start training
    print(f"\nStarting training from scratch...")
    tokens_per_step = (
        training_cfg['per_device_train_batch_size'] * 
        model_cfg['sequence_length'] * 
        training_cfg['gradient_accumulation_steps']
    )
    print(f"Tokens per step (per device): {tokens_per_step:,}\n")
    
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    final_output_dir = os.path.join(training_cfg['output_dir'], "final_model")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"Training complete! Model saved to {final_output_dir}")


if __name__ == "__main__":
    main()
