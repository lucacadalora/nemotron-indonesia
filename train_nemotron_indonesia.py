#!/usr/bin/env python3
"""
Nemotron-Indonesia Training Script
Optimized for 8× H200 (141GB each)
Supports: continued pre-training, SFT, DPO

Usage:
    torchrun --nproc_per_node=8 train_nemotron_indonesia.py \
        --mode pretrain \
        --model_name nvidia/nemotron-3-8b-base-4k \
        --data_path ./data/indonesian_corpus \
        --output_dir ./models/nemotron-indonesia-8b \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --num_epochs 3

Requirements:
    pip install torch transformers datasets accelerate peft bitsandbytes flash-attn
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for Nemotron-Indonesia"""
    # Model
    model_name: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
    output_dir: str = "./models/nemotron-indonesia-8b"
    
    # Data
    data_path: str = "./data"
    max_length: int = 4096
    
    # Training
    mode: str = "pretrain"  # pretrain, sft, dpo
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.5e-5
    num_epochs: int = 1  # For pre-training, 1 epoch over 20B tokens
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimization
    fp16: bool = False
    bf16: bool = True  # H200 supports BF16 natively
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    
    # LoRA (for memory efficiency if needed)
    use_lora: bool = False  # Full fine-tuning with 8x H200 (141GB each)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    
    # DeepSpeed / FSDP
    deepspeed_config: Optional[str] = "./configs/deepspeed_zero3.json"
    fsdp: bool = False
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # Evaluation
    eval_split: float = 0.05
    
    # Distributed
    local_rank: int = -1


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def load_tokenizer(config: TrainingConfig):
    """Load and configure tokenizer"""
    logger.info(f"Loading tokenizer: {config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Nemotron uses special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add Indonesian special tokens if needed
    special_tokens = {
        'additional_special_tokens': [
            '<|user|>', '<|assistant|>', '<|system|>',
            '<|java|>', '<|sunda|>', '<|bali|>',  # Language tags
        ]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {num_added} special tokens")
    
    return tokenizer


def load_model(config: TrainingConfig, tokenizer):
    """Load model with optimizations for H200"""
    logger.info(f"Loading model: {config.model_name}")
    
    # Load model in BF16 (H200 optimized) with DeepSpeed support
    if config.deepspeed_config:
        logger.info(f"Using DeepSpeed config: {config.deepspeed_config}")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attention_2=config.flash_attention,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attention_2=config.flash_attention,
            device_map='auto' if not dist.is_initialized() else None,
        )
    
    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Apply LoRA if configured
    if config.use_lora:
        logger.info(f"Applying LoRA (r={config.lora_r}, alpha={config.lora_alpha})")
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


def prepare_pretraining_data(config: TrainingConfig, tokenizer):
    """Prepare Indonesian corpus for continued pre-training"""
    logger.info("Loading pre-training data...")
    
    datasets_to_load = [
        ('oscar-corpus/OSCAR-2301', 'id', 0.3),  # 30% of OSCAR
        ('cc100', 'id', 0.3),  # Common Crawl
        ('wikimedia/wikipedia', '20231101.id', 0.2),  # Wikipedia
    ]
    
    all_datasets = []
    
    for dataset_name, subset, ratio in datasets_to_load:
        try:
            logger.info(f"Loading {dataset_name} ({subset})...")
            ds = load_dataset(dataset_name, subset, split='train', streaming=True)
            
            # Take subset
            ds = ds.shuffle(seed=42)
            
            # Tokenize
            def tokenize_function(examples):
                texts = examples['text'] if 'text' in examples else examples['content']
                return tokenizer(
                    texts,
                    truncation=True,
                    max_length=config.max_length,
                    return_special_tokens_mask=True,
                )
            
            ds = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
            all_datasets.append(ds)
            
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
    
    # Mix datasets
    if len(all_datasets) > 1:
        mixed = concatenate_datasets(all_datasets)
    else:
        mixed = all_datasets[0]
    
    logger.info(f"Total training examples: {len(mixed)}")
    
    return mixed


def prepare_sft_data(config: TrainingConfig, tokenizer):
    """Prepare instruction data for supervised fine-tuning"""
    logger.info("Loading SFT data...")
    
    # Load IndoMMLU and other instruction datasets
    datasets_to_load = [
        ('indommlu', None),  # Academic benchmark
        ('indonlp/nusatranslation_senti', None),  # Sentiment
    ]
    
    all_datasets = []
    
    for dataset_name, subset in datasets_to_load:
        try:
            ds = load_dataset(dataset_name, subset, split='train')
            
            # Format as instruction-completion
            def format_instruction(examples):
                prompts = []
                for i in range(len(examples['question'])):
                    prompt = f"<|user|>\n{examples['question'][i]}\n<|assistant|>\n{examples['answer'][i]}"
                    prompts.append(prompt)
                
                return {'text': prompts}
            
            ds = ds.map(format_instruction, batched=True)
            all_datasets.append(ds)
            
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
    
    # Load custom instruction data if available
    custom_path = Path(config.data_path) / 'instructions.jsonl'
    if custom_path.exists():
        logger.info(f"Loading custom instructions from {custom_path}")
        custom_ds = load_dataset('json', data_files=str(custom_path), split='train')
        all_datasets.append(custom_ds)
    
    mixed = concatenate_datasets(all_datasets)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=config.max_length,
            padding='max_length',
        )
    
    mixed = mixed.map(tokenize_function, batched=True)
    
    return mixed


def create_trainer(config: TrainingConfig, model, tokenizer, train_dataset, eval_dataset):
    """Create HuggingFace Trainer with H200 optimizations"""
    
    # Training arguments optimized for 8× H200
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        
        # Precision
        bf16=config.bf16,  # H200 optimized
        fp16=config.fp16,
        
        # Logging
        logging_steps=config.logging_steps,
        logging_dir=f"{config.output_dir}/logs",
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        
        # Optimization
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
        # DeepSpeed
        deepspeed=config.deepspeed_config if config.deepspeed_config else None,
        
        # Reporting
        report_to=["tensorboard"],
        
        # Push to hub (optional)
        # push_to_hub=True,
        # hub_model_id="jatevo/nemotron-indonesia-8b",
    )
    
    # Data collator
    if config.mode == 'pretrain':
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
        )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    return trainer


def evaluate_model(model, tokenizer, eval_dataset):
    """Evaluate on IndoMMLU benchmark"""
    logger.info("Evaluating on IndoMMLU...")
    
    from datasets import load_metric
    
    # Load IndoMMLU test set
    indommlu = load_dataset('indommlu', split='test')
    
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for example in indommlu:
            prompt = example['question']
            choices = example['choices']
            answer = example['answer']
            
            # Format prompt
            full_prompt = f"<|user|>\n{prompt}\nPilihan: {', '.join(choices)}\n<|assistant|>\nJawaban:"
            
            inputs = tokenizer(full_prompt, return_tensors='pt').to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check if correct
            if str(answer) in prediction:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"IndoMMLU Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Nemotron-Indonesia')
    parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'sft', 'dpo'])
    parser.add_argument('--model_name', type=str, default='nvidia/nemotron-3-8b-base-4k')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./models/nemotron-indonesia-8b')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        mode=args.mode,
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        use_lora=args.use_lora,
        local_rank=args.local_rank,
    )
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    config.local_rank = local_rank
    
    logger.info(f"Starting Nemotron-Indonesia training")
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"GPUs: {world_size}× H200")
    logger.info(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps * world_size}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(config)
    
    # Load model
    model = load_model(config, tokenizer)
    
    # Prepare data
    if config.mode == 'pretrain':
        train_dataset = prepare_pretraining_data(config, tokenizer)
    else:
        train_dataset = prepare_sft_data(config, tokenizer)
    
    # Split eval
    eval_dataset = train_dataset.select(range(int(len(train_dataset) * config.eval_split)))
    train_dataset = train_dataset.select(range(int(len(train_dataset) * config.eval_split), len(train_dataset)))
    
    # Create trainer
    trainer = create_trainer(config, model, tokenizer, train_dataset, eval_dataset)
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"Model saved to {config.output_dir}")
    
    # Evaluate
    if rank == 0:
        accuracy = evaluate_model(model, tokenizer, eval_dataset)
        
        # Save results
        with open(f"{config.output_dir}/results.txt", 'w') as f:
            f.write(f"IndoMMLU Accuracy: {accuracy:.2%}\n")
            f.write(f"Model: {config.model_name}\n")
            f.write(f"Training mode: {config.mode}\n")
            f.write(f"Epochs: {config.num_epochs}\n")
            f.write(f"Learning rate: {config.learning_rate}\n")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
