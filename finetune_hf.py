#!/usr/bin/env python3
"""
HuggingFace fine-tuning script for training models on generated conversations.

This script:
1. Loads conversations from generated conversation files (e.g., data/feedback_1k_8b.json)
2. Prepares data in the format required by HuggingFace transformers
3. Masks all turns except the last assistant message (sets weights to zero for other turns)
4. Fine-tunes using HuggingFace Trainer
5. Saves the fine-tuned model and tokenizer
"""

import json
import os
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import random
import math

# HuggingFace imports
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class HFFineTuner:
    """Handles fine-tuning of HuggingFace models."""
    
    def __init__(self, model_name: str, use_lora: bool = True, lora_config: Optional[Dict] = None):
        """
        Initialize the fine-tuner with HuggingFace model.
        
        Args:
            model_name: HuggingFace model name/path
            use_lora: Whether to use LoRA fine-tuning
            lora_config: LoRA configuration parameters
        """
        self.model_name = model_name
        self.use_lora = use_lora and PEFT_AVAILABLE
        self.lora_config = lora_config or {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
        }
        
        self.tokenizer = None
        self.model = None
        self.stats = {
            'total_conversations': 0,
            'train_conversations': 0,
            'val_conversations': 0,
            'processed_conversations': 0,
            'skipped_conversations': 0,
            'total_tokens': 0,
            'training_tokens': 0,
            'max_context_turns': None
        }
    
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Apply LoRA if requested
        if self.use_lora:
            if not PEFT_AVAILABLE:
                print("Warning: PEFT not available, falling back to full fine-tuning")
                self.use_lora = False
            else:
                print("Applying LoRA configuration...")
                lora_config = LoraConfig(**self.lora_config)
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
    
    def prepare_training_data(self, conversations: List[Dict[str, Any]], 
                            train_split: float = 0.9, 
                            max_context_turns: Optional[int] = None,
                            max_length: int = 2048) -> tuple[Dataset, Dataset]:
        """
        Convert conversations to HuggingFace datasets.
        
        Args:
            conversations: List of conversation dictionaries
            train_split: Fraction of data to use for training
            max_context_turns: Maximum number of turns to include in context
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model_and_tokenizer() first.")
        
        # Shuffle and split data
        conversations_copy = conversations.copy()
        random.shuffle(conversations_copy)
        
        split_idx = int(len(conversations_copy) * train_split)
        train_conversations = conversations_copy[:split_idx]
        val_conversations = conversations_copy[split_idx:]
        
        print(f"Splitting data: {len(train_conversations)} training, {len(val_conversations)} validation")
        
        def process_conversations(conv_list: List[Dict[str, Any]], is_training: bool = True) -> List[Dict[str, Any]]:
            processed_data = []
            
            for conv in conv_list:
                conversation_context = conv.get('conversation_context', [])
                
                # Skip conversations that are too short
                if len(conversation_context) < 2:
                    self.stats['skipped_conversations'] += 1
                    continue
                
                # Truncate context if specified
                if max_context_turns is not None and len(conversation_context) > max_context_turns:
                    conversation_context = conversation_context[-max_context_turns:]
                
                # Find the last assistant message
                last_assistant_idx = None
                for i in range(len(conversation_context) - 1, -1, -1):
                    if conversation_context[i]['role'] == 'assistant':
                        last_assistant_idx = i
                        break
                
                if last_assistant_idx is None:
                    self.stats['skipped_conversations'] += 1
                    continue
                
                # Build the conversation text up to but not including the last assistant response
                context_text = ""
                target_text = ""
                
                for i, turn in enumerate(conversation_context):
                    role = turn['role']
                    content = turn['content']
                    
                    if i < last_assistant_idx:
                        # This is context (will be masked)
                        if role == 'user':
                            context_text += f"<|user|>\n{content}\n\n"
                        elif role == 'assistant':
                            context_text += f"<|assistant|>\n{content}\n\n"
                    elif i == last_assistant_idx:
                        # This is the target assistant response (will be trained on)
                        context_text += f"<|assistant|>\n"
                        target_text = f"{content}\n\n"
                
                # Combine context and target
                full_text = context_text + target_text
                
                # Tokenize the full conversation
                full_tokens = self.tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors=None
                )
                
                # Tokenize just the context to find where target starts
                context_tokens = self.tokenizer(
                    context_text,
                    truncation=False,
                    padding=False,
                    return_tensors=None
                )
                
                # Create labels - mask context, keep target
                labels = [-100] * len(full_tokens['input_ids'])
                context_length = len(context_tokens['input_ids'])
                
                # Only train on the target tokens (after context)
                if context_length < len(full_tokens['input_ids']):
                    for i in range(context_length, len(full_tokens['input_ids'])):
                        labels[i] = full_tokens['input_ids'][i]
                
                # Skip if no target tokens to train on
                training_token_count = len([l for l in labels if l != -100])
                if training_token_count == 0:
                    self.stats['skipped_conversations'] += 1
                    continue
                
                if is_training:
                    self.stats['training_tokens'] += training_token_count
                
                self.stats['total_tokens'] += len(full_tokens['input_ids'])
                
                processed_data.append({
                    'input_ids': full_tokens['input_ids'],
                    'attention_mask': full_tokens['attention_mask'],
                    'labels': labels
                })
                
                self.stats['processed_conversations'] += 1
                if is_training:
                    self.stats['train_conversations'] += 1
                else:
                    self.stats['val_conversations'] += 1
            
            return processed_data
        
        # Process training and validation data
        train_data = process_conversations(train_conversations, is_training=True)
        val_data = process_conversations(val_conversations, is_training=False)
        
        self.stats['total_conversations'] = len(conversations)
        self.stats['max_context_turns'] = max_context_turns
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Debug: print sample data structure
        if len(train_data) > 0:
            sample = train_data[0]
            print(f"Sample data structure:")
            print(f"  input_ids type: {type(sample['input_ids'])}, length: {len(sample['input_ids'])}")
            print(f"  attention_mask type: {type(sample['attention_mask'])}, length: {len(sample['attention_mask'])}")
            print(f"  labels type: {type(sample['labels'])}, length: {len(sample['labels'])}")
        
        return train_dataset, val_dataset
    
    def create_trainer(self, train_dataset: Dataset, val_dataset: Dataset, 
                      output_dir: str, training_args: Dict[str, Any]) -> Trainer:
        """
        Create a HuggingFace Trainer.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory for saving models
            training_args: Training arguments dictionary
            
        Returns:
            Configured Trainer
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        # Create training arguments
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            **training_args
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if len(val_dataset) > 0 else None,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if len(val_dataset) > 0 else None
        )
        
        return trainer
    
    def train(self, trainer: Trainer) -> Dict[str, Any]:
        """
        Run the training.
        
        Args:
            trainer: Configured Trainer
            
        Returns:
            Training results
        """
        print("Starting training...")
        start_time = datetime.now()
        
        # Train the model
        train_result = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"Training completed in {duration:.2f} seconds")
        
        # Save the model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(trainer.args.output_dir)
        
        return {
            'train_result': train_result,
            'training_duration': duration,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
    
    def save_results(self, results: Dict[str, Any], training_params: Dict[str, Any], 
                    output_file: str):
        """
        Save training results and parameters.
        
        Args:
            results: Training results
            training_params: Training parameters
            output_file: Output file path
        """
        final_results = {
            'model_name': self.model_name,
            'use_lora': self.use_lora,
            'lora_config': self.lora_config if self.use_lora else None,
            'training_results': results,
            'training_parameters': training_params,
            'training_stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Results saved to: {output_file}")
    
    def print_stats(self):
        """Print training data preparation statistics."""
        print(f"\n=== Training Data Statistics ===")
        print(f"Total conversations: {self.stats['total_conversations']}")
        print(f"Training conversations: {self.stats['train_conversations']}")
        print(f"Validation conversations: {self.stats['val_conversations']}")
        print(f"Processed conversations: {self.stats['processed_conversations']}")
        print(f"Skipped conversations: {self.stats['skipped_conversations']}")
        print(f"Total tokens: {self.stats['total_tokens']}")
        print(f"Training tokens: {self.stats['training_tokens']}")
        print(f"Masked tokens: {self.stats['total_tokens'] - self.stats['training_tokens']}")
        if self.stats['max_context_turns'] is not None:
            print(f"Context truncated to: {self.stats['max_context_turns']} turns")
        if self.stats['total_conversations'] > 0:
            train_ratio = self.stats['train_conversations'] / (self.stats['train_conversations'] + self.stats['val_conversations'])
            print(f"Train/Val split: {train_ratio:.1%}/{1-train_ratio:.1%}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Fine-tune HuggingFace models on generated conversations')
    parser.add_argument('--input', '-i', 
                       default='data/feedback_1k_8b.json',
                       help='Input JSON file with generated conversations')
    parser.add_argument('--model', '-m', 
                       default='microsoft/DialoGPT-medium',
                       help='HuggingFace model name or path')
    parser.add_argument('--output-dir', '-o',
                       default='./fine_tuned_model',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', '-e', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--batch-size', '-b', type=int, default=4,
                       help='Training batch size (default: 4)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                       help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--max-length', type=int, default=2048,
                       help='Maximum sequence length (default: 2048)')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                       help='Warmup ratio (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--max-conversations', type=int,
                       help='Maximum number of conversations to process (for testing)')
    parser.add_argument('--train-split', type=float, default=0.9,
                       help='Fraction of data to use for training (default: 0.9)')
    parser.add_argument('--max-context-turns', type=int,
                       help='Maximum number of conversation turns to include in context')
    parser.add_argument('--no-lora', action='store_true',
                       help='Use full fine-tuning instead of LoRA')
    parser.add_argument('--lora-r', type=int, default=16,
                       help='LoRA rank (default: 16)')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha (default: 32)')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                       help='LoRA dropout (default: 0.1)')
    parser.add_argument('--wandb-project',
                       help='Weights & Biases project name for logging')
    parser.add_argument('--save-steps', type=int, default=500,
                       help='Save checkpoint every N steps (default: 500)')
    parser.add_argument('--eval-steps', type=int, default=500,
                       help='Evaluate every N steps (default: 500)')
    parser.add_argument('--logging-steps', type=int, default=100,
                       help='Log every N steps (default: 100)')
    
    args = parser.parse_args()
    
    # Set up wandb if requested
    if args.wandb_project and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project)
    
    # Load conversations
    print(f"Loading conversations from {args.input}...")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Limit conversations for testing if specified
        if args.max_conversations:
            conversations = conversations[:args.max_conversations]
            print(f"Limited to {len(conversations)} conversations for testing")
        
        print(f"Loaded {len(conversations)} conversations")
    except FileNotFoundError:
        print(f"Error: Could not find input file {args.input}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}")
        return 1
    
    # Initialize fine-tuner
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    } if PEFT_AVAILABLE else None
    
    fine_tuner = HFFineTuner(
        model_name=args.model,
        use_lora=not args.no_lora,
        lora_config=lora_config
    )
    
    # Load model and tokenizer
    try:
        fine_tuner.load_model_and_tokenizer()
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Prepare training data
    print(f"\nPreparing training data with {args.train_split:.1%} train / {1-args.train_split:.1%} validation split...")
    if args.max_context_turns:
        print(f"Context will be truncated to last {args.max_context_turns} turns")
    
    try:
        train_dataset, val_dataset = fine_tuner.prepare_training_data(
            conversations,
            train_split=args.train_split,
            max_context_turns=args.max_context_turns,
            max_length=args.max_length
        )
        fine_tuner.print_stats()
    except Exception as e:
        print(f"Error preparing training data: {e}")
        return 1
    
    if len(train_dataset) == 0:
        print("Error: No valid training data found")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training arguments
    training_args = {
        'num_train_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'warmup_ratio': args.warmup_ratio,
        'weight_decay': args.weight_decay,
        'logging_steps': args.logging_steps,
        'save_steps': args.save_steps,
        'eval_steps': args.eval_steps,
        'eval_strategy': 'steps' if len(val_dataset) > 0 else 'no',
        'save_strategy': 'steps',
        'load_best_model_at_end': True if len(val_dataset) > 0 else False,
        'metric_for_best_model': 'eval_loss' if len(val_dataset) > 0 else None,
        'greater_is_better': False,
        'report_to': 'wandb' if args.wandb_project and WANDB_AVAILABLE else None,
        'run_name': f"finetune_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'dataloader_pin_memory': False,
        'fp16': torch.cuda.is_available(),
    }
    
    # Create trainer
    trainer = fine_tuner.create_trainer(train_dataset, val_dataset, args.output_dir, training_args)
    
    # Train the model
    try:
        results = fine_tuner.train(trainer)
    except Exception as e:
        print(f"Error during training: {e}")
        return 1
    
    # Save results
    training_params = {
        'input_file': args.input,
        'model': args.model,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_length': args.max_length,
        'warmup_ratio': args.warmup_ratio,
        'weight_decay': args.weight_decay,
        'max_conversations': args.max_conversations,
        'train_split': args.train_split,
        'max_context_turns': args.max_context_turns,
        'use_lora': not args.no_lora,
        'lora_config': lora_config if not args.no_lora else None,
        'wandb_project': args.wandb_project,
    }
    
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    results_file = f"models/{base_name}_hf_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('models', exist_ok=True)
    
    fine_tuner.save_results(results, training_params, results_file)
    
    print(f"\n🎉 Fine-tuning complete!")
    print(f"Model saved to: {args.output_dir}")
    print(f"Results saved to: {results_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())
