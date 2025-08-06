#!/usr/bin/env python3
"""
Fine-tuning script for training models on generated conversations using the Together API.

This script:
1. Loads conversations from generated conversation files (e.g., data/feedback_1k_8b.json)
2. Prepares data in the format required by Together API
3. Masks all turns except the last assistant message (sets weights to zero for other turns)
4. Uploads data and launches fine-tuning job
5. Monitors training progress and saves results
"""

import json
import os
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from together import Together
from together.utils import check_file
import tempfile

class FineTuner:
    """Handles fine-tuning of models using the Together API."""
    
    def __init__(self, api_key: Optional[str] = None, wandb_api_key: Optional[str] = None):
        """
        Initialize the fine-tuner with Together API.
        
        Args:
            api_key: Together API key (if None, will look for TOGETHER_API_KEY env var)
            wandb_api_key: Weights & Biases API key (if None, will look for WANDB_API_KEY env var)
        """
        if api_key is None:
            api_key = os.getenv('TOGETHER_API_KEY')
        
        if not api_key:
            raise ValueError("Together API key is required. Set TOGETHER_API_KEY environment variable or pass api_key parameter.")
        
        self.client = Together(api_key=api_key)
        self.wandb_api_key = wandb_api_key or os.getenv('WANDB_API_KEY')
        self.stats = {
            'total_conversations': 0,
            'train_conversations': 0,
            'val_conversations': 0,
            'processed_conversations': 0,
            'skipped_conversations': 0,
            'total_messages': 0,
            'training_messages': 0,
            'max_context_turns': None
        }
    
    def prepare_training_data(self, conversations: List[Dict[str, Any]], train_split: float = 0.9, max_context_turns: Optional[int] = None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Convert conversations to the format required by Together API and split into train/val.
        Only trains on the last assistant message in each conversation.
        
        Args:
            conversations: List of conversation dictionaries
            train_split: Fraction of data to use for training (default: 0.9)
            max_context_turns: Maximum number of turns to include in context (None for no limit)
            
        Returns:
            Tuple of (training_data, validation_data) in Together API format
        """
        import random
        
        # Create a copy and shuffle for random split
        conversations_copy = conversations.copy()
        random.shuffle(conversations_copy)
        
        # Calculate split point
        split_idx = int(len(conversations_copy) * train_split)
        train_conversations = conversations_copy[:split_idx]
        val_conversations = conversations_copy[split_idx:]
        
        print(f"Splitting data: {len(train_conversations)} training, {len(val_conversations)} validation")
        
        def process_conversations(conv_list: List[Dict[str, Any]], is_training: bool = True) -> List[Dict[str, Any]]:
            processed_data = []
            
            for conv in conv_list:
                conversation_context = conv.get('conversation_context', [])
                
                # Skip conversations that are too short or don't have assistant messages
                if len(conversation_context) < 2:
                    self.stats['skipped_conversations'] += 1
                    continue
                
                # Truncate context if max_context_turns is specified
                if max_context_turns is not None and len(conversation_context) > max_context_turns:
                    # Keep the last max_context_turns turns
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
                
                # Create messages list with weights
                messages = []
                for i, turn in enumerate(conversation_context):
                    message = {
                        "role": turn['role'],
                        "content": turn['content']
                    }
                    
                    # Only train on the last assistant message, mask all others
                    if i == last_assistant_idx and turn['role'] == 'assistant':
                        message["weight"] = 1  # Train on this message
                        if is_training:
                            self.stats['training_messages'] += 1
                    else:
                        message["weight"] = 0  # Mask this message (don't train on it)
                    
                    messages.append(message)
                    self.stats['total_messages'] += 1
                
                training_example = {"messages": messages}
                processed_data.append(training_example)
                self.stats['processed_conversations'] += 1
                
                if is_training:
                    self.stats['train_conversations'] += 1
                else:
                    self.stats['val_conversations'] += 1
            
            return processed_data
        
        # Process training and validation data
        training_data = process_conversations(train_conversations, is_training=True)
        validation_data = process_conversations(val_conversations, is_training=False)
        
        self.stats['total_conversations'] = len(conversations)
        self.stats['max_context_turns'] = max_context_turns
        return training_data, validation_data
    
    def save_training_data(self, training_data: List[Dict[str, Any]], validation_data: List[Dict[str, Any]], base_path: str) -> tuple[str, str]:
        """
        Save training and validation data to JSONL format required by Together API.
        
        Args:
            training_data: List of training examples
            validation_data: List of validation examples
            base_path: Base path for saving files (will add _train.jsonl and _val.jsonl)
            
        Returns:
            Tuple of (training_file_path, validation_file_path)
        """
        # Create file paths
        base_name = os.path.splitext(base_path)[0]
        train_path = f"{base_name}_train.jsonl"
        val_path = f"{base_name}_val.jsonl"
        
        # Save training data
        with open(train_path, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Save validation data
        with open(val_path, 'w', encoding='utf-8') as f:
            for example in validation_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"Training data saved to: {train_path} ({len(training_data)} examples)")
        print(f"Validation data saved to: {val_path} ({len(validation_data)} examples)")
        
        return train_path, val_path
    
    def upload_training_file(self, file_path: str) -> str:
        """
        Upload training file to Together API and return file ID.
        
        Args:
            file_path: Path to the training JSONL file
            
        Returns:
            File ID from Together API
        """
        print(f"Checking file format: {file_path}")
        sft_report = check_file(file_path)
        print(f"File check report: {json.dumps(sft_report, indent=2)}")
        
        if not sft_report.get("is_check_passed", False):
            raise ValueError(f"File format check failed: {sft_report.get('message', 'Unknown error')}")
        
        print(f"Uploading training file to Together API...")
        train_file_resp = self.client.files.upload(file_path, check=True)
        print(f"File uploaded with ID: {train_file_resp.id}")
        
        return train_file_resp.id
    
    def start_fine_tuning(self, 
                         training_file_id: str,
                         validation_file_id: Optional[str] = None,
                         base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
                         n_epochs: int = 3,
                         suffix: Optional[str] = None,
                         learning_rate: float = 1e-5,
                         lora: bool = True,
                         batch_size: Union[int, str] = "max",
                         #wandb_project: Optional[str] = None
                         ) -> str:
        """
        Start a fine-tuning job.
        
        Args:
            training_file_id: ID of the uploaded training file
            validation_file_id: ID of the uploaded validation file (optional)
            base_model: Base model to fine-tune
            n_epochs: Number of training epochs
            suffix: Suffix for the fine-tuned model name
            learning_rate: Learning rate for training
            lora: Whether to use LoRA fine-tuning
            batch_size: Batch size for training
            
        Returns:
            Fine-tuning job ID
        """
        if suffix is None:
            suffix = f"counterfactual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Starting fine-tuning job...")
        print(f"  Base model: {base_model}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  LoRA: {lora}")
        print(f"  Suffix: {suffix}")
        if validation_file_id:
            print(f"  Validation file: {validation_file_id}")
        if self.wandb_api_key:
            print(f"  W&B enabled")
        
        # Handle batch size - convert to int if it's a numeric string
        if isinstance(batch_size, str) and batch_size != "max":
            try:
                batch_size = int(batch_size)
            except ValueError:
                batch_size = "max"
        
        # Build fine-tuning parameters
        ft_params = {
            "training_file": training_file_id,
            "model": base_model,
            "train_on_inputs": "auto",
            "n_epochs": n_epochs,
            "n_checkpoints": 1,
            "lora": lora,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "suffix": suffix,
            "warmup_ratio": 0,
        }
        
        # Add validation file if provided
        if validation_file_id:
            ft_params["validation_file"] = validation_file_id
        
        # Add wandb config if available
        if self.wandb_api_key:
            ft_params["wandb_api_key"] = self.wandb_api_key
        
        ft_resp = self.client.fine_tuning.create(**ft_params)
        
        print(f"Fine-tuning job started with ID: {ft_resp.id}")
        return ft_resp.id or ""
    
    def monitor_training(self, job_id: str, check_interval: int = 30) -> Any:
        """
        Monitor training progress and return final job info.
        
        Args:
            job_id: Fine-tuning job ID
            check_interval: How often to check status (seconds)
            
        Returns:
            Final job information
        """
        print(f"Monitoring training job: {job_id}")
        print(f"Checking status every {check_interval} seconds...")
        
        while True:
            try:
                job_info = self.client.fine_tuning.retrieve(job_id)
                status = job_info.status
                
                print(f"Status: {status}")
                
                if status == 'completed':
                    print(f"✅ Training completed successfully!")
                    print(f"Fine-tuned model: {job_info.output_name}")
                    return job_info
                elif status in ['failed', 'cancelled']:
                    print(f"❌ Training {status}")
                    return job_info
                elif status in ['pending', 'queued', 'running', 'uploading']:
                    print(f"⏳ Training in progress... ({status})")
                    time.sleep(check_interval)
                else:
                    print(f"⚠️  Unknown status: {status}")
                    time.sleep(check_interval)
                    
            except Exception as e:
                print(f"Error checking status: {e}")
                time.sleep(check_interval)
    
    def save_results(self, job_info: Any, results_path: str, training_params: Optional[Dict[str, Any]] = None, training_stats: Optional[Dict[str, Any]] = None):
        """
        Save training results to JSON file.
        
        Args:
            job_info: Job information from Together API
            results_path: Path to save results
            training_params: Training parameters used to start the job
            training_stats: Additional training statistics
        """
        results = {
            'job_id': job_info.id,
            'status': job_info.status,
            'output_name': getattr(job_info, 'output_name', None),
            'base_model': getattr(job_info, 'model', None),
            'created_at': getattr(job_info, 'created_at', None),
            'finished_at': getattr(job_info, 'finished_at', None),
            'hyperparameters': getattr(job_info, 'hyperparameters', {}),
            'training_file_id': getattr(job_info, 'training_file', None),
            'timestamp': datetime.now().isoformat(),
            'training_stats': training_stats or self.stats,
            'training_parameters': training_params or {}
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Results saved to: {results_path}")
        return results
    
    def print_stats(self):
        """Print training data preparation statistics."""
        print(f"\n=== Training Data Statistics ===")
        print(f"Total conversations: {self.stats['total_conversations']}")
        print(f"Training conversations: {self.stats['train_conversations']}")
        print(f"Validation conversations: {self.stats['val_conversations']}")
        print(f"Processed conversations: {self.stats['processed_conversations']}")
        print(f"Skipped conversations: {self.stats['skipped_conversations']}")
        print(f"Total messages: {self.stats['total_messages']}")
        print(f"Training messages (weight=1): {self.stats['training_messages']}")
        print(f"Masked messages (weight=0): {self.stats['total_messages'] - self.stats['training_messages']}")
        if self.stats['max_context_turns'] is not None:
            print(f"Context truncated to: {self.stats['max_context_turns']} turns")
        if self.stats['total_conversations'] > 0:
            train_ratio = self.stats['train_conversations'] / (self.stats['train_conversations'] + self.stats['val_conversations'])
            print(f"Train/Val split: {train_ratio:.1%}/{1-train_ratio:.1%}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Fine-tune models on generated conversations')
    parser.add_argument('--input', '-i', 
                       default='data/feedback_1k_8b.json',
                       help='Input JSON file with generated conversations')
    parser.add_argument('--model', '-m', 
                       default='meta-llama/Meta-Llama-3.1-8B-Instruct-Reference',
                       help='Base model to fine-tune')
    parser.add_argument('--epochs', '-e', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5)')
    parser.add_argument('--suffix', '-s', 
                       help='Suffix for the fine-tuned model name (default: auto-generated)')
    parser.add_argument('--lora', action='store_true', default=True,
                       help='Use LoRA fine-tuning (default: True)')
    parser.add_argument('--full', action='store_true',
                       help='Use full fine-tuning instead of LoRA')
    parser.add_argument('--batch-size', '-b', default='max',
                       help='Batch size for training (default: max)')
    parser.add_argument('--no-monitor', action='store_true',
                       help='Start training but don\'t wait for completion')
    parser.add_argument('--check-interval', type=int, default=30,
                       help='Status check interval in seconds (default: 30)')
    parser.add_argument('--max-conversations', type=int,
                       help='Maximum number of conversations to process (for testing)')
    parser.add_argument('--train-split', type=float, default=0.9,
                       help='Fraction of data to use for training (default: 0.9)')
    parser.add_argument('--max-context-turns', type=int,
                       help='Maximum number of conversation turns to include in context (default: no limit)')
    #parser.add_argument('--wandb-project', 
    #                   help='Weights & Biases project name for logging')
    parser.add_argument('--wandb-api-key',
                       help='Weights & Biases API key (or set WANDB_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Handle LoRA vs full fine-tuning
    use_lora = not args.full
    
    # Check if Together API key is available
    if not os.getenv('TOGETHER_API_KEY'):
        print("Error: TOGETHER_API_KEY environment variable not set.")
        print("Please set it with: export TOGETHER_API_KEY='your-api-key-here'")
        return 1
    
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
    try:
        fine_tuner = FineTuner(wandb_api_key=args.wandb_api_key)
    except ValueError as e:
        print(f"Error initializing fine-tuner: {e}")
        return 1
    
    # Prepare training data
    print(f"\nPreparing training data with {args.train_split:.1%} train / {1-args.train_split:.1%} validation split...")
    if args.max_context_turns:
        print(f"Context will be truncated to last {args.max_context_turns} turns")
    training_data, validation_data = fine_tuner.prepare_training_data(
        conversations, 
        train_split=args.train_split,
        max_context_turns=args.max_context_turns
    )
    fine_tuner.print_stats()
    
    if not training_data:
        print("Error: No valid training data found")
        return 1
    
    # Save training and validation data to files
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    base_path = f"data/{base_name}_training"
    train_file_path, val_file_path = fine_tuner.save_training_data(training_data, validation_data, base_path)
    
    # Upload training and validation files
    try:
        print("\nUploading training file...")
        training_file_id = fine_tuner.upload_training_file(train_file_path)
        
        validation_file_id = None
        if validation_data:
            print("Uploading validation file...")
            validation_file_id = fine_tuner.upload_training_file(val_file_path)
    except Exception as e:
        print(f"Error uploading training files: {e}")
        return 1
    
    # Start fine-tuning
    try:
        job_id = fine_tuner.start_fine_tuning(
            training_file_id=training_file_id,
            validation_file_id=validation_file_id,
            base_model=args.model,
            n_epochs=args.epochs,
            suffix=args.suffix,
            learning_rate=args.learning_rate,
            lora=use_lora,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"Error starting fine-tuning: {e}")
        return 1
    
    # Prepare training parameters record
    training_params = {
        'input_file': args.input,
        'train_file_path': train_file_path,
        'val_file_path': val_file_path,
        'training_file_id': training_file_id,
        'validation_file_id': validation_file_id,
        'base_model': args.model,
        'n_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'lora': use_lora,
        'batch_size': args.batch_size,
        'suffix': args.suffix,
        'max_conversations': args.max_conversations,
        'train_split': args.train_split,
        'max_context_turns': args.max_context_turns,
        'check_interval': args.check_interval,
        'no_monitor': args.no_monitor
    }
    
    # Save initial job info
    results_file = f"models/{base_name}_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    if args.no_monitor:
        # Just save initial job info and exit
        try:
            job_info = fine_tuner.client.fine_tuning.retrieve(job_id)
            fine_tuner.save_results(job_info, results_file, training_params)
            print(f"\nFine-tuning job started. Check status with:")
            print(f"  Job ID: {job_id}")
            print(f"  Results will be saved to: {results_file}")
        except Exception as e:
            print(f"Error saving initial results: {e}")
        return 0
    
    # Monitor training
    try:
        final_job_info = fine_tuner.monitor_training(job_id, args.check_interval)
        fine_tuner.save_results(final_job_info, results_file, training_params)
        
        print(f"\n🎉 Fine-tuning complete!")
        if hasattr(final_job_info, 'output_name') and getattr(final_job_info, 'output_name', None):
            print(f"Fine-tuned model: {getattr(final_job_info, 'output_name')}")
        print(f"Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Monitoring interrupted by user")
        print(f"Training job continues running with ID: {job_id}")
        try:
            job_info = fine_tuner.client.fine_tuning.retrieve(job_id)
            fine_tuner.save_results(job_info, results_file, training_params)
        except:
            pass
        return 1
    except Exception as e:
        print(f"Error during monitoring: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())