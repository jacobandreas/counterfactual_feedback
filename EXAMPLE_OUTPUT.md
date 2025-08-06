# Example Fine-tuning Results File

This is an example of what gets saved in `models/{name}_finetune_{timestamp}.json` after running the fine-tuning script:

```json
{
  "job_id": "ft-3cab62db-643b",
  "status": "completed",
  "output_name": "lingo_mit/Meta-Llama-3.1-8B-Instruct-Reference-test_params-9cbbd9b3",
  "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
  "created_at": "2025-07-25T14:03:16.78Z",
  "finished_at": "2025-07-25T14:45:22.15Z",
  "hyperparameters": {
    "batch_size": 32,
    "learning_rate": 1e-05,
    "n_epochs": 3,
    "lora": true
  },
  "training_file_id": "file-a7cf120a-0d73-40d6-807a-1933689b4e31",
  "timestamp": "2025-07-25T10:03:19.792997",
  "training_stats": {
    "total_conversations": 1000,
    "processed_conversations": 987,
    "skipped_conversations": 13,
    "total_messages": 3254,
    "training_messages": 987
  },
  "training_parameters": {
    "input_file": "data/feedback_1k_8b.json",
    "training_file_path": "data/feedback_1k_8b_training.jsonl",
    "training_file_id": "file-a7cf120a-0d73-40d6-807a-1933689b4e31",
    "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    "n_epochs": 3,
    "learning_rate": 1e-05,
    "lora": true,
    "batch_size": "max",
    "suffix": "test_params",
    "max_conversations": null,
    "check_interval": 30,
    "no_monitor": false
  }
}
```

## Key Fields Explained:

### Job Information
- **job_id**: Together API job identifier for tracking
- **status**: Current status (pending, running, completed, failed)
- **output_name**: Name of the fine-tuned model
- **created_at/finished_at**: Timestamps for job lifecycle

### Training Configuration
- **training_parameters**: Complete record of all command-line arguments and settings
  - **input_file**: Original conversation file used for training
  - **training_file_path**: Generated JSONL file path
  - **training_file_id**: Together API file ID for the training data
  - All model parameters (epochs, learning rate, LoRA settings, etc.)

### Data Statistics
- **training_stats**: Statistics about data processing
  - **total_conversations**: Number of conversations in input file
  - **processed_conversations**: Successfully processed conversations
  - **skipped_conversations**: Conversations without valid assistant messages
  - **training_messages**: Number of messages with weight=1 (actual training targets)

### API Response
- **hyperparameters**: Final hyperparameters used by Together API
- **training_file_id**: Reference to uploaded training data

This comprehensive record allows you to:
1. **Reproduce experiments** exactly
2. **Track data sources** and preprocessing steps
3. **Compare different training runs**
4. **Debug training issues**
5. **Audit model training provenance**
