# Complete Workflow: From Conversation Processing to Model Evaluation

This document outlines the complete workflow for processing conversations, fine-tuning models, and evaluating their performance using the counterfactual feedback system.

## Overview

This system provides three main capabilities:
1. **Conversation Processing**: Compare feedback-based vs alternative generation methods
2. **Model Fine-tuning**: Train models on generated conversations via Together API
3. **Model Evaluation**: Compare model performance using LM-as-judge methodology

## Step-by-Step Workflow

### 1. Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set your Together API key
export TOGETHER_API_KEY="your-api-key-here"
```

### 2. Process Conversations (Choose Method)

#### Option A: Feedback-based Method
```bash
# Test with small sample first
python process_conversations.py --method feedback -n 10 -o data/test_feedback.json

# Process full dataset
python process_conversations.py --method feedback -o data/feedback_improved.json
```

#### Option B: Alternative Generation Method
```bash
# Test with small sample first
python process_conversations.py --method alternative -n 10 -o data/test_alternative.json

# Process full dataset
python process_conversations.py --method alternative -o data/alternative_improved.json
```

#### Option C: Generate Both for Comparison
```bash
# Generate both methods on same data for comparison
python process_conversations.py --method feedback -n 1000 -o data/feedback_1k.json
python process_conversations.py --method alternative -n 1000 -o data/alternative_1k.json
```

### 3. Fine-tune Models

#### Basic Fine-tuning
```bash
# Fine-tune on feedback-based improvements
python finetune.py --input data/feedback_improved.json

# Fine-tune on alternative generations
python finetune.py --input data/alternative_improved.json
```

#### Advanced Fine-tuning Options
```bash
# Custom model and parameters
python finetune.py \\
  --input data/feedback_improved.json \\
  --model meta-llama/Meta-Llama-3.1-8B-Instruct-Reference \\
  --epochs 5 \\
  --learning-rate 2e-5 \\
  --suffix my_experiment

# Full fine-tuning instead of LoRA
python finetune.py --input data/feedback_improved.json --full

# Start training without monitoring (useful for batch jobs)
python finetune.py --input data/feedback_improved.json --no-monitor
```

### 4. Evaluate Models

#### Compare Different Generation Methods
```bash
# Compare feedback vs alternative trained models
python evaluate.py \\
  --model1 your_feedback_trained_model \\
  --model2 your_alternative_trained_model \\
  --judge meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \\
  --test-size 1000

# Compare base model vs fine-tuned model
python evaluate.py \\
  --model1 meta-llama/Meta-Llama-3.1-8B-Instruct-Reference \\
  --model2 your_finetuned_model \\
  --judge meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
```

#### Quick Development Comparisons
```bash
# Fast comparison for iterative development
python evaluate.py -m1 model_v1 -m2 model_v2 -j judge_model --test-size 100 --delay 0.2

# Use processed conversations as test set
python evaluate.py -m1 model1 -m2 model2 -j judge_model -c data/feedback_1k.json --test-size 500
```

### 5. Testing and Validation

#### Test Processing Logic
```bash
python test_processing.py
```

#### Test Fine-tuning Logic
```bash
python test_finetune.py
```

#### Test Evaluation Logic
```bash
python test_evaluate.py
```

#### End-to-End Testing
```bash
# Test complete pipeline with small datasets
python process_conversations.py --method feedback -n 5 -o data/test_feedback.json
python finetune.py --input data/test_feedback.json --max-conversations 1 --no-monitor --suffix test
python evaluate.py -m1 base_model -m2 your_test_model -j judge_model --test-size 10
```

## Complete Research Pipeline

### Comprehensive Comparison Study
```bash
# 1. Generate conversations with both methods
python process_conversations.py --method feedback -n 5000 -o data/feedback_5k.json
python process_conversations.py --method alternative -n 5000 -o data/alternative_5k.json

# 2. Fine-tune models on each method
python finetune.py --input data/feedback_5k.json --suffix feedback_study
python finetune.py --input data/alternative_5k.json --suffix alternative_study

# 3. Evaluate all combinations
python evaluate.py -m1 base_model -m2 feedback_trained_model -j judge_model -o results/base_vs_feedback.json
python evaluate.py -m1 base_model -m2 alternative_trained_model -j judge_model -o results/base_vs_alternative.json  
python evaluate.py -m1 feedback_trained_model -m2 alternative_trained_model -j judge_model -o results/feedback_vs_alternative.json

# 4. Analyze results
# Load and compare JSON files in results/ directory
```

## Key Features

### Conversation Processing
- **Two Methods**: Compare feedback-based vs alternative generation
- **Flexible Models**: Support for any Together API model
- **Rate Limiting**: Built-in delays to respect API limits
- **Progress Tracking**: Real-time status updates
- **Error Handling**: Robust error recovery

### Fine-tuning
- **Loss Masking**: Only trains on the last assistant message
- **LoRA Support**: Efficient fine-tuning with LoRA (default)
- **Full Fine-tuning**: Option for complete model fine-tuning
- **Job Monitoring**: Real-time training progress updates
- **Format Validation**: Automatic JSONL format checking

## Output Files

### Conversation Processing Output
```json
{
  "conversation_id": "...",
  "improved": true,
  "generation_method": "feedback_based",
  "original_assistant_response": "...",
  "conversation_context": [...],
  "improvement_timestamp": "..."
}
```

### Fine-tuning Output
- **Training Data**: `data/{input_name}_training.jsonl`
- **Results**: `models/{input_name}_finetune_{timestamp}.json`
- **Model**: Available via Together API after training completes

## Best Practices

### For Processing
1. **Start Small**: Test with `--max-conversations 10` first
2. **Monitor Rate Limits**: Adjust `--delay` if needed
3. **Save Intermediate Results**: Use descriptive output filenames
4. **Compare Methods**: Run both methods on same data subset

### For Fine-tuning
1. **Validate Data**: Use `test_finetune.py` to verify data format
2. **Test First**: Use `--max-conversations 1` for initial testing
3. **Monitor Training**: Let monitoring run for first few jobs
4. **Save Results**: Models and logs are automatically saved

### For Comparison Studies
1. **Use Same Data**: Process identical conversation sets with both methods
2. **Document Parameters**: Save model names, hyperparameters, and timestamps
3. **Evaluate Results**: Compare fine-tuned model performance

## Troubleshooting

### Common Issues
- **API Key**: Ensure `TOGETHER_API_KEY` is set
- **Rate Limits**: Increase `--delay` if getting rate limit errors
- **File Format**: Integer weights (0,1) required, not floats (0.0,1.0)
- **Memory**: Large datasets may need processing in batches

### Debugging Commands
```bash
# Check API connectivity
python -c "import os; from together import Together; print('API Key set:', bool(os.getenv('TOGETHER_API_KEY')))"

# Validate conversation format
python -c "import json; data=json.load(open('data/conversations.json')); print(f'Conversations: {len(data)}')"

# Check fine-tuning job status
python -c "from together import Together; client=Together(); print(client.fine_tuning.list())"
```

## Performance Tips

1. **Batch Processing**: Process conversations in chunks for large datasets
2. **Parallel Jobs**: Run multiple fine-tuning jobs with different parameters
3. **Model Selection**: Start with smaller models for faster iteration
4. **LoRA vs Full**: Use LoRA for faster training, full for maximum customization

## Next Steps

After completing the workflow:
1. **Evaluate Models**: Test fine-tuned models on held-out conversations
2. **Compare Methods**: Analyze which approach works better for your use case
3. **Iterate**: Adjust parameters and rerun experiments
4. **Deploy**: Use the best performing model in production

---

For more details, see the main [README.md](README.md) file.
