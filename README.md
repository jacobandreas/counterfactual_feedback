# Conversation Processing with Feedback

This project processes conve## How it Works

### Two Generation Methods:

#### 1. Feedback-based Method (`--method feedback`)
- **UR5 (Positive Feedback)**: Conversations are ### `evaluate.py`
Compares two models using a judge model to determine which generates better responses.

**Key Features:**
- Uses the last 1000 conversations from `data/filtered_conversations.json` as test set (contains original conversations with user feedback)
- Removes the last two turns (assistant response + user feedback) to create evaluation context
- Generates responses from both test models for each conversation context
- Uses a judge model to compare responses and pick the winner
- Provides statistical analysis with confidence intervals and significance tests
- Saves comprehensive results including all evaluations and metadata

**Data Requirements:**
- Conversations must end with: `[..., assistant_response, user_feedback]`
- Use original conversation files like `data/conversations.json` or `data/filtered_conversations.json`
- Do NOT use processed files from `process_conversations.py` (these only contain up to assistant response)anged
- **UR2/UR3 (Negative Feedback)**: 
  - Identifies the last assistant response before user feedback
  - Sends conversation + feedback to Together API with improvement prompt
  - Replaces assistant response with improved version based on user feedback
  - Adds metadata about the improvement

#### 2. Alternative Generation Method (`--method alternative`)
- **UR5 (Positive Feedback)**: Conversations are kept unchanged
- **UR2/UR3 (Negative Feedback)**:
  - Identifies the last assistant response before user feedback
  - Sends conversation context (without feedback) to Together API
  - Generates a completely new assistant response to the user's question
  - Ignores the user feedback entirely, creating an independent alternative between users and AI assistants, using the Together API to improve assistant responses based on negative user feedback.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a Together API key:**
   - Sign up at [Together.ai](https://together.ai/)
   - Get your API key from the dashboard
   - Set it as an environment variable:
     ```bash
     export TOGETHER_API_KEY="your-api-key-here"
     ```

## Usage

### 1. Filter conversations (if not done already)
```bash
python filter_conversations.py
```
This creates `data/filtered_conversations_ur235.json` with UR2, UR3, and UR5 conversations.

### 2. Test the processing system
```bash
python test_processing.py
```
This runs tests to verify the logic and API connectivity.

### 3. Process conversations

**Test with a small sample first:**
```bash
python process_conversations.py --max-conversations 10 --output data/test_improved_conversations.json
```

**Process all conversations:**
```bash
python process_conversations.py
```

## Script Details

### `filter_conversations.py`
- Filters original conversations to keep only UR2, UR3, UR5 categories
- UR2 & UR3 = negative feedback (need improvement)
- UR5 = positive feedback (keep unchanged)

### `process_conversations.py`
Main processing script with two generation methods:

**Feedback-based method (default)**: Uses user feedback to improve responses
**Alternative generation method**: Generates new responses without using feedback

Options:
- `--input` / `-i`: Input JSON file (default: `data/filtered_conversations_ur235.json`)
- `--output` / `-o`: Output JSON file (default: `data/improved_conversations.json`)
- `--method`: Generation method - `feedback` or `alternative` (default: `feedback`)
- `--max-conversations` / `-n`: Limit number of conversations (for testing)
- `--delay` / `-d`: Delay between API calls in seconds (default: 0.1)
- `--model` / `-m`: Together API model to use

### `test_processing.py`
Test suite to verify functionality before processing large datasets.

## How it Works

1. **UR5 (Positive Feedback)**: Conversations are kept unchanged
2. **UR2/UR3 (Negative Feedback)**: 
   - Identifies the last assistant response before user feedback
   - Sends conversation to Together API with improvement prompt
   - Replaces assistant response with improved version
   - Adds metadata about the improvement

## Output Format

Improved conversations include additional fields:
- `improved`: Boolean indicating if conversation was modified
- `generation_method`: Either "feedback_based" or "alternative_generation"
- `original_assistant_response`: The original assistant response (for comparison)
- `improvement_timestamp`: When the improvement was made

## Comparing Methods

To compare the effectiveness of both methods:

```bash
# Generate responses using feedback
python process_conversations.py --method feedback -n 1000 --output data/feedback_responses.json

# Generate responses without feedback
python process_conversations.py --method alternative -n 1000 --output data/alternative_responses.json

# Then compare the quality of responses in both files
```

## Rate Limiting

The script includes a delay between API calls to respect rate limits. Adjust with `--delay` parameter if needed.

## Examples

**Quick test with feedback method:**
```bash
python process_conversations.py --method feedback -n 5 -o test_feedback.json
```

**Quick test with alternative method:**
```bash
python process_conversations.py --method alternative -n 5 -o test_alternative.json
```

**Full processing with feedback (default):**
```bash
python process_conversations.py
```

**Full processing with alternative generation:**
```bash
python process_conversations.py --method alternative -o data/alternative_conversations.json
```

**Compare both methods on same data:**
```bash
python process_conversations.py -m "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" --method feedback -n 100 -o feedback_results.json
python process_conversations.py -m "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" --method alternative -n 100 -o alternative_results.json
```

## Fine-tuning

After generating improved conversations, you can fine-tune models using the generated data:

### `finetune.py`
Fine-tune models using the generated conversations via Together API.

**Key Features:**
- Only trains on the last assistant message in each conversation (using loss masking)
- Supports both LoRA and full fine-tuning
- Automatic data preparation and JSONL formatting
- Job monitoring with progress updates
- Configurable training parameters

**Usage:**
```bash
# Basic fine-tuning with default settings
python finetune.py --input data/feedback_1k_8b.json

# Custom model and parameters
python finetune.py -i data/improved_conversations.json -m meta-llama/Meta-Llama-3.1-8B-Instruct-Reference -e 5 --suffix my_model

# Full fine-tuning instead of LoRA
python finetune.py --input data/feedback_responses.json --full

# Start training without monitoring
python finetune.py --input data/alternative_responses.json --no-monitor
```

**Options:**
- `--input` / `-i`: Input JSON file with generated conversations
- `--model` / `-m`: Base model to fine-tune (default: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo)
- `--epochs` / `-e`: Number of training epochs (default: 3)
- `--learning-rate` / `-lr`: Learning rate (default: 1e-5)
- `--suffix` / `-s`: Suffix for fine-tuned model name
- `--lora`: Use LoRA fine-tuning (default)
- `--full`: Use full fine-tuning instead of LoRA
- `--batch-size` / `-b`: Training batch size
- `--no-monitor`: Start training without waiting for completion
- `--check-interval`: Status check interval in seconds (default: 30)

**Data Preparation:**
The script automatically prepares training data by:
- Converting conversations to JSONL format required by Together API
- Adding weight=1 to the last assistant message (target for training)  
- Adding weight=0 to all other messages (masked from loss calculation)
- Filtering out conversations without assistant responses

**Output Files:**
The fine-tuning process creates several files:
- **Training Data**: `data/{input_name}_training.jsonl` - JSONL formatted data for Together API
- **Results**: `models/{input_name}_finetune_{timestamp}.json` - Complete training record including:
  - Job status and model information
  - Training parameters (input file, model, epochs, learning rate, etc.)
  - Training statistics (conversations processed, messages, etc.)
  - File IDs for tracking and reproduction

**Testing:**
```bash
python test_finetune.py
```

## Model Evaluation

After processing conversations and/or fine-tuning models, you can evaluate them using the LM-as-judge evaluation system:

### `evaluate.py`
Compares two models using a judge model to determine which generates better responses.

**Key Features:**
- Uses the last 1000 conversations from `data/conversations.json` as test set
- Generates responses from both test models for each conversation context
- Uses a judge model to compare responses and pick the winner
- Provides statistical analysis with confidence intervals and significance tests
- Saves comprehensive results including all evaluations and metadata

**Usage:**
```bash
# Basic model comparison
python evaluate.py --model1 meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --model2 meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --judge meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo

# Quick test with smaller dataset
python evaluate.py -m1 model1 -m2 model2 -j judge_model --test-size 50

# Use custom conversations file
python evaluate.py -m1 model1 -m2 model2 -j judge_model -c data/feedback_1k_8b.json --test-size 100

# Custom output location
python evaluate.py -m1 model1 -m2 model2 -j judge_model -o results/my_evaluation.json
```

**Options:**
- `--model1` / `-m1`: First model to evaluate (required)
- `--model2` / `-m2`: Second model to evaluate (required)  
- `--judge` / `-j`: Judge model for comparison (required)
- `--conversations` / `-c`: Path to conversations file (default: data/conversations.json)
- `--test-size` / `-n`: Number of conversations to use (default: 1000)
- `--delay` / `-d`: Delay between API calls in seconds (default: 0.5)
- `--output` / `-o`: Output file path (default: auto-generated in results/)

**Example Use Cases:**
```bash
# Compare feedback vs alternative generation models
python evaluate.py -m1 feedback_trained_model -m2 alternative_trained_model -j meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo

# Compare base model vs fine-tuned model
python evaluate.py -m1 meta-llama/Meta-Llama-3.1-8B-Instruct-Reference -m2 your_finetuned_model -j meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo

# Quick comparison for development
python evaluate.py -m1 model_v1 -m2 model_v2 -j judge_model --test-size 100 --delay 0.2
```

**Output:**
Evaluation results are saved in JSON format containing:
- Win/loss/tie statistics for both models
- Statistical significance analysis
- Confidence intervals for win rates
- Complete evaluation details for each conversation
- Metadata about models, timing, and parameters

**Testing:**
