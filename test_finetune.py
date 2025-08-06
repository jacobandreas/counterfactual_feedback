#!/usr/bin/env python3
"""
Test script for the fine-tuning functionality.
"""

import json
import os
from finetune import FineTuner

def create_test_data():
    """Create test conversation data for testing the fine-tuning script."""
    test_conversations = [
        {
            "conversation_id": "test_1",
            "conversation_context": [
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ],
            "category": "UR5"
        },
        {
            "conversation_id": "test_2", 
            "conversation_context": [
                {"role": "user", "content": "Tell me about Python."},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "Can you be more specific?"},
                {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for its simplicity and readability."}
            ],
            "category": "UR2"
        },
        {
            "conversation_id": "test_3",
            "conversation_context": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ],
            "category": "UR5"
        }
    ]
    return test_conversations

def test_data_preparation():
    """Test the data preparation functionality."""
    print("=== Testing Fine-tuning Data Preparation ===\n")
    
    # Create test data
    test_conversations = create_test_data()
    print(f"Created {len(test_conversations)} test conversations")
    
    # Test fine-tuner (without API key for testing)
    class MockFineTuner(FineTuner):
        def __init__(self):
            self.stats = {
                'total_conversations': 0,
                'processed_conversations': 0,
                'skipped_conversations': 0,
                'total_messages': 0,
                'training_messages': 0
            }
    
    fine_tuner = MockFineTuner()
    
    # Prepare training data
    training_data = fine_tuner.prepare_training_data(test_conversations)
    
    print(f"\nPrepared {len(training_data)} training examples")
    fine_tuner.print_stats()
    
    # Show examples of prepared data
    print(f"\n=== Training Data Examples ===")
    for i, example in enumerate(training_data):
        print(f"\nExample {i+1}:")
        for j, message in enumerate(example['messages']):
            weight_info = f" (weight: {message['weight']})" if 'weight' in message else ""
            role = message['role'].upper()
            content = message['content'][:50] + "..." if len(message['content']) > 50 else message['content']
            print(f"  {j+1}. {role}: {content}{weight_info}")
    
    # Verify that only the last assistant messages have weight 1.0
    print(f"\n=== Verification ===")
    correct_weighting = True
    for i, example in enumerate(training_data):
        messages = example['messages']
        
        # Find last assistant message
        last_assistant_idx = None
        for j in range(len(messages) - 1, -1, -1):
            if messages[j]['role'] == 'assistant':
                last_assistant_idx = j
                break
        
        if last_assistant_idx is None:
            print(f"❌ Example {i+1}: No assistant message found")
            correct_weighting = False
            continue
        
        # Check weights
        for j, message in enumerate(messages):
            expected_weight = 1 if (j == last_assistant_idx and message['role'] == 'assistant') else 0
            actual_weight = message.get('weight', 0)
            
            if actual_weight != expected_weight:
                print(f"❌ Example {i+1}, Message {j+1}: Expected weight {expected_weight}, got {actual_weight}")
                correct_weighting = False
    
    if correct_weighting:
        print("✅ All examples have correct weighting (only last assistant message has weight 1)")
    else:
        print("❌ Some examples have incorrect weighting")
    
    return training_data

def test_jsonl_format():
    """Test JSONL file creation."""
    print(f"\n=== Testing JSONL Format ===")
    
    training_data = test_data_preparation()
    
    # Save to JSONL
    test_file = "data/test_training.jsonl"
    
    class MockFineTuner(FineTuner):
        def __init__(self):
            pass
    
    fine_tuner = MockFineTuner()
    saved_path = fine_tuner.save_training_data(training_data, test_file)
    
    # Verify JSONL format
    print(f"\nVerifying JSONL format in {test_file}:")
    try:
        with open(test_file, 'r') as f:
            lines = f.readlines()
        
        print(f"✅ File has {len(lines)} lines")
        
        # Check that each line is valid JSON
        for i, line in enumerate(lines):
            try:
                data = json.loads(line.strip())
                if 'messages' not in data:
                    print(f"❌ Line {i+1}: Missing 'messages' key")
                else:
                    print(f"✅ Line {i+1}: Valid JSON with {len(data['messages'])} messages")
            except json.JSONDecodeError as e:
                print(f"❌ Line {i+1}: Invalid JSON - {e}")
        
        # Show first few lines
        print(f"\nFirst line content:")
        print(json.dumps(json.loads(lines[0]), indent=2))
        
    except Exception as e:
        print(f"❌ Error reading JSONL file: {e}")
    
    # Clean up
    try:
        os.remove(test_file)
        print(f"✅ Test file cleaned up")
    except:
        pass

def main():
    """Run all tests."""
    print("Testing fine-tuning script functionality...\n")
    
    # Test data preparation
    test_data_preparation()
    
    # Test JSONL format
    test_jsonl_format()
    
    print(f"\n🎯 Usage Examples:")
    print(f"  # Fine-tune with default settings:")
    print(f"  python finetune.py --input data/feedback_1k_8b.json")
    print(f"  ")
    print(f"  # Fine-tune with custom parameters:")
    print(f"  python finetune.py -i data/feedback_1k_8b.json -m meta-llama/Meta-Llama-3.1-8B-Instruct-Reference -e 5 --suffix my_model")
    print(f"  ")
    print(f"  # Start training without monitoring:")
    print(f"  python finetune.py --input data/feedback_1k_8b.json --no-monitor")
    print(f"  ")
    print(f"  # Full fine-tuning instead of LoRA:")
    print(f"  python finetune.py --input data/feedback_1k_8b.json --full")
    
    return 0

if __name__ == "__main__":
    exit(main())
