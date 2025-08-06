#!/usr/bin/env python3
"""
Test script for the evaluation functionality.
"""

import json
import os
from evaluate import ModelEvaluator

def create_test_conversations():
    """Create test conversation data for testing the evaluation script."""
    test_conversations = [
        {
            "conversation_id": "test_1",
            "conversation_context": [
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "Thanks, that's correct!"}
            ],
            "category": "UR5"
        },
        {
            "conversation_id": "test_2", 
            "conversation_context": [
                {"role": "user", "content": "Tell me about Python."},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "Can you be more specific?"},
                {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for its simplicity and readability."},
                {"role": "user", "content": "That's much better, thank you."}
            ],
            "category": "UR2"
        },
        {
            "conversation_id": "test_3",
            "conversation_context": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "Perfect!"}
            ],
            "category": "UR5"
        }
    ]
    return test_conversations

def test_data_preparation():
    """Test the data preparation functionality."""
    print("=== Testing Evaluation Data Preparation ===\n")
    
    # Create test data
    test_conversations = create_test_conversations()
    print(f"Created {len(test_conversations)} test conversations")
    
    # Test evaluator (without API key for testing data prep)
    class MockEvaluator(ModelEvaluator):
        def __init__(self):
            # Skip API initialization for testing
            self.results = {
                'model1_wins': 0,
                'model2_wins': 0,
                'ties': 0,
                'errors': 0,
                'total_evaluations': 0,
                'evaluations': []
            }
    
    evaluator = MockEvaluator()
    
    # Test conversation context preparation
    print(f"\n=== Testing Context Preparation ===")
    for i, conv in enumerate(test_conversations):
        context = evaluator.prepare_conversation_context(conv)
        original = evaluator.get_original_response(conv)
        
        print(f"\nConversation {i+1}:")
        print(f"  Original context length: {len(conv['conversation_context'])}")
        print(f"  Prepared context length: {len(context)}")
        print(f"  Original response: {original[:50]}...")
        
        # Show the context
        for j, turn in enumerate(context):
            role = turn['role'].upper()
            content = turn['content'][:30] + "..." if len(turn['content']) > 30 else turn['content']
            print(f"    {j+1}. {role}: {content}")
    
    # Test judge prompt creation
    print(f"\n=== Testing Judge Prompt Creation ===")
    conv = test_conversations[0]
    context = evaluator.prepare_conversation_context(conv)
    original = evaluator.get_original_response(conv)
    
    response1 = "The answer is 4."
    response2 = "Two plus two equals four."
    
    judge_messages = evaluator.create_judge_prompt(context, response1, response2, original)
    
    print(f"Judge prompt created with {len(judge_messages)} messages")
    print(f"First 200 chars of prompt:")
    print(judge_messages[0]['content'][:200] + "...")
    
    return True

def test_statistics():
    """Test the statistical calculation functionality."""
    print(f"\n=== Testing Statistics Calculation ===")
    
    class MockEvaluator(ModelEvaluator):
        def __init__(self):
            self.results = {
                'model1_wins': 15,
                'model2_wins': 10,
                'ties': 5,
                'errors': 0,
                'total_evaluations': 30,
                'evaluations': []
            }
    
    evaluator = MockEvaluator()
    stats = evaluator.calculate_statistics()
    
    print(f"Sample statistics calculation:")
    print(f"  Model 1 win rate: {stats['model1_win_rate']:.1%}")
    print(f"  Model 2 win rate: {stats['model2_win_rate']:.1%}")
    print(f"  Tie rate: {stats['tie_rate']:.1%}")
    print(f"  Model 1 confidence interval: {stats['model1_confidence_interval_95']}")
    print(f"  Statistical test p-value: {stats['statistical_test']['p_value']:.4f}")
    print(f"  Significant difference: {stats['statistical_test']['significant_at_0.05']}")
    print(f"  Interpretation: {stats['statistical_test']['interpretation']}")
    
    return True

def test_file_operations():
    """Test file loading and result saving."""
    print(f"\n=== Testing File Operations ===")
    
    # Create a temporary test file
    test_data = create_test_conversations()
    test_file = "data/test_conversations_eval.json"
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Created test file: {test_file}")
    
    # Test loading
    class MockEvaluator(ModelEvaluator):
        def __init__(self):
            pass
    
    evaluator = MockEvaluator()
    loaded_conversations = evaluator.load_test_set(test_file, test_size=2)
    
    print(f"Loaded {len(loaded_conversations)} conversations from test file")
    
    # Clean up
    try:
        os.remove(test_file)
        print(f"✅ Test file cleaned up")
    except:
        pass
    
    return True

def main():
    """Run all tests."""
    print("Testing evaluation script functionality...\n")
    
    try:
        # Test data preparation
        test_data_preparation()
        
        # Test statistics
        test_statistics()
        
        # Test file operations
        test_file_operations()
        
        print(f"\n🎯 Usage Examples:")
        print(f"  # Compare two models with a judge:")
        print(f"  python evaluate.py --model1 meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --model2 meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --judge meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
        print(f"  ")
        print(f"  # Quick test with smaller dataset:")
        print(f"  python evaluate.py -m1 model1 -m2 model2 -j judge_model --test-size 50")
        print(f"  ")
        print(f"  # Use custom conversations file:")
        print(f"  python evaluate.py -m1 model1 -m2 model2 -j judge_model -c data/feedback_1k_8b.json --test-size 100")
        
        print(f"\n✅ All tests passed!")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
