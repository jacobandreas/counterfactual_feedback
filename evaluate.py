#!/usr/bin/env python3
"""
Evaluation script for comparing models using LM-as-judge paradigm.

This script:
1. Takes the last 1000 conversations from data/conversations.json as test set
2. Generates responses from two test models for each conversation context
3. Uses a judge model to compare the responses and pick the winner
4. Computes statistics and saves results to JSON

Usage:
    python evaluate.py --model1 model_name_1 --model2 model_name_2 --judge judge_model_name
"""

import json
import os
import time
import argparse
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from together import Together
import math

class ModelEvaluator:
    """Handles model evaluation using LM-as-judge paradigm."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the evaluator with Together API."""
        if api_key is None:
            api_key = os.getenv('TOGETHER_API_KEY')
        
        if not api_key:
            raise ValueError("Together API key is required. Set TOGETHER_API_KEY environment variable.")
        
        self.client = Together(api_key=api_key)
        self.results = {
            'model1_wins': 0,
            'model2_wins': 0,
            'ties': 0,
            'errors': 0,
            'total_evaluations': 0,
            'evaluations': []
        }
    
    def load_test_set(self, path: str = "data/filtered_conversations.json", limit: int = 1000) -> List[Dict[str, Any]]:
        """Load conversations for evaluation testing.
        
        Uses original conversation files that end with: [..., assistant_response, user_feedback]
        Do NOT use processed files from process_conversations.py (those only contain up to assistant response).
        
        Args:
            path: Path to conversation file (use original/filtered files, not processed ones)
            limit: Maximum number of conversations to load (from the end)
        """
        
        print(f"Loading test set from {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            all_conversations = json.load(f)
        
        # Take the last 'limit' conversations
        test_conversations = all_conversations[-limit:]
        print(f"Loaded {len(test_conversations)} conversations for testing")
        
        # Filter to only use conversations with at least 3 turns and proper structure
        # We need: [..., assistant_response, user_feedback]
        valid_conversations = []
        for conv in test_conversations:
            context = conv.get('conversation_context', [])
            if (len(context) >= 3 and 
                context[-1].get('role') == 'user' and  # Last turn is user feedback
                context[-2].get('role') == 'assistant'):  # Second-to-last is assistant response
                valid_conversations.append(conv)
        
        print(f"Found {len(valid_conversations)} valid conversations (with assistant responses followed by user feedback)")
        return valid_conversations
    
    def prepare_conversation_context(self, conversation: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Prepare conversation context by removing the last two turns (assistant response + user feedback).
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            List of messages without the last assistant response and user feedback
        """
        context = conversation.get('conversation_context', [])
        
        # Remove the last two turns: assistant response and user feedback
        # This leaves us with the conversation context up to the point where we want to generate a response
        if len(context) >= 2:
            context_without_last_two = context[:-2]
        else:
            context_without_last_two = []
        
        # Convert to the expected format
        prepared_context = []
        for turn in context_without_last_two:
            prepared_context.append({
                'role': turn.get('role', ''),
                'content': turn.get('content', '')
            })
        
        return prepared_context
    
    def get_original_response(self, conversation: Dict[str, Any]) -> str:
        """Get the original assistant response from the conversation (second-to-last turn)."""
        context = conversation.get('conversation_context', [])
        # The assistant response should be the second-to-last turn
        if len(context) >= 2 and context[-2].get('role') == 'assistant':
            return context[-2].get('content', '')
        return ""
    
    def generate_model_response(self, model_name: str, context: List[Dict[str, str]], max_retries: int = 3) -> Optional[str]:
        """
        Generate a response from the specified model given the conversation context.
        
        Args:
            model_name: Name of the model to use
            context: Conversation context (list of messages)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated response or None if failed
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=context,
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.0
                )
                
                # Extract content from response with safe access
                try:
                    content = response.choices[0].message.content  # type: ignore
                    if content:
                        return str(content).strip()
                except (AttributeError, IndexError, TypeError):
                    pass
                
                return None
            except Exception as e:
                print(f"Error generating response with {model_name} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None
        return None
    
    def create_judge_prompt(self, context: List[Dict[str, str]], response1: str, response2: str, original: str) -> List[Dict[str, str]]:
        """
        Create a prompt for the judge model to compare two responses.
        
        Args:
            context: Conversation context
            response1: Response from model 1
            response2: Response from model 2
            original: Original response from the conversation
            
        Returns:
            Messages list for the judge model
        """
        # Convert context to a readable format
        context_text = ""
        for turn in context:
            role = turn['role'].upper()
            content = turn['content']
            context_text += f"{role}: {content}\n\n"

        judge_prompt = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants given the conversation displayed below. "
    "You should choose the assistant that follows the user's instructions and answers the user's question best. Your evaluation should consider "
    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
    "comparing the responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is best, "[[B]]" if assistant B is best, or "[[TIE]]" if both assistants are equally good.\n'
    f"\nConversation Context:\n{context_text}\n"
    f"Response A: {response1}\n"
    f"Response B: {response2}\n"
   )

        return [{"role": "user", "content": judge_prompt}]
    
    def judge_responses(self, judge_model: str, context: List[Dict[str, str]], 
                       response1: str, response2: str, original: str, 
                       max_retries: int = 3) -> Optional[str]:
        """
        Use the judge model to compare two responses.
        
        Args:
            judge_model: Name of the judge model
            context: Conversation context
            response1: Response from model 1
            response2: Response from model 2
            original: Original response from conversation
            max_retries: Maximum retry attempts
            
        Returns:
            Judge decision: "A", "B", "TIE", or None if failed
        """
        judge_messages = self.create_judge_prompt(context, response1, response2, original)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=judge_model,
                    messages=judge_messages,
                    max_tokens=10,
                    temperature=0.1,  # Lower temperature for more consistent judging
                    repetition_penalty=1.0
                )
                
                # Extract and parse judgment with safe access
                try:
                    content = response.choices[0].message.content  # type: ignore
                    if content:
                        judgment = str(content).strip().upper()
                        
                        # Parse the judgment
                        if "A" in judgment and "B" not in judgment:
                            return "A"
                        elif "B" in judgment and "A" not in judgment:
                            return "B"
                        elif "TIE" in judgment:
                            return "TIE"
                        else:
                            print(f"Ambiguous judgment: {judgment}")
                            if attempt == max_retries - 1:
                                return None
                except (AttributeError, IndexError, TypeError):
                    pass
                    
            except Exception as e:
                print(f"Error getting judgment (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def evaluate_conversation(self, conversation: Dict[str, Any], model1: str, model2: str, 
                            judge_model: str, conv_idx: int) -> Dict[str, Any]:
        """
        Evaluate a single conversation with both models and judge the results.
        
        Args:
            conversation: Conversation to evaluate
            model1: Name of first test model
            model2: Name of second test model
            judge_model: Name of judge model
            conv_idx: Index of conversation for logging
            
        Returns:
            Evaluation result dictionary
        """
        print(f"Evaluating conversation {conv_idx + 1}...")
        
        # Prepare context (without last assistant turn)
        context = self.prepare_conversation_context(conversation)
        original_response = self.get_original_response(conversation)
        
        if not context:
            return {
                'conversation_id': conversation.get('conversation_id', f'conv_{conv_idx}'),
                'status': 'error',
                'error': 'Empty context after removing last two turns (assistant response + user feedback)'
            }

        #print("\n\n\n\n\n\n\n")
        #print(context)
        
        # Generate responses from both models
        #print(f"  Generating response from {model1}...")
        response1 = self.generate_model_response(model1, context)
        #print(response1)
        
        #print(f"  Generating response from {model2}...")
        response2 = self.generate_model_response(model2, context)
        #print(response2)
        
        if response1 is None or response2 is None:
            return {
                'conversation_id': conversation.get('conversation_id', f'conv_{conv_idx}'),
                'status': 'error',
                'error': f'Failed to generate responses (model1: {response1 is not None}, model2: {response2 is not None})'
            }
        
        #print(f"  Generating judgment from {judge_model}...")
        # Randomize order for fair judging
        if random.random() < 0.5:
            # Model 1 as A, Model 2 as B
            judgment = self.judge_responses(judge_model, context, response1, response2, original_response)
            order = "normal"
        else:
            # Model 2 as A, Model 1 as B (reversed)
            judgment = self.judge_responses(judge_model, context, response2, response1, original_response)
            order = "reversed"
        
        if judgment is None:
            return {
                'conversation_id': conversation.get('conversation_id', f'conv_{conv_idx}'),
                'status': 'error',
                'error': 'Failed to get judgment from judge model'
            }

        #print("Judge picks answer:", judgment)
        
        # Convert judgment based on order
        if order == "reversed":
            if judgment == "A":
                judgment = "B"
            elif judgment == "B":
                judgment = "A"
            # TIE remains TIE
        
        # Map judgment to winner
        if judgment == "A":
            winner = "model1"
        elif judgment == "B":
            winner = "model2"
        else:  # TIE
            winner = "tie"

        #print("Judge picks model:", winner)
        
        return {
            'conversation_id': conversation.get('conversation_id', f'conv_{conv_idx}'),
            'status': 'success',
            'context': context,
            'model1_response': response1,
            'model2_response': response2,
            'original_response': original_response,
            'judgment': judgment,
            'winner': winner,
            'order': order
        }
    
    def run_evaluation(self, test_conversations: List[Dict[str, Any]], 
                      model1: str, model2: str, judge_model: str,
                      delay: float = 0.5) -> Dict[str, Any]:
        """
        Run the full evaluation on all test conversations.
        
        Args:
            test_conversations: List of conversations to evaluate
            model1: Name of first test model
            model2: Name of second test model  
            judge_model: Name of judge model
            delay: Delay between API calls (seconds)
            
        Returns:
            Complete evaluation results
        """
        print(f"\n=== Starting Evaluation ===")
        print(f"Model 1: {model1}")
        print(f"Model 2: {model2}")
        print(f"Judge: {judge_model}")
        print(f"Test conversations: {len(test_conversations)}")
        print(f"Delay between calls: {delay}s")
        
        start_time = datetime.now()
        
        for i, conversation in enumerate(test_conversations):
            # Evaluate this conversation
            result = self.evaluate_conversation(conversation, model1, model2, judge_model, i)
            self.results['evaluations'].append(result)
            
            # Update statistics
            if result['status'] == 'success':
                winner = result['winner']
                if winner == 'model1':
                    self.results['model1_wins'] += 1
                elif winner == 'model2':
                    self.results['model2_wins'] += 1
                else:  # tie
                    self.results['ties'] += 1
                self.results['total_evaluations'] += 1
            else:
                self.results['errors'] += 1
            
            # Progress update
            if (i + 1) % 10 == 0 or i == len(test_conversations) - 1:
                self.print_progress(i + 1, len(test_conversations))
            
            # Rate limiting
            if i < len(test_conversations) - 1:
                time.sleep(delay)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate final statistics
        stats = self.calculate_statistics()
        
        final_results = {
            'metadata': {
                'model1': model1,
                'model2': model2,  
                'judge_model': judge_model,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'total_conversations': len(test_conversations)
            },
            'summary': {
                'model1_wins': self.results['model1_wins'],
                'model2_wins': self.results['model2_wins'],
                'ties': self.results['ties'],
                'errors': self.results['errors'],
                'total_evaluations': self.results['total_evaluations'],
                'success_rate': self.results['total_evaluations'] / len(test_conversations)
            },
            'statistics': stats,
            'evaluations': self.results['evaluations']
        }
        
        return final_results
    
    def print_progress(self, completed: int, total: int):
        """Print evaluation progress."""
        pct = (completed / total) * 100
        print(f"Progress: {completed}/{total} ({pct:.1f}%) - "
              f"Model1: {self.results['model1_wins']}, "
              f"Model2: {self.results['model2_wins']}, "
              f"Ties: {self.results['ties']}, "
              f"Errors: {self.results['errors']}")
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistical measures for the evaluation results."""
        total = self.results['total_evaluations']
        if total == 0:
            return {'error': 'No successful evaluations'}
        
        model1_wins = self.results['model1_wins']
        model2_wins = self.results['model2_wins']
        ties = self.results['ties']
        
        # Win rates
        model1_win_rate = model1_wins / total
        model2_win_rate = model2_wins / total
        tie_rate = ties / total
        
        # Confidence intervals for win rates (Wilson score interval)
        def wilson_confidence_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
            if total == 0:
                return (0.0, 0.0)
            
            # Use approximation for z-score (1.96 for 95% confidence)
            z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
            p = successes / total
            
            denominator = 1 + z**2 / total
            center = (p + z**2 / (2 * total)) / denominator
            margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
            
            return (max(0, center - margin), min(1, center + margin))
        
        model1_ci = wilson_confidence_interval(model1_wins, total)
        model2_ci = wilson_confidence_interval(model2_wins, total)
        
        # Binomial test for statistical significance
        # H0: models are equally good (p = 0.5 for model1 wins among non-ties)
        non_tie_total = model1_wins + model2_wins
        if non_tie_total > 0:
            # Exact binomial test
            # Two-tailed test: calculate probability of seeing a result at least as extreme
            
            def binomial_coefficient(n: int, k: int) -> float:
                """Calculate binomial coefficient C(n,k) = n! / (k! * (n-k)!)"""
                if k > n or k < 0:
                    return 0.0
                if k == 0 or k == n:
                    return 1.0
                
                # Use the multiplicative formula to avoid overflow
                result = 1.0
                for i in range(min(k, n - k)):
                    result = result * (n - i) / (i + 1)
                return result
            
            def binomial_probability(n: int, k: int, p: float = 0.5) -> float:
                """Calculate P(X = k) where X ~ Binomial(n, p)"""
                return binomial_coefficient(n, k) * (p ** k) * ((1 - p) ** (n - k))
            
            # Two-tailed test calculation
            observed_successes = model1_wins
            n = non_tie_total
            
            # For a two-tailed test, we want P(result at least as extreme as observed)
            # If we observe k successes out of n trials:
            # - If k <= n/2: extreme results are k or fewer, OR (n-k) or more
            # - If k > n/2: extreme results are k or more, OR (n-k) or fewer
            
            if observed_successes <= n / 2:
                # Few successes observed: extreme = very few OR very many
                left_tail = sum(binomial_probability(n, k) for k in range(0, observed_successes + 1))
                right_tail = sum(binomial_probability(n, k) for k in range(n - observed_successes, n + 1))
                p_value = left_tail + right_tail
            else:
                # Many successes observed: extreme = very many OR very few
                right_tail = sum(binomial_probability(n, k) for k in range(observed_successes, n + 1))
                left_tail = sum(binomial_probability(n, k) for k in range(0, n - observed_successes + 1))
                p_value = left_tail + right_tail
            
            # Ensure p-value doesn't exceed 1.0 due to floating point errors
            p_value = min(1.0, p_value)
            
            significant = p_value < 0.05
        else:
            p_value = 1.0
            significant = False
        
        return {
            'total_evaluations': total,
            'model1_win_rate': model1_win_rate,
            'model2_win_rate': model2_win_rate,
            'tie_rate': tie_rate,
            'model1_confidence_interval_95': model1_ci,
            'model2_confidence_interval_95': model2_ci,
            'non_tie_evaluations': non_tie_total,
            'statistical_test': {
                'test': 'binomial_test',
                'null_hypothesis': 'models are equally good',
                'p_value': p_value,
                'significant_at_0.05': significant,
                'interpretation': f"Model 1 significantly {'better' if model1_wins > model2_wins else 'worse'} than Model 2" if significant else "No significant difference between models"
            }
        }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate models using LM-as-judge paradigm')
    parser.add_argument('--model1', '-m1', required=True,
                       help='First model to evaluate')
    parser.add_argument('--model2', '-m2', required=True,
                       help='Second model to evaluate')
    parser.add_argument('--judge', '-j', required=True,
                       help='Judge model for comparison')
    parser.add_argument('--conversations', '-c', 
                       default='data/filtered_conversations.json',
                       help='Path to conversations file (default: data/filtered_conversations.json)')
    parser.add_argument('--test-size', '-n', type=int, default=1000,
                       help='Number of conversations to use for testing (default: 1000)')
    parser.add_argument('--delay', '-d', type=float, default=0.5,
                       help='Delay between API calls in seconds (default: 0.5)')
    parser.add_argument('--output', '-o',
                       help='Output file path (default: auto-generated in results/)')
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv('TOGETHER_API_KEY'):
        print("Error: TOGETHER_API_KEY environment variable not set.")
        print("Please set it with: export TOGETHER_API_KEY='your-api-key-here'")
        return 1
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Generate output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model1_short = args.model1.split('/')[-1].replace('-', '_')
        model2_short = args.model2.split('/')[-1].replace('-', '_')
        args.output = f"results/evaluation_{model1_short}_vs_{model2_short}_{timestamp}.json"
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Load test set
        test_conversations = evaluator.load_test_set(args.conversations, args.test_size)
        
        if not test_conversations:
            print("Error: No valid conversations found for testing")
            return 1
        
        # Run evaluation
        results = evaluator.run_evaluation(
            test_conversations, 
            args.model1, 
            args.model2, 
            args.judge,
            args.delay
        )
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Print summary
        print(f"\n=== Evaluation Complete ===")
        print(f"Results saved to: {args.output}")
        print(f"\nSummary:")
        print(f"  {args.model1}: {results['summary']['model1_wins']} wins ({results['statistics']['model1_win_rate']:.1%})")
        print(f"  {args.model2}: {results['summary']['model2_wins']} wins ({results['statistics']['model2_win_rate']:.1%})")
        print(f"  Ties: {results['summary']['ties']} ({results['statistics']['tie_rate']:.1%})")
        print(f"  Success rate: {results['summary']['success_rate']:.1%}")
        
        stats_info = results['statistics']['statistical_test']
        print(f"\nStatistical test: {stats_info['interpretation']}")
        print(f"  P-value: {stats_info['p_value']:.4f}")
        print(f"  Significant: {stats_info['significant_at_0.05']}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
