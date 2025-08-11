#!/usr/bin/env python3
"""
Evaluation script for comparing models using LM-as-judge paradigm with support for HuggingFace and Together API models.

This script supports two modes:

1. Model Comparison (original functionality):
   - Takes the last N conversations from data/conversations.json as test set
   - Generates responses from two test models for each conversation context
   - Uses a judge model to compare the responses and pick the winner
   - Computes statistics and saves results to JSON

2. Feedback Dataset Evaluation (new functionality):
   - Takes a feedback dataset file (e.g., data/feedback_10k_8b.json)
   - Compares the last assistant turn in conversation_context vs original_assistant_response
   - Uses a judge model to determine which response is better
   - Skips entries missing original_assistant_response

Model Specifications:
   - HuggingFace models: Use "hf:" prefix, e.g., "hf:meta-llama/Llama-2-7b-chat-hf"
   - Together API models: Use "together:" prefix, e.g., "together:meta-llama/Llama-2-7b-chat-hf"
   - For backward compatibility, models without prefix default to Together API

Usage:
    # Model comparison mode with HuggingFace models
    python evaluate.py compare --model1 hf:meta-llama/Llama-2-7b-chat-hf --model2 hf:meta-llama/Llama-2-13b-chat-hf --judge together:meta-llama/Llama-2-70b-chat-hf
    
    # Model comparison mode with Together API models
    python evaluate.py compare --model1 together:meta-llama/Llama-2-7b-chat-hf --model2 together:meta-llama/Llama-2-13b-chat-hf --judge together:meta-llama/Llama-2-70b-chat-hf
    
    # Feedback dataset evaluation mode with HuggingFace judge
    python evaluate.py feedback --dataset data/feedback_10k_8b.json --judge hf:meta-llama/Llama-2-70b-chat-hf
    
    # Mixed setup (HF model vs Together model, judged by Together)
    python evaluate.py compare --model1 hf:meta-llama/Llama-2-7b-chat-hf --model2 together:meta-llama/Llama-2-13b-chat-hf --judge together:meta-llama/Llama-2-70b-chat-hf

Requirements:
    - For Together API: Set TOGETHER_API_KEY environment variable
    - For HuggingFace models: Install transformers and torch (pip install transformers torch)
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

# Optional HuggingFace imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class ModelEvaluator:
    """Handles model evaluation using LM-as-judge paradigm with support for Together API and HuggingFace models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the evaluator with Together API and HuggingFace support."""
        # Together API setup
        if api_key is None:
            api_key = os.getenv('TOGETHER_API_KEY')
        
        self.together_client = None
        if api_key:
            self.together_client = Together(api_key=api_key)
        
        # HuggingFace models cache
        self.hf_models = {}  # Cache for loaded HF models
        self.hf_tokenizers = {}  # Cache for loaded HF tokenizers
        
        self.results = {
            'model1_wins': 0,
            'model2_wins': 0,
            'ties': 0,
            'errors': 0,
            'total_evaluations': 0,
            'evaluations': []
        }
    
    def parse_model_name(self, model_spec: str) -> Tuple[str, str]:
        """
        Parse model specification into source and model name.
        
        Args:
            model_spec: Model specification like "hf:meta-llama/Llama-2-7b-chat-hf" or "together:meta-llama/Llama-2-7b-chat-hf"
            
        Returns:
            Tuple of (source, model_name)
        """
        if ':' in model_spec:
            source, model_name = model_spec.split(':', 1)
            return source.lower(), model_name
        else:
            # Default to together for backward compatibility
            return 'together', model_spec
    
    def load_hf_model(self, model_name: str) -> Tuple[Any, Any]:
        """
        Load HuggingFace model and tokenizer with caching.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if not HF_AVAILABLE:
            raise ValueError("HuggingFace transformers not available. Install with: pip install transformers torch")
        
        if model_name not in self.hf_models:
            print(f"Loading HuggingFace model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            self.hf_models[model_name] = model
            self.hf_tokenizers[model_name] = tokenizer
            print(f"Loaded HuggingFace model: {model_name}")
        
        return self.hf_models[model_name], self.hf_tokenizers[model_name]
    
    def generate_hf_response(self, model_name: str, context: List[Dict[str, str]], max_retries: int = 3) -> Optional[str]:
        """
        Generate a response using a HuggingFace model.
        
        Uses the model's chat template if available for proper formatting.
        Ensures only one assistant turn is generated by stopping at dialog markers.
        
        Args:
            model_name: HuggingFace model name
            context: Conversation context (list of messages)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated response or None if failed
        """
        try:
            model, tokenizer = self.load_hf_model(model_name)
            
            # Use the model's chat template if available
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                # Apply chat template
                prompt = tokenizer.apply_chat_template(
                    context, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback to manual formatting if no chat template
                print(f"Warning: No chat template found for {model_name}, using fallback formatting")
                prompt = ""
                for turn in context:
                    role = turn['role']
                    content = turn['content']
                    if role == 'user':
                        prompt += f"User: {content}\n"
                    elif role == 'assistant':
                        prompt += f"Assistant: {content}\n"
                prompt += "Assistant: "
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move to device if using GPU
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode only the new tokens (the generated response)
            response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Clean up the response - stop at first occurrence of dialog markers
            response = response.strip()
            
            # Split on common dialog turn indicators and take only the first part
            # Include both ChatML-style and other common chat template markers
            stop_markers = ["\nUser:", "\nAssistant:", "\n<|user|>", "\n<|assistant|>"]
            for marker in stop_markers:
                if marker in response:
                    response = response.split(marker)[0].strip()
                    break
            
            return response if response else None
            
        except Exception as e:
            print(f"Error generating HuggingFace response with {model_name}: {e}")
            return None
    
    def generate_together_response(self, model_name: str, context: List[Dict[str, str]], max_retries: int = 3) -> Optional[str]:
        """
        Generate a response using Together API.
        
        Args:
            model_name: Together model name
            context: Conversation context (list of messages)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated response or None if failed
        """
        if not self.together_client:
            print("Together API client not available. Set TOGETHER_API_KEY environment variable.")
            return None
        
        for attempt in range(max_retries):
            try:
                response = self.together_client.chat.completions.create(
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
                print(f"Error generating Together response with {model_name} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None
        return None
    
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
    
    def load_feedback_dataset(self, path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load feedback dataset for comparing current vs original responses.
        
        Args:
            path: Path to feedback dataset file
            limit: Maximum number of conversations to load (optional)
        """
        
        print(f"Loading feedback dataset from {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            all_conversations = json.load(f)
        
        # Limit if specified
        if limit:
            all_conversations = all_conversations[:limit]
        
        print(f"Loaded {len(all_conversations)} conversations from feedback dataset")
        
        # Filter to only use conversations that have original_assistant_response
        valid_conversations = []
        for conv in all_conversations:
            if 'original_assistant_response' in conv and conv['original_assistant_response']:
                context = conv.get('conversation_context', [])
                # Make sure the conversation context has at least one turn and ends with assistant
                if len(context) >= 1 and context[-1].get('role') == 'assistant':
                    valid_conversations.append(conv)
        
        print(f"Found {len(valid_conversations)} valid conversations with original responses to compare")
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
    
    def prepare_feedback_conversation_context(self, conversation: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Prepare conversation context from feedback dataset by removing the last assistant turn.
        
        Args:
            conversation: Conversation dictionary from feedback dataset
            
        Returns:
            List of messages without the last assistant response
        """
        context = conversation.get('conversation_context', [])
        
        # Remove the last turn (assistant response) to get the context for comparison
        if len(context) >= 1 and context[-1].get('role') == 'assistant':
            context_without_last = context[:-1]
        else:
            context_without_last = context
        
        # Convert to the expected format
        prepared_context = []
        for turn in context_without_last:
            prepared_context.append({
                'role': turn.get('role', ''),
                'content': turn.get('content', '')
            })
        
        return prepared_context
    
    def get_current_response(self, conversation: Dict[str, Any]) -> str:
        """Get the current assistant response from the feedback dataset (last turn)."""
        context = conversation.get('conversation_context', [])
        # The current assistant response should be the last turn
        if len(context) >= 1 and context[-1].get('role') == 'assistant':
            return context[-1].get('content', '')
        return ""
    
    def get_original_response(self, conversation: Dict[str, Any]) -> str:
        """Get the original assistant response from the conversation (second-to-last turn)."""
        context = conversation.get('conversation_context', [])
        # The assistant response should be the second-to-last turn
        if len(context) >= 2 and context[-2].get('role') == 'assistant':
            return context[-2].get('content', '')
        return ""
    
    def generate_model_response(self, model_spec: str, context: List[Dict[str, str]], max_retries: int = 3) -> Optional[str]:
        """
        Generate a response from the specified model given the conversation context.
        
        Args:
            model_spec: Model specification (e.g., "hf:meta-llama/Llama-2-7b-chat-hf" or "together:meta-llama/Llama-2-7b-chat-hf")
            context: Conversation context (list of messages)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated response or None if failed
        """
        source, model_name = self.parse_model_name(model_spec)
        
        if source == 'hf':
            return self.generate_hf_response(model_name, context, max_retries)
        elif source == 'together':
            return self.generate_together_response(model_name, context, max_retries)
        else:
            print(f"Unknown model source: {source}. Use 'hf:' or 'together:' prefix.")
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
    '"[[A]]" if assistant A is best, "[[B]]" if assistant B is best, or "[[TIE]]" if both assistants are equally good. Your response should be no longer than about 50 words.\n'
    f"\nConversation Context:\n{context_text}\n"
    f"Response A: {response1}\n"
    f"Response B: {response2}\n"
   )

        return [{"role": "user", "content": judge_prompt}]
    
    def judge_responses(self, judge_model_spec: str, context: List[Dict[str, str]], 
                       response1: str, response2: str, original: str, 
                       max_retries: int = 3) -> Optional[str]:
        """
        Use the judge model to compare two responses.
        
        Args:
            judge_model_spec: Judge model specification (e.g., "hf:..." or "together:...")
            context: Conversation context
            response1: Response from model 1
            response2: Response from model 2
            original: Original response from conversation
            max_retries: Maximum retry attempts
            
        Returns:
            Judge decision: "A", "B", "TIE", or None if failed
        """
        judge_messages = self.create_judge_prompt(context, response1, response2, original)
        
        # Use the unified response generation for the judge
        for attempt in range(max_retries):
            try:
                # Generate judgment using the unified method
                response_content = self.generate_model_response(judge_model_spec, judge_messages, max_retries=1)
                
                if response_content:
                    judgment = str(response_content).strip().upper()
                    
                    # Parse the judgment
                    if "[[A]]" in judgment and "[[B]]" not in judgment:
                        return "A"
                    elif "[[B]]" in judgment and "[[A]]" not in judgment:
                        return "B"
                    elif "[[TIE]]" in judgment:
                        return "TIE"
                    else:
                        print(f"Ambiguous judgment: {judgment}")
                        if attempt == max_retries - 1:
                            return None
                else:
                    print(f"No response from judge model (attempt {attempt + 1})")
                    
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
    
    def evaluate_feedback_conversation(self, conversation: Dict[str, Any], judge_model: str, conv_idx: int) -> Dict[str, Any]:
        """
        Evaluate a single conversation from feedback dataset by comparing current vs original response.
        
        Args:
            conversation: Conversation from feedback dataset
            judge_model: Name of judge model
            conv_idx: Index of conversation for logging
            
        Returns:
            Evaluation result dictionary
        """
        print(f"Evaluating feedback conversation {conv_idx + 1}...")
        
        # Get the context (without the current assistant response)
        context = self.prepare_feedback_conversation_context(conversation)
        current_response = self.get_current_response(conversation)
        original_response = conversation.get('original_assistant_response', '')
        
        if not context:
            return {
                'conversation_id': conversation.get('conversation_id', f'conv_{conv_idx}'),
                'status': 'error',
                'error': 'Empty context after removing current assistant response'
            }
        
        if not current_response:
            return {
                'conversation_id': conversation.get('conversation_id', f'conv_{conv_idx}'),
                'status': 'error',
                'error': 'No current assistant response found'
            }
        
        if not original_response:
            return {
                'conversation_id': conversation.get('conversation_id', f'conv_{conv_idx}'),
                'status': 'error',
                'error': 'No original assistant response found'
            }
        
        # Randomize order for fair judging
        if random.random() < 0.5:
            # Current as A, Original as B
            judgment = self.judge_responses(judge_model, context, current_response, original_response, "")
            order = "normal"  # Current=A, Original=B
        else:
            # Original as A, Current as B (reversed)
            judgment = self.judge_responses(judge_model, context, original_response, current_response, "")
            order = "reversed"  # Original=A, Current=B
        
        if judgment is None:
            return {
                'conversation_id': conversation.get('conversation_id', f'conv_{conv_idx}'),
                'status': 'error',
                'error': 'Failed to get judgment from judge model'
            }
        
        # Convert judgment based on order
        if order == "reversed":
            if judgment == "A":
                judgment = "B"  # Original won, but we want Current perspective
            elif judgment == "B":
                judgment = "A"  # Current won
            # TIE remains TIE
        
        # Map judgment to winner (from perspective of current vs original)
        if judgment == "A":
            winner = "current"  # Current response won
        elif judgment == "B":
            winner = "original"  # Original response won
        else:  # TIE
            winner = "tie"
        
        return {
            'conversation_id': conversation.get('conversation_id', f'conv_{conv_idx}'),
            'status': 'success',
            'context': context,
            'current_response': current_response,
            'original_response': original_response,
            'judgment': judgment,
            'winner': winner,
            'order': order,
            'model': conversation.get('model', 'unknown'),
            'category': conversation.get('category', 'unknown'),
            'label': conversation.get('label', None)
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
    
    def run_feedback_evaluation(self, test_conversations: List[Dict[str, Any]], 
                               judge_model: str, delay: float = 0.5) -> Dict[str, Any]:
        """
        Run evaluation on feedback dataset comparing current vs original responses.
        
        Args:
            test_conversations: List of conversations from feedback dataset
            judge_model: Name of judge model
            delay: Delay between API calls
            
        Returns:
            Complete evaluation results
        """
        print(f"\nStarting feedback evaluation...")
        print(f"Judge model: {judge_model}")
        print(f"Total conversations: {len(test_conversations)}")
        
        results = []
        
        for i, conversation in enumerate(test_conversations):
            result = self.evaluate_feedback_conversation(conversation, judge_model, i)
            results.append(result)
            
            # Add delay between requests
            if i < len(test_conversations) - 1:
                time.sleep(delay)
        
        # Calculate summary statistics
        feedback_results = {
            'current_wins': 0,
            'original_wins': 0,
            'ties': 0,
            'errors': 0,
            'total_evaluations': len(results)
        }
        
        for result in results:
            if result['status'] == 'success':
                winner = result['winner']
                if winner == 'current':
                    feedback_results['current_wins'] += 1
                elif winner == 'original':
                    feedback_results['original_wins'] += 1
                elif winner == 'tie':
                    feedback_results['ties'] += 1
            else:
                feedback_results['errors'] += 1
        
        successful_evaluations = feedback_results['total_evaluations'] - feedback_results['errors']
        success_rate = successful_evaluations / feedback_results['total_evaluations'] if feedback_results['total_evaluations'] > 0 else 0
        
        print(f"\nFeedback Evaluation Summary:")
        print(f"  Total evaluations: {feedback_results['total_evaluations']}")
        print(f"  Successful: {successful_evaluations} ({success_rate:.1%})")
        print(f"  Errors: {feedback_results['errors']}")
        print(f"  Current response wins: {feedback_results['current_wins']}")
        print(f"  Original response wins: {feedback_results['original_wins']}")
        print(f"  Ties: {feedback_results['ties']}")
        
        # Calculate win rates and confidence intervals for feedback evaluation
        non_tie_total = feedback_results['current_wins'] + feedback_results['original_wins']
        if non_tie_total > 0:
            current_win_rate = feedback_results['current_wins'] / non_tie_total
            original_win_rate = feedback_results['original_wins'] / non_tie_total
            
            # 95% confidence intervals using Wilson score interval
            def wilson_confidence_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
                """Calculate Wilson score confidence interval."""
                if total == 0:
                    return (0.0, 0.0)
                
                z = 1.96 if confidence == 0.95 else 1.645  # 95% or 90% confidence
                p = successes / total
                n = total
                
                denominator = 1 + (z**2 / n)
                centre = (p + (z**2 / (2*n))) / denominator
                margin = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
                
                return (max(0.0, centre - margin), min(1.0, centre + margin))
            
            current_ci = wilson_confidence_interval(feedback_results['current_wins'], non_tie_total)
            original_ci = wilson_confidence_interval(feedback_results['original_wins'], non_tie_total)
            
            # Statistical significance test - need to import scipy
            try:
                from scipy.stats import binomtest
                binom_result = binomtest(feedback_results['current_wins'], non_tie_total, 0.5)
                p_value = binom_result.pvalue
            except ImportError:
                # Fallback to simple calculation if scipy not available
                p_value = 1.0 if feedback_results['current_wins'] == feedback_results['original_wins'] else 0.1
            
            significant = p_value < 0.05
        else:
            current_win_rate = original_win_rate = 0.0
            current_ci = original_ci = (0.0, 0.0)
            p_value = 1.0
            significant = False
        
        tie_rate = feedback_results['ties'] / feedback_results['total_evaluations'] if feedback_results['total_evaluations'] > 0 else 0
        
        # Prepare final results
        final_results = {
            'evaluation_type': 'feedback_comparison',
            'judge_model': judge_model,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'test_conversations': len(test_conversations),
                'delay': delay
            },
            'summary': {
                'total_evaluations': feedback_results['total_evaluations'],
                'successful_evaluations': successful_evaluations,
                'errors': feedback_results['errors'],
                'success_rate': success_rate,
                'current_wins': feedback_results['current_wins'],
                'original_wins': feedback_results['original_wins'],
                'ties': feedback_results['ties']
            },
            'statistics': {
                'current_win_rate': current_win_rate,
                'original_win_rate': original_win_rate,
                'tie_rate': tie_rate,
                'current_confidence_interval_95': current_ci,
                'original_confidence_interval_95': original_ci,
                'non_tie_evaluations': non_tie_total,
                'statistical_test': {
                    'test': 'binomial_test',
                    'null_hypothesis': 'current and original responses are equally good',
                    'p_value': p_value,
                    'significant_at_0.05': significant,
                    'interpretation': f"Current response significantly {'better' if feedback_results['current_wins'] > feedback_results['original_wins'] else 'worse'} than original" if significant else "No significant difference between current and original responses"
                }
            },
            'detailed_results': results
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
    
    # Add mode selection
    subparsers = parser.add_subparsers(dest='mode', help='Evaluation mode')
    
    # Model comparison mode (original functionality)
    model_parser = subparsers.add_parser('compare', help='Compare two models')
    model_parser.add_argument('--model1', '-m1', required=True,
                             help='First model to evaluate (e.g., hf:meta-llama/Llama-2-7b-chat-hf or together:meta-llama/Llama-2-7b-chat-hf)')
    model_parser.add_argument('--model2', '-m2', required=True,
                             help='Second model to evaluate (e.g., hf:meta-llama/Llama-2-13b-chat-hf or together:meta-llama/Llama-2-13b-chat-hf)')
    model_parser.add_argument('--judge', '-j', required=True,
                             help='Judge model for comparison (e.g., hf:meta-llama/Llama-2-70b-chat-hf or together:meta-llama/Llama-2-70b-chat-hf)')
    model_parser.add_argument('--conversations', '-c', 
                             default='data/filtered_conversations.json',
                             help='Path to conversations file (default: data/filtered_conversations.json)')
    model_parser.add_argument('--test-size', '-n', type=int, default=1000,
                             help='Number of conversations to use for testing (default: 1000)')
    model_parser.add_argument('--delay', '-d', type=float, default=0.5,
                             help='Delay between API calls in seconds (default: 0.5)')
    model_parser.add_argument('--output', '-o',
                             help='Output file path (default: auto-generated in results/)')
    
    # Feedback comparison mode (new functionality)
    feedback_parser = subparsers.add_parser('feedback', help='Compare current vs original responses in feedback dataset')
    feedback_parser.add_argument('--dataset', '-d', required=True,
                                help='Path to feedback dataset file (e.g., data/feedback_10k_8b.json)')
    feedback_parser.add_argument('--judge', '-j', required=True,
                                help='Judge model for comparison (e.g., hf:meta-llama/Llama-2-70b-chat-hf or together:meta-llama/Llama-2-70b-chat-hf)')
    feedback_parser.add_argument('--limit', '-l', type=int,
                                help='Maximum number of conversations to evaluate (optional)')
    feedback_parser.add_argument('--delay', type=float, default=0.5,
                                help='Delay between API calls in seconds (default: 0.5)')
    feedback_parser.add_argument('--output', '-o',
                                help='Output file path (default: auto-generated in results/)')
    
    args = parser.parse_args()
    
    # Check if mode was specified
    if not args.mode:
        parser.print_help()
        print("\nError: Please specify evaluation mode (compare or feedback)")
        return 1
    
    # Check if Together API models are being used and warn if API key is missing
    def uses_together_api(models_to_check):
        for model in models_to_check:
            if model:
                source, _ = evaluator_temp.parse_model_name(model)
                if source == 'together':
                    return True
        return False
    
    # Create a temporary evaluator to access parse_model_name
    evaluator_temp = ModelEvaluator()
    
    models_to_check = []
    if args.mode == 'compare':
        models_to_check = [args.model1, args.model2, args.judge]
    elif args.mode == 'feedback':
        models_to_check = [args.judge]
    
    if uses_together_api(models_to_check) and not os.getenv('TOGETHER_API_KEY'):
        print("Warning: Using Together API models but TOGETHER_API_KEY environment variable not set.")
        print("Please set it with: export TOGETHER_API_KEY='your-api-key-here'")
        print("Together API models will fail without the API key.")
    
    # Check HuggingFace availability if needed
    def uses_huggingface(models_to_check):
        for model in models_to_check:
            if model:
                source, _ = evaluator_temp.parse_model_name(model)
                if source == 'hf':
                    return True
        return False
    
    if uses_huggingface(models_to_check) and not HF_AVAILABLE:
        print("Error: Using HuggingFace models but transformers/torch not available.")
        print("Please install with: pip install transformers torch")
        return 1
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator()
        results = None
        
        if args.mode == 'compare':
            # Original model comparison mode
            # Generate output filename if not provided
            if not args.output:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model1_short = args.model1.split('/')[-1].replace('-', '_')
                model2_short = args.model2.split('/')[-1].replace('-', '_')
                args.output = f"results/evaluation_{model1_short}_vs_{model2_short}_{timestamp}.json"
            
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
            
            # Print summary for model comparison
            print(f"\n=== Model Comparison Evaluation Complete ===")
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
            
        elif args.mode == 'feedback':
            # New feedback comparison mode
            # Generate output filename if not provided
            if not args.output:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
                args.output = f"results/feedback_evaluation_{dataset_name}_{timestamp}.json"
            
            # Load feedback dataset
            test_conversations = evaluator.load_feedback_dataset(args.dataset, args.limit)
            
            if not test_conversations:
                print("Error: No valid conversations found for testing")
                return 1
            
            # Run feedback evaluation
            results = evaluator.run_feedback_evaluation(
                test_conversations,
                args.judge,
                args.delay
            )
            
            # Print summary for feedback comparison
            print(f"\n=== Feedback Evaluation Complete ===")
            print(f"Results saved to: {args.output}")
            print(f"\nSummary:")
            print(f"  Current responses: {results['summary']['current_wins']} wins ({results['statistics']['current_win_rate']:.1%})")
            print(f"  Original responses: {results['summary']['original_wins']} wins ({results['statistics']['original_win_rate']:.1%})")
            print(f"  Ties: {results['summary']['ties']} ({results['statistics']['tie_rate']:.1%})")
            print(f"  Success rate: {results['summary']['success_rate']:.1%}")
            
            stats_info = results['statistics']['statistical_test']
            print(f"\nStatistical test: {stats_info['interpretation']}")
            print(f"  P-value: {stats_info['p_value']:.4f}")
            print(f"  Significant: {stats_info['significant_at_0.05']}")
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
