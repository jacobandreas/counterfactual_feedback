#!/usr/bin/env python3
"""
Script to process filtered conversations and improve assistant responses
based on negative user feedback using the Together API.

This script:
1. Loads conversations from filtered_conversations_ur235.json
2. For UR5 (positive feedback) conversations: keeps them unchanged
3. For UR2/UR3 (negative feedback) conversations: uses Together API to improve the last assistant response
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
from together import Together
import argparse
from datetime import datetime

class ConversationProcessor:
    """Handles processing of conversations with feedback."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", method: str = "feedback"):
        """
        Initialize the processor with Together API.
        
        Args:
            api_key: Together API key (if None, will look for TOGETHER_API_KEY env var)
            model_name: The model to use for improving responses
            method: Processing method - "feedback", "alternative", or "positive_only"
        """
        if api_key is None:
            api_key = os.getenv('TOGETHER_API_KEY')
        
        if not api_key:
            raise ValueError("Together API key is required. Set TOGETHER_API_KEY environment variable or pass api_key parameter.")
        
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        self.method = method
        self.stats = {
            'total_processed': 0,
            'ur5_unchanged': 0,
            'ur2_ur3_improved': 0,
            'positive_only_extracted': 0,
            'api_errors': 0,
            'skipped': 0
        }
    
    def create_improvement_prompt(self, conversation_context: List[Dict[str, str]]) -> str:
        """
        Create a prompt for the LM to improve the assistant's response based on user feedback.
        
        Args:
            conversation_context: The conversation history
            
        Returns:
            The prompt string for the improvement model
        """
        # Build conversation history (all but the last user message which is feedback)
        conversation_history = ""
        feedback_message = conversation_context[-1]['content']
        
        for i, turn in enumerate(conversation_context[:-1]):
            role = turn['role'].upper()
            content = turn['content']
            conversation_history += f"{role}: {content}\n\n"
        
        prompt = f"""You are an AI assistant tasked with improving another AI assistant's response based on user feedback.

Here is the conversation history:

{conversation_history}

The user then provided this feedback about the assistant's last response:

USER FEEDBACK: {feedback_message}

Your task is to rewrite the assistant's last response to address the user's feedback. The feedback indicates that the original response was inadequate, incorrect, or unhelpful.

Guidelines:
1. If the feedback indicates the response was factually incorrect, provide a corrected response or say "I don't know" if uncertain
2. If the feedback asks for more information or clarification, provide a more comprehensive response
3. If the feedback indicates the response was unhelpful, rewrite it to be more useful
4. If the feedback suggests the assistant misunderstood the question, provide a response that addresses what the user actually wanted
5. Maintain the same tone and style as the original assistant
6. Keep the response concise but complete

It is very important to *rewrite* the last turn of the conversation, rather than responding to the user's feedback. Your output should be an appropriate response to the user's last message: 

USER: {conversation_context[-3]['content']}

while incorporating feedback.

Please provide ONLY the improved assistant response, without any meta-commentary or explanation. Some inputs may have been incorrectly included so if you see a user message that doesn't obviously provide actionable feedback, please output N/A; otherwise output the improved response."""

        return prompt
    
    def improve_assistant_response(self, conversation_context: List[Dict[str, str]]) -> Optional[str]:
        """
        Use Together API to improve the assistant's response based on feedback.
        
        Args:
            conversation_context: The conversation history
            
        Returns:
            Improved response string or None if failed
        """
        try:
            prompt = self.create_improvement_prompt(conversation_context)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1,  # Low temperature for more consistent improvements
                top_p=0.9,
                repetition_penalty=1.0,
            )
            
            improved_response = response.choices[0].message.content.strip()
            if improved_response.lower() == "n/a":
                return None
            return improved_response
            
        except Exception as e:
            print(f"Error calling Together API: {e}")
            self.stats['api_errors'] += 1
            return None
    
    def create_alternative_prompt(self, conversation_context: List[Dict[str, str]]) -> str:
        """
        Create a prompt for generating an alternative assistant response without using feedback.
        
        Args:
            conversation_context: The conversation history (excluding feedback)
            
        Returns:
            The prompt string for alternative generation
        """
        # Build conversation history up to the point where we need a response (excluding feedback)
        conversation_history = ""
        
        # Find the user message that the assistant should respond to
        # (second-to-last in the original, or last after removing feedback)
        response_target_idx = -3  # User message before assistant response before feedback
        if len(conversation_context) > abs(response_target_idx):
            target_user_message = conversation_context[response_target_idx]['content']
        else:
            # Fallback: use the first user message if conversation is too short
            target_user_message = conversation_context[0]['content']
        
        # Build context up to (but not including) the assistant response we're replacing
        for i, turn in enumerate(conversation_context):
            # Stop before the assistant response we want to replace
            if i >= len(conversation_context) - 2:  # Stop before last assistant + feedback
                break
            role = turn['role'].upper()
            content = turn['content']
            conversation_history += f"{role}: {content}\n\n"
        
        prompt = f"""You are an AI assistant. Please provide a helpful, accurate, and appropriate response to the user's message.

Here is the conversation context:

{conversation_history}

The user's message that needs a response:
USER: {target_user_message}

Please provide a clear, helpful, and accurate response. Focus on being informative and addressing what the user is asking for.

Respond as the assistant would, without any meta-commentary or explanation. Provide ONLY the assistant's response:"""

        return prompt
    
    def generate_alternative_response(self, conversation_context: List[Dict[str, str]]) -> Optional[str]:
        """
        Generate an alternative assistant response without using feedback.
        
        Args:
            conversation_context: The conversation history
            
        Returns:
            Alternative response string or None if failed
        """
        try:
            prompt = self.create_alternative_prompt(conversation_context)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3,  # Slightly higher temperature for variation
                top_p=0.9,
                repetition_penalty=1.0,
            )
            
            alternative_response = response.choices[0].message.content.strip()
            return alternative_response
            
        except Exception as e:
            print(f"Error calling Together API: {e}")
            self.stats['api_errors'] += 1
            return None
    
    def extract_positive_only(self, conversation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract only positive interactions by keeping conversations up to points with positive feedback.
        Removes any turns that led to negative feedback.
        
        Args:
            conversation: The conversation dictionary
            
        Returns:
            Conversation with only positive interaction chains, or None if no positive chains found
        """
        conversation_context = conversation.get('conversation_context', [])
        if len(conversation_context) < 2:
            return None
            
        # For positive_only method, we only keep conversations that end with positive feedback (UR5)
        # and remove the feedback turn itself, keeping only the positive interaction chain
        if conversation.get('category') == 'UR5':
            # Create a copy without the last turn (which is the positive feedback)
            processed_conversation = conversation.copy()
            processed_conversation['conversation_context'] = conversation_context[:-1].copy()
            processed_conversation['extraction_method'] = 'positive_only'
            processed_conversation['original_category'] = conversation.get('category')
            processed_conversation['extraction_timestamp'] = datetime.now().isoformat()
            return processed_conversation
        
        # For UR2/UR3 conversations, we discard them entirely in positive_only mode
        return None
    
    def process_conversation(self, conversation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single conversation based on its category.
        
        Args:
            conversation: The conversation dictionary
            
        Returns:
            Processed conversation (modified or unchanged)
        """
        category = conversation.get('category', '')
        conversation_context = conversation.get('conversation_context', [])
        
        # Make a copy to avoid modifying the original
        processed_conversation = conversation.copy()
        
        # Handle positive_only method
        if self.method == "positive_only":
            extracted = self.extract_positive_only(conversation)
            if extracted:
                self.stats['positive_only_extracted'] += 1
                return extracted
            else:
                # For positive_only method, skip conversations that don't have positive feedback
                self.stats['skipped'] += 1
                # Return None to indicate this conversation should be filtered out
                return None
        
        # Handle feedback and alternative methods
        if category == 'UR5':
            # Positive feedback - no changes needed
            self.stats['ur5_unchanged'] += 1
            return processed_conversation
        
        elif category in ['UR2', 'UR3']:
            # Negative feedback - improve the assistant response
            if len(conversation_context) < 2:
                print(f"Warning: Conversation {conversation.get('conversation_id', 'unknown')} too short to process")
                self.stats['skipped'] += 1
                return processed_conversation
            
            # Find the last assistant response that needs to be improved
            last_assistant_idx = None
            for i in range(len(conversation_context) - 2, -1, -1):  # Start from second-to-last
                if conversation_context[i]['role'] == 'assistant':
                    last_assistant_idx = i
                    break
            
            if last_assistant_idx is None:
                print(f"Warning: No assistant response found to improve in conversation {conversation.get('conversation_id', 'unknown')}")
                self.stats['skipped'] += 1
                return processed_conversation
            
            # Store the original response BEFORE making any changes
            original_response = conversation_context[last_assistant_idx]['content']
            
            # Generate new response based on the chosen method
            if self.method == "feedback":
                new_response = self.improve_assistant_response(conversation_context)
                method_used = "feedback_based"
            elif self.method == "alternative":
                new_response = self.generate_alternative_response(conversation_context)
                method_used = "alternative_generation"
            else:  # positive_only - shouldn't reach here for UR2/UR3
                self.stats['skipped'] += 1
                return processed_conversation
            
            if new_response:
                # Create a deep copy of the conversation context to avoid modifying the original
                processed_conversation['conversation_context'] = [turn.copy() for turn in conversation_context]
                
                # Replace the assistant's response with the new version
                processed_conversation['conversation_context'][last_assistant_idx]['content'] = new_response
                
                # Add metadata about the improvement
                processed_conversation['improved'] = True
                processed_conversation['generation_method'] = method_used
                processed_conversation['original_assistant_response'] = original_response
                processed_conversation['improvement_timestamp'] = datetime.now().isoformat()
                
                self.stats['ur2_ur3_improved'] += 1
            else:
                self.stats['skipped'] += 1
            
            return processed_conversation
        
        else:
            print(f"Warning: Unknown category '{category}' in conversation {conversation.get('conversation_id', 'unknown')}")
            self.stats['skipped'] += 1
            return processed_conversation
    
    def process_conversations(self, conversations: List[Dict[str, Any]], 
                            max_conversations: Optional[int] = None,
                            delay_between_calls: float = 0.1) -> List[Dict[str, Any]]:
        """
        Process all conversations.
        
        Args:
            conversations: List of conversation dictionaries
            max_conversations: Maximum number to process (for testing)
            delay_between_calls: Delay between API calls to avoid rate limits
            
        Returns:
            List of processed conversations
        """
        processed_conversations = []
        total = min(len(conversations), max_conversations) if max_conversations else len(conversations)
        
        print(f"Processing {total} conversations...")
        
        for i, conversation in enumerate(conversations[:max_conversations] if max_conversations else conversations):
            if i % 100 == 0:
                print(f"Progress: {i}/{total} conversations processed")
            
            processed_conv = self.process_conversation(conversation)
            
            # Skip conversations that were filtered out (returned None)
            if processed_conv is None:
                continue
                
            # Remove the last user feedback turn for feedback and alternative methods
            if self.method in ["feedback", "alternative"] and processed_conv.get('conversation_context'):
                if processed_conv['conversation_context'][-1].get('role') == 'user':
                    del processed_conv['conversation_context'][-1]
            
            processed_conversations.append(processed_conv)
            
            self.stats['total_processed'] += 1
            
            # Add delay between API calls for rate limiting
            if conversation.get('category') in ['UR2', 'UR3'] and delay_between_calls > 0:
                time.sleep(delay_between_calls)
        
        return processed_conversations
    
    def print_stats(self):
        """Print processing statistics."""
        method_names = {
            "feedback": "feedback-based improvement",
            "alternative": "alternative generation", 
            "positive_only": "positive interactions extraction"
        }
        method_name = method_names.get(self.method, self.method)
        
        print(f"\n=== Processing Statistics ({method_name}) ===")
        print(f"Total processed: {self.stats['total_processed']}")
        print(f"UR5 (positive feedback, unchanged): {self.stats['ur5_unchanged']}")
        
        if self.method == "feedback":
            print(f"UR2/UR3 (negative feedback, improved): {self.stats['ur2_ur3_improved']}")
        elif self.method == "alternative":
            print(f"UR2/UR3 (alternative responses generated): {self.stats['ur2_ur3_improved']}")
        elif self.method == "positive_only":
            print(f"Positive-only conversations extracted: {self.stats['positive_only_extracted']}")
        
        print(f"API errors: {self.stats['api_errors']}")
        print(f"Skipped: {self.stats['skipped']}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Process conversations with user feedback')
    parser.add_argument('--input', '-i', 
                       default='data/filtered_conversations_ur235.json',
                       help='Input JSON file with filtered conversations')
    parser.add_argument('--output', '-o',
                       default='data/improved_conversations.json',
                       help='Output JSON file for processed conversations')
    parser.add_argument('--max-conversations', '-n', type=int,
                       help='Maximum number of conversations to process (for testing)')
    parser.add_argument('--delay', '-d', type=float, default=0.1,
                       help='Delay between API calls in seconds (default: 0.1)')
    parser.add_argument('--model', '-m', 
                       default='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
                       help='Together API model to use for improvements')
    parser.add_argument('--method', choices=['feedback', 'alternative', 'positive_only'], 
                       default='feedback',
                       help='Generation method: "feedback" uses user feedback to improve responses, "alternative" generates new responses without feedback, "positive_only" extracts only positive interactions (default: feedback)')
    
    args = parser.parse_args()
    
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
        print(f"Loaded {len(conversations)} conversations")
    except FileNotFoundError:
        print(f"Error: Could not find input file {args.input}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}")
        return 1
    
    # Initialize processor
    try:
        processor = ConversationProcessor(
            model_name=args.model, 
            method=args.method
        )
        print(f"Using {args.method} generation method")
    except ValueError as e:
        print(f"Error initializing processor: {e}")
        return 1
    
    # Process conversations
    processed_conversations = processor.process_conversations(
        conversations, 
        max_conversations=args.max_conversations,
        delay_between_calls=args.delay
    )
    
    # Save results
    print(f"\nSaving processed conversations to {args.output}...")
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(processed_conversations, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(processed_conversations)} processed conversations")
    except Exception as e:
        print(f"Error saving results: {e}")
        return 1
    
    # Print statistics
    processor.print_stats()
    
    print(f"\nProcessing complete!")
    print(f"Results saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())
