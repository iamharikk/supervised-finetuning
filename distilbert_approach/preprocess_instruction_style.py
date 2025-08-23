import json
import re
from typing import List, Dict

def create_instruction_style_dataset(input_file: str, output_file: str):
    """Convert Q&A pairs to instruction-style format for fine-tuning."""
    
    # Load original Q&A data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    instruction_data = []
    
    # Base instruction templates
    base_instruction = "You are a financial analyst AI assistant. Answer the following question about TCS's financial data accurately and concisely based on the provided information."
    
    for item in data:
        question = item['question']
        answer = item['answer']
        context = item['context']
        
        # Create instruction-style format
        instruction_example = {
            "instruction": base_instruction,
            "input": f"Context: {context}\n\nQuestion: {question}",
            "output": answer
        }
        
        instruction_data.append(instruction_example)
        
        # Add variations for better training
        # Variation 1: Direct question format
        direct_instruction = {
            "instruction": "Answer the financial question based on TCS data.",
            "input": question,
            "output": answer
        }
        instruction_data.append(direct_instruction)
        
        # Variation 2: Conversational format
        conversational_instruction = {
            "instruction": "You are helping analyze TCS financial performance. Provide accurate answers.",
            "input": f"Based on TCS's financial data: {question}",
            "output": answer
        }
        instruction_data.append(conversational_instruction)
    
    # Save instruction-style dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(instruction_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {len(instruction_data)} instruction-style examples")
    print(f"Saved to {output_file}")
    
    return instruction_data

def create_alpaca_style_dataset(input_file: str, output_file: str):
    """Create Alpaca-style instruction dataset."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    alpaca_data = []
    
    for item in data:
        question = item['question']
        answer = item['answer']
        
        # Alpaca format
        alpaca_example = {
            "instruction": "Answer the following question about TCS financial data.",
            "input": question,
            "output": answer
        }
        
        alpaca_data.append(alpaca_example)
        
        # Add calculation-focused examples for growth questions
        if "growth" in question.lower() or "change" in question.lower():
            calc_example = {
                "instruction": "Calculate and explain the financial metric requested.",
                "input": f"Calculate: {question}",
                "output": f"Based on TCS financial data: {answer}"
            }
            alpaca_data.append(calc_example)
    
    # Save Alpaca-style dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {len(alpaca_data)} Alpaca-style examples")
    print(f"Saved to {output_file}")
    
    return alpaca_data

def main():
    """Main function to create instruction-style datasets."""
    
    input_file = '../data/distilbert_simple_format.json'
    
    # Create different instruction formats
    print("Creating instruction-style datasets...")
    
    # Standard instruction format
    instruction_data = create_instruction_style_dataset(
        input_file, 
        '../data/instruction_style_dataset.json'
    )
    
    # Alpaca format
    alpaca_data = create_alpaca_style_dataset(
        input_file,
        '../data/alpaca_style_dataset.json'
    )
    
    # Show examples
    print("\nExample instruction-style format:")
    print(json.dumps(instruction_data[0], indent=2))
    
    print("\nExample Alpaca-style format:")
    print(json.dumps(alpaca_data[0], indent=2))

if __name__ == "__main__":
    main()