import json

def create_simple_instruction_format():
    """Create simple instruction-output format without context."""
    
    # Load current data
    with open('../data/distilbert_simple_format.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to simple instruction format
    instruction_data = []
    
    for item in data:
        question = item['question']
        answer = item['answer']
        
        # Simple instruction-output format
        instruction_example = {
            "instruction": question,
            "output": answer
        }
        
        instruction_data.append(instruction_example)
    
    # Save simple format
    with open('../data/simple_instruction_format.json', 'w', encoding='utf-8') as f:
        json.dump(instruction_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {len(instruction_data)} simple instruction examples")
    
    # Show example
    print("\nExample format:")
    for i in range(3):
        print(json.dumps(instruction_data[i], indent=2))
        print("-" * 40)
    
    return instruction_data

if __name__ == "__main__":
    create_simple_instruction_format()