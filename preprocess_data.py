import re
import json
from typing import List, Tuple

def parse_qa_pairs(file_path: str) -> List[Tuple[str, str]]:
    """Parse Q&A pairs from the text file."""
    qa_pairs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract Q&A pairs using regex
    pattern = r'Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\n\n|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        if question and answer:
            qa_pairs.append((question, answer))
    
    return qa_pairs

def save_training_data_json(qa_pairs: List[Tuple[str, str]], output_path: str):
    """Save Q&A pairs as JSON with instruction-output format."""
    training_data = []
    
    for question, answer in qa_pairs:
        training_data.append({
            "instruction": f"Answer the following financial question about TCS: {question}",
            "output": answer
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

def main():
    input_file = "data/q-and-a/fin-data-qa-pairs.txt"
    output_file = "data/training_data.json"
    
    print("Parsing Q&A pairs...")
    qa_pairs = parse_qa_pairs(input_file)
    print(f"Found {len(qa_pairs)} Q&A pairs")
    
    print("Saving training data as JSON...")
    save_training_data_json(qa_pairs, output_file)
    print(f"JSON training data saved to {output_file}")
    
    # Print sample
    print("\nSample JSON formatted data:")
    sample_entry = {
        "instruction": f"Answer the following financial question about TCS: {qa_pairs[0][0]}",
        "output": qa_pairs[0][1]
    }
    print(f"1. {sample_entry}")
    
    if len(qa_pairs) > 1:
        sample_entry2 = {
            "instruction": f"Answer the following financial question about TCS: {qa_pairs[1][0]}",
            "output": qa_pairs[1][1]
        }
        print(f"2. {sample_entry2}")

if __name__ == "__main__":
    main()