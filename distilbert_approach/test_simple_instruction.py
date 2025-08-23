import torch
import json
from finetune_simple_instruction import SimpleInstructionFineTuner

class SimpleInstructionTester:
    """Test class for simple instruction model."""
    
    def __init__(self, model_path='./models/simple_instruction_qa'):
        self.model_path = model_path
        self.trainer = SimpleInstructionFineTuner(output_dir=model_path)
    
    def load_model(self):
        """Load the simple instruction model."""
        return self.trainer.load_model()
    
    def test_sample_questions(self):
        """Test with sample questions."""
        sample_questions = [
            "What was TCS's sales turnover in 2025?",
            "What was TCS's net profit in 2025?",
            "What was TCS's employee cost in 2025?",
            "What was TCS's total income in 2024?",
            "How much was TCS's other income in 2025?",
            "What was TCS's reported net profit in 2024?"
        ]
        
        print("\n" + "="*60)
        print("TESTING SIMPLE INSTRUCTION MODEL (NO CONTEXT)")
        print("="*60)
        
        for i, question in enumerate(sample_questions, 1):
            answer, confidence = self.trainer.predict(question)
            print(f"\n{i}. Instruction: {question}")
            print(f"   Output: {answer}")
            print(f"   Confidence: {confidence:.1%}")
            print("-" * 50)
    
    def compare_with_ground_truth(self, test_data_path='../data/simple_instruction_format.json'):
        """Compare with ground truth."""
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print(f"Test data file not found: {test_data_path}")
            return 0
        
        print("\n" + "="*60)
        print("COMPARING SIMPLE INSTRUCTION MODEL WITH GROUND TRUTH")
        print("="*60)
        
        correct = 0
        total = min(10, len(test_data))
        
        for i, item in enumerate(test_data[:total]):
            instruction = item['instruction']
            expected_output = item['output']
            
            predicted_output, confidence = self.trainer.predict(instruction)
            
            is_correct = expected_output.strip() == predicted_output.strip()
            
            if is_correct:
                correct += 1
            
            status = "[PASS]" if is_correct else "[FAIL]"
            print(f"\n{i+1}. {status} Instruction: {instruction}")
            print(f"   Expected: {expected_output}")
            print(f"   Predicted: {predicted_output} (confidence: {confidence:.1%})")
            print("-" * 50)
        
        accuracy = (correct / total) * 100
        print(f"\nSimple Instruction Model Accuracy: {accuracy:.1f}% ({correct}/{total})")
        return accuracy
    
    def compare_all_approaches(self):
        """Compare all approaches we've built."""
        print("\n" + "="*70)
        print("COMPREHENSIVE COMPARISON: ALL APPROACHES")
        print("="*70)
        
        # Test simple instruction model
        simple_accuracy = self.compare_with_ground_truth()
        
        print(f"\n" + "="*50)
        print("FINAL APPROACH COMPARISON")
        print("="*50)
        print(f"1. Baseline (Pre-trained + context): 70.0%")
        print(f"2. Span-based Fine-tuned (with context): 100.0%") 
        print(f"3. Simple Instruction (NO context): {simple_accuracy:.1f}%")
        
        print(f"\n📋 KEY DIFFERENCES:")
        print(f"• Baseline: Uses pre-trained model + context")
        print(f"• Span-based: Extracts answers from context")
        print(f"• Simple Instruction: Direct question → answer (your preferred format!)")
        
        return {
            'baseline': 70.0,
            'span_based': 100.0, 
            'simple_instruction': simple_accuracy
        }

def main():
    """Main testing function."""
    print("Simple Instruction Model Tester (No Context Format)")
    print("=" * 55)
    
    # Initialize tester
    tester = SimpleInstructionTester()
    
    # Load model
    if not tester.load_model():
        print("Failed to load simple instruction model. Please train it first:")
        print("python finetune_simple_instruction.py")
        return
    
    # Run tests
    tester.test_sample_questions()
    
    # Compare all approaches
    tester.compare_all_approaches()
    
    print("\nSimple instruction model testing completed!")

if __name__ == "__main__":
    main()