import torch
import json
from finetune_distilbert_instruction import DistilBERTInstructionFineTuner

class DistilBERTInstructionTester:
    """Test class for instruction-style DistilBERT model."""
    
    def __init__(self, model_path='./models/distilbert_instruction_qa'):
        self.model_path = model_path
        self.trainer = DistilBERTInstructionFineTuner(output_dir=model_path)
        
    def load_model(self):
        """Load the instruction-tuned model."""
        return self.trainer.load_model()
    
    def test_with_sample_questions(self):
        """Test with sample questions."""
        sample_questions = [
            "What was TCS's sales turnover in 2025?",
            "What was TCS's net profit in 2025?", 
            "What was TCS's employee cost in 2025?",
            "What was the growth in TCS's net profit from 2024 to 2025?",
            "What is TCS's profit margin in 2025?",
            "What was TCS's total income in 2024?"
        ]
        
        print("\n" + "="*70)
        print("TESTING INSTRUCTION-STYLE DISTILBERT MODEL")
        print("="*70)
        
        for i, question in enumerate(sample_questions, 1):
            answer, confidence = self.trainer.predict_answer(question)
            print(f"\n{i}. Question: {question}")
            print(f"   Answer: {answer}")
            print(f"   Confidence: {confidence:.1%}")
            print("-" * 60)
    
    def compare_with_ground_truth(self, test_data_path='../data/distilbert_simple_format.json'):
        """Compare with ground truth answers."""
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print(f"Test data file not found: {test_data_path}")
            return 0
        
        print("\n" + "="*70)
        print("COMPARING INSTRUCTION MODEL WITH GROUND TRUTH")
        print("="*70)
        
        correct = 0
        total = min(10, len(test_data))
        
        for i, item in enumerate(test_data[:total]):
            question = item['question']
            ground_truth = item['answer']
            
            predicted_answer, confidence = self.trainer.predict_answer(question)
            
            # Check accuracy
            is_correct = self._check_answer_accuracy(ground_truth, predicted_answer)
            
            if is_correct:
                correct += 1
            
            status = "[PASS]" if is_correct else "[FAIL]"
            print(f"\n{i+1}. {status} Question: {question}")
            print(f"   Ground Truth: {ground_truth}")
            print(f"   Predicted: {predicted_answer} (confidence: {confidence:.1%})")
            print("-" * 60)
        
        accuracy = (correct / total) * 100
        print(f"\nInstruction Model Accuracy: {accuracy:.1f}% ({correct}/{total})")
        return accuracy
    
    def _check_answer_accuracy(self, ground_truth, predicted_answer):
        """Check if predicted answer matches ground truth."""
        return ground_truth.strip() == predicted_answer.strip()
    
    def compare_all_models(self):
        """Compare instruction model with other approaches."""
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*70)
        
        # Test instruction model
        instruction_accuracy = self.compare_with_ground_truth()
        
        # Load baseline results if available
        baseline_accuracy = 0
        try:
            with open('baseline_results.json', 'r') as f:
                baseline_results = json.load(f)
                baseline_accuracy = baseline_results.get('accuracy', 0)
        except FileNotFoundError:
            print("Baseline results not found")
        
        print(f"\n" + "="*50)
        print("FINAL COMPARISON")
        print("="*50)
        print(f"Baseline (Pre-trained): {baseline_accuracy:.1f}%")
        print(f"Original Fine-tuned: 50.0%")
        print(f"Improved Fine-tuned: 100.0%")
        print(f"Instruction-style: {instruction_accuracy:.1f}%")
        
        # Determine best approach
        results = {
            'Baseline': baseline_accuracy,
            'Original Fine-tuned': 50.0,
            'Improved Fine-tuned': 100.0,
            'Instruction-style': instruction_accuracy
        }
        
        best_model = max(results, key=results.get)
        best_score = results[best_model]
        
        print(f"\nBest performing model: {best_model} ({best_score:.1f}%)")
        
        return results

def main():
    """Main testing function."""
    print("DistilBERT Instruction-Style Model Tester")
    print("=" * 50)
    
    # Initialize tester
    tester = DistilBERTInstructionTester()
    
    # Load model
    if not tester.load_model():
        print("Failed to load instruction model. Please train it first:")
        print("python finetune_distilbert_instruction.py")
        return
    
    # Run tests
    tester.test_with_sample_questions()
    
    # Compare with ground truth and other models
    tester.compare_all_models()
    
    print("\nInstruction-style model testing completed!")

if __name__ == "__main__":
    main()