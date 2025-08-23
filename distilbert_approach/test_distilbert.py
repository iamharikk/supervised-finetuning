import torch
import json
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
from finetune_distilbert import DistilBERTFinancialQA

class DistilBERTQATester:
    """Test class for DistilBERT financial Q&A model."""
    
    def __init__(self, model_path='./models/distilbert_financial_qa'):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.context = None
        
    def load_model(self):
        """Load the fine-tuned DistilBERT model."""
        try:
            print(f"Loading model from {self.model_path}...")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_path)
            self.model = DistilBertForQuestionAnswering.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have trained the model first.")
            return False
    
    def load_context(self, data_file='../data/distilbert_simple_format.json'):
        """Load context from the training data."""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data and len(data) > 0:
                self.context = data[0]['context']  # Use the shared context
                print(f"Context loaded: {len(self.context)} characters")
                return True
        except Exception as e:
            print(f"Error loading context: {e}")
            return False
    
    def answer_question(self, question, context=None, max_length=512):
        """Answer a question using the fine-tuned DistilBERT model."""
        if not self.model or not self.tokenizer:
            return "Model not loaded. Please load the model first.", 0.0
        
        if context is None:
            context = self.context
        
        if context is None:
            return "No context available. Please load context first.", 0.0
        
        # Tokenize input
        inputs = self.tokenizer(
            question,
            context,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get answer span
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        # Find best start and end positions
        start_idx = torch.argmax(start_logits, dim=1)[0]
        end_idx = torch.argmax(end_logits, dim=1)[0]
        
        # Ensure end >= start
        if end_idx < start_idx:
            end_idx = start_idx
        
        # Extract answer tokens
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Calculate confidence (simple approach)
        start_confidence = torch.softmax(start_logits, dim=1)[0][start_idx].item()
        end_confidence = torch.softmax(end_logits, dim=1)[0][end_idx].item()
        confidence = (start_confidence + end_confidence) / 2
        
        # Clean up answer
        answer = answer.strip()
        if not answer:
            answer = "Unable to find answer in context"
            confidence = 0.0
        
        return answer, confidence
    
    def test_with_sample_questions(self):
        """Test the model with predefined sample questions."""
        sample_questions = [
            "What was TCS's sales turnover in 2025?",
            "What was TCS's net profit in 2025?",
            "What was TCS's employee cost in 2025?",
            "What was the growth in TCS's net profit from 2024 to 2025?",
            "What is TCS's profit margin in 2025?",
            "What was TCS's total income in 2024?"
        ]
        
        print("\n" + "="*60)
        print("TESTING MODEL WITH SAMPLE QUESTIONS")
        print("="*60)
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\n{i}. Question: {question}")
            answer, confidence = self.answer_question(question)
            print(f"   Answer: {answer}")
            print(f"   Confidence: {confidence:.1%}")
            print("-" * 50)
    
    def compare_with_ground_truth(self, test_data_path='../data/distilbert_simple_format.json'):
        """Compare model outputs with ground truth answers."""
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print(f"Test data file not found: {test_data_path}")
            return
        
        print("\n" + "="*60)
        print("COMPARING WITH GROUND TRUTH")
        print("="*60)
        
        correct = 0
        total = min(10, len(test_data))  # Test first 10 samples
        
        for i, item in enumerate(test_data[:total]):
            question = item['question']
            ground_truth = item['answer']
            
            generated_answer, confidence = self.answer_question(question)
            
            # Simple accuracy check - look for key numbers or exact match
            is_correct = self._check_answer_accuracy(ground_truth, generated_answer)
            
            if is_correct:
                correct += 1
            
            status = "[PASS]" if is_correct else "[FAIL]"
            print(f"\n{i+1}. {status} Question: {question}")
            print(f"   Ground Truth: {ground_truth}")
            print(f"   Generated: {generated_answer} (confidence: {confidence:.1%})")
            print("-" * 50)
        
        accuracy = (correct / total) * 100
        print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{total})")
        return accuracy
    
    def _check_answer_accuracy(self, ground_truth, generated_answer):
        """Check if generated answer is accurate compared to ground truth."""
        # Convert to lowercase for comparison
        gt_lower = ground_truth.lower()
        gen_lower = generated_answer.lower()
        
        # Extract numbers from both answers
        import re
        gt_numbers = re.findall(r'[\d,]+\.?\d*', ground_truth)
        gen_numbers = re.findall(r'[\d,]+\.?\d*', generated_answer)
        
        # Check if key numbers match
        for gt_num in gt_numbers:
            if gt_num in generated_answer:
                return True
        
        # Check for partial string match
        if len(gt_lower) > 10:  # For longer answers
            return gt_lower in gen_lower or gen_lower in gt_lower
        else:  # For shorter answers
            return gt_lower == gen_lower
    
    def interactive_test(self):
        """Interactive testing mode."""
        print("\n" + "="*60)
        print("INTERACTIVE TESTING MODE")
        print("Type 'quit' to exit")
        print("="*60)
        
        while True:
            question = input("\nEnter your financial question about TCS: ").strip()
            
            if question.lower() == 'quit':
                print("Exiting interactive mode...")
                break
            
            if question:
                answer, confidence = self.answer_question(question)
                print(f"Answer: {answer}")
                print(f"Confidence: {confidence:.1%}")
            else:
                print("Please enter a valid question.")
    
    def benchmark_performance(self):
        """Comprehensive performance benchmark."""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK")
        print("="*60)
        
        # Test with different question types
        question_types = {
            "Specific Values": [
                "What was TCS's sales turnover in 2025?",
                "What was TCS's net profit in 2024?"
            ],
            "Growth Questions": [
                "What was the growth in TCS's net profit from 2024 to 2025?",
                "How much did TCS's employee cost increase from 2024 to 2025?"
            ],
            "Ratios/Percentages": [
                "What is TCS's profit margin in 2025?",
                "What was TCS's equity dividend percentage in 2025?"
            ]
        }
        
        for category, questions in question_types.items():
            print(f"\n{category}:")
            for question in questions:
                answer, confidence = self.answer_question(question)
                print(f"  Q: {question}")
                print(f"  A: {answer} (confidence: {confidence:.1%})")

def main():
    """Main testing function."""
    print("DistilBERT Financial Q&A Model Tester")
    print("=" * 45)
    
    # Initialize tester
    tester = DistilBERTQATester()
    
    # Load model
    if not tester.load_model():
        print("Failed to load model. Please train the model first using:")
        print("python finetune_distilbert.py")
        return
    
    # Load context
    if not tester.load_context():
        print("Failed to load context. Please run preprocessing first:")
        print("python preprocess_distilbert.py")
        return
    
    # Run tests
    tester.test_with_sample_questions()
    accuracy = tester.compare_with_ground_truth()
    
    if accuracy >= 90:
        print(f"\n🎉 Excellent performance! Accuracy: {accuracy:.1f}%")
    elif accuracy >= 70:
        print(f"\n👍 Good performance! Accuracy: {accuracy:.1f}%")
    else:
        print(f"\n📈 Room for improvement. Accuracy: {accuracy:.1f}%")
    
    # Optional: Interactive testing
    user_input = input("\nWould you like to try interactive testing? (y/n): ")
    if user_input.lower() == 'y':
        tester.interactive_test()

if __name__ == "__main__":
    main()