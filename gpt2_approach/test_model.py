import torch
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from finetune_gpt2 import GPT2FineTuner

class FinancialQAModelTester:
    """Test class for the fine-tuned GPT-2 model."""
    
    def __init__(self, model_path='./models/financial_qa_gpt2'):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            print(f"Loading model from {self.model_path}...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_answer(self, question, max_length=150, temperature=0.7, do_sample=True):
        """Generate an answer for a given financial question."""
        if not self.model or not self.tokenizer:
            print("Model not loaded. Please load the model first.")
            return None
            
        # Format the input prompt
        prompt = f"Answer the following financial question about TCS: {question}"
        
        # Tokenize input with attention mask
        encoding = self.tokenizer(prompt, return_tensors='pt', padding=True)
        inputs = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the input prompt)
        if prompt in response:
            answer = response.replace(prompt, "").strip()
        else:
            answer = response
            
        return answer
    
    def test_with_sample_questions(self):
        """Test the model with predefined sample questions."""
        sample_questions = [
            "What was TCS's sales turnover in Mar '25?",
            "What was TCS's net profit in Mar '25?",
            "What was TCS's employee cost in Mar '25?",
            "What was the growth in TCS's net profit from Mar '24 to Mar '25?",
            "What is TCS's profit margin in 2025?"
        ]
        
        print("\n" + "="*60)
        print("TESTING MODEL WITH SAMPLE QUESTIONS")
        print("="*60)
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\n{i}. Question: {question}")
            answer = self.generate_answer(question)
            print(f"   Answer: {answer}")
            print("-" * 50)
    
    def test_with_custom_question(self, question):
        """Test the model with a custom question."""
        print(f"\nCustom Question: {question}")
        answer = self.generate_answer(question)
        print(f"Generated Answer: {answer}")
        return answer
    
    def compare_with_ground_truth(self, test_data_path='data/training_data.json'):
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
        
        # Test with first 5 samples
        for i, item in enumerate(test_data[:5]):
            instruction = item['instruction']
            ground_truth = item['output']
            
            # Extract question from instruction
            question = instruction.replace("Answer the following financial question about TCS: ", "")
            
            generated_answer = self.generate_answer(question)
            
            print(f"\n{i+1}. Question: {question}")
            print(f"   Ground Truth: {ground_truth}")
            print(f"   Generated: {generated_answer}")
            print("-" * 50)
    
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
                answer = self.generate_answer(question)
                print(f"Answer: {answer}")
            else:
                print("Please enter a valid question.")
    
    def evaluate_model_performance(self, test_data_path='data/training_data.json', num_samples=10):
        """Evaluate model performance on a subset of data."""
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print(f"Test data file not found: {test_data_path}")
            return
        
        print("\n" + "="*60)
        print(f"EVALUATING MODEL PERFORMANCE ON {num_samples} SAMPLES")
        print("="*60)
        
        correct_responses = 0
        total_samples = min(num_samples, len(test_data))
        
        for i, item in enumerate(test_data[:total_samples]):
            instruction = item['instruction']
            ground_truth = item['output']
            
            # Extract question from instruction
            question = instruction.replace("Answer the following financial question about TCS: ", "")
            
            generated_answer = self.generate_answer(question)
            
            # Simple evaluation: check if key numbers/terms are present
            # This is a basic evaluation - could be made more sophisticated
            is_correct = self._simple_answer_check(ground_truth, generated_answer)
            
            if is_correct:
                correct_responses += 1
                
            print(f"{i+1}. {'[PASS]' if is_correct else '[FAIL]'} Question: {question[:50]}...")
        
        accuracy = (correct_responses / total_samples) * 100
        print(f"\nSimple Accuracy: {accuracy:.1f}% ({correct_responses}/{total_samples})")
    
    def _simple_answer_check(self, ground_truth, generated_answer):
        """Simple check to see if generated answer contains key information."""
        # Extract numbers from ground truth
        import re
        gt_numbers = re.findall(r'[\d,]+\.?\d*', ground_truth)
        gen_numbers = re.findall(r'[\d,]+\.?\d*', generated_answer)
        
        # Check if at least one key number is present
        for num in gt_numbers[:2]:  # Check first 2 numbers
            if num in generated_answer:
                return True
        return False

def main():
    """Main testing function."""
    print("Financial Q&A Model Tester")
    print("=" * 40)
    
    # Initialize tester
    tester = FinancialQAModelTester()
    
    # Try to load the model
    if not tester.load_model():
        print("Failed to load model. Make sure you have trained the model first.")
        return
    
    # Run different types of tests
    tester.test_with_sample_questions()
    tester.compare_with_ground_truth()
    tester.evaluate_model_performance()
    
    # Optional: Interactive testing
    user_input = input("\nWould you like to try interactive testing? (y/n): ")
    if user_input.lower() == 'y':
        tester.interactive_test()

if __name__ == "__main__":
    main()