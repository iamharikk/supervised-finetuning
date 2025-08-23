import torch
import json
from transformers import (
    DistilBertTokenizer, 
    DistilBertForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from torch.utils.data import Dataset
import os

class DistilBERTQADataset(Dataset):
    """Dataset for DistilBERT Q&A fine-tuning."""
    
    def __init__(self, tokenizer, data_file, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load simple format data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} Q&A examples")
        
        # Process each example
        for item in data:
            question = item['question']
            context = item['context']
            answer = item['answer']
            
            # Find answer in context (case insensitive)
            context_lower = context.lower()
            answer_lower = answer.lower()
            
            start_idx = context_lower.find(answer_lower)
            if start_idx == -1:
                # Try to find key numbers from the answer
                import re
                numbers = re.findall(r'[\d,]+\.?\d*', answer)
                for num in numbers:
                    start_idx = context.find(num)
                    if start_idx != -1:
                        answer = num  # Use the number as answer
                        break
            
            if start_idx != -1:
                end_idx = start_idx + len(answer)
                self.examples.append({
                    'question': question,
                    'context': context,
                    'answer': answer,
                    'start_char': start_idx,
                    'end_char': end_idx
                })
            else:
                print(f"Warning: Could not find answer '{answer}' in context")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize question and context
        encoding = self.tokenizer(
            example['question'],
            example['context'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            verbose=False
        )
        
        # Find the actual token positions for the answer
        offset_mapping = encoding['offset_mapping'].squeeze().tolist()
        start_char = example['start_char']
        end_char = example['end_char']
        
        # Find token positions that correspond to the character positions
        start_token = 0
        end_token = 0
        
        for i, (offset_start, offset_end) in enumerate(offset_mapping):
            # Skip special tokens (they have offset (0,0))
            if offset_start == 0 and offset_end == 0:
                continue
                
            # Find start token
            if start_token == 0 and offset_start <= start_char < offset_end:
                start_token = i
            
            # Find end token  
            if offset_start < end_char <= offset_end:
                end_token = i
                break
        
        # Fallback if no proper alignment found
        if start_token == 0 or end_token == 0:
            start_token = 1
            end_token = min(10, self.max_length - 2)
        
        # Ensure end >= start
        if end_token < start_token:
            end_token = start_token
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'start_positions': torch.tensor(start_token, dtype=torch.long),
            'end_positions': torch.tensor(end_token, dtype=torch.long)
        }

class DistilBERTFinancialQA:
    """DistilBERT fine-tuner for financial Q&A."""
    
    def __init__(self, model_name='distilbert-base-cased-distilled-squad', output_dir='./models/distilbert_financial_qa'):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize tokenizer and model."""
        print("Loading DistilBERT tokenizer and model...")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(self.model_name)
        
        print(f"Model loaded: {self.model_name}")
        print(f"This model is already fine-tuned on SQuAD dataset")
    
    def train(self, data_file, epochs=3, batch_size=8, learning_rate=3e-5, 
              save_steps=100, eval_steps=100, warmup_steps=100):
        """Fine-tune DistilBERT on financial Q&A data."""
        
        if not self.tokenizer or not self.model:
            self.setup_model_and_tokenizer()
        
        print("Preparing dataset...")
        dataset = DistilBERTQADataset(self.tokenizer, data_file)
        
        if len(dataset) == 0:
            print("No valid examples found in dataset!")
            return
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        if val_size == 0:
            val_size = 1
            train_size = len(dataset) - 1
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=10,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",
            save_total_limit=2,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save model
        print("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Model saved to {self.output_dir}")
    
    def load_model(self, model_path=None):
        """Load fine-tuned model."""
        if model_path is None:
            model_path = self.output_dir
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_path)
        self.model.eval()
        print(f"Fine-tuned model loaded from {model_path}")

def main():
    """Main training function."""
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('./models', exist_ok=True)
    
    # Initialize fine-tuner
    fine_tuner = DistilBERTFinancialQA(
        model_name='distilbert-base-cased-distilled-squad',  # Pre-trained on SQuAD
        output_dir='./models/distilbert_financial_qa'
    )
    
    # Setup model
    fine_tuner.setup_model_and_tokenizer()
    
    # Train model
    fine_tuner.train(
        data_file='../data/distilbert_simple_format.json',
        epochs=4,
        batch_size=4,  # Smaller batch size for better convergence
        learning_rate=2e-5,  # Lower learning rate since model is already trained
        save_steps=50,
        eval_steps=50,
        warmup_steps=50
    )
    
    print("Fine-tuning completed!")
    print("\nNext steps:")
    print("1. Run the test script to evaluate performance")
    print("2. Try interactive Q&A with your financial data")

if __name__ == "__main__":
    main()