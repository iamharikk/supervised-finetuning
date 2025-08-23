import torch
import json
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import os
from torch.utils.data import Dataset
from typing import Dict, List

class FinancialQADataset(Dataset):
    """Custom dataset for financial Q&A pairs from JSON."""
    
    def __init__(self, tokenizer, json_file_path, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Format as: Instruction + Output for causal language modeling
        self.examples = []
        for item in self.data:
            text = f"{item['instruction']} {item['output']}<|endoftext|>"
            self.examples.append(text)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

class GPT2FineTuner:
    """Fine-tune GPT-2 for financial Q&A."""
    
    def __init__(self, model_name='gpt2', output_dir='./models/financial_qa_gpt2'):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize tokenizer and model."""
        print("Loading tokenizer and model...")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded: {self.model_name}")
        print(f"Vocab size: {len(self.tokenizer)}")
        
    def prepare_dataset(self, data_path, max_length=512):
        """Prepare dataset for training."""
        dataset = FinancialQADataset(self.tokenizer, data_path, max_length)
        return dataset
        
    def train(self, data_path, epochs=3, batch_size=2, learning_rate=5e-5, 
              save_steps=100, eval_steps=100, warmup_steps=50):
        """Fine-tune the model."""
        
        if not self.tokenizer or not self.model:
            self.setup_model_and_tokenizer()
            
        print("Preparing dataset...")
        dataset = self.prepare_dataset(data_path)
        
        # Split dataset into train/validation (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=5,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the model
        print("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Model saved to {self.output_dir}")
        
    def load_finetuned_model(self, model_path=None):
        """Load a fine-tuned model."""
        if model_path is None:
            model_path = self.output_dir
            
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.eval()
        print(f"Fine-tuned model loaded from {model_path}")

def main():
    """Main training function."""
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('./models', exist_ok=True)
    
    # Initialize fine-tuner
    fine_tuner = GPT2FineTuner(
        model_name='gpt2',
        output_dir='./models/financial_qa_gpt2'
    )
    
    # Setup model
    fine_tuner.setup_model_and_tokenizer()
    
    # Train the model (improved parameters for better quality)
    fine_tuner.train(
        data_path='data/training_data.json',
        epochs=8,  # Increased for better memorization of financial facts
        batch_size=1,  # Keep at 1 for memory efficiency
        learning_rate=2e-5,  # Lower learning rate for stability
        save_steps=25,  # More frequent saves
        eval_steps=25,  # More frequent evaluation
        warmup_steps=20  # More warmup for stability
    )
    
    print("Fine-tuning completed!")

if __name__ == "__main__":
    main()