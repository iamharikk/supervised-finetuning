import torch
import json
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from torch.utils.data import Dataset
import os
import numpy as np

class DistilBERTInstructionDataset(Dataset):
    """Dataset for instruction-style fine-tuning with DistilBERT."""
    
    def __init__(self, tokenizer, data_file, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.answer_to_id = {}
        self.id_to_answer = {}
        
        # Load and process data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loading {len(data)} Q&A examples for instruction-style training...")
        
        # Create answer vocabulary
        unique_answers = set()
        for item in data:
            answer = item['answer'].strip()
            unique_answers.add(answer)
        
        # Create answer mappings
        for idx, answer in enumerate(sorted(unique_answers)):
            self.answer_to_id[answer] = idx
            self.id_to_answer[idx] = answer
        
        print(f"Created vocabulary of {len(unique_answers)} unique answers")
        
        # Process examples
        for item in data:
            question = item['question']
            answer = item['answer'].strip()
            
            # Create instruction-style input
            instruction = "You are a financial analyst. Answer this question about TCS based on the data provided."
            
            # Format as instruction-input
            formatted_input = f"Instruction: {instruction}\n\nQuestion: {question}\n\nAnswer:"
            
            if answer in self.answer_to_id:
                self.examples.append({
                    'input_text': formatted_input,
                    'answer': answer,
                    'label': self.answer_to_id[answer]
                })
        
        print(f"Created {len(self.examples)} instruction examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize the instruction-formatted input
        encoding = self.tokenizer(
            example['input_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(example['label'], dtype=torch.long)
        }

class DistilBERTInstructionFineTuner:
    """DistilBERT instruction-style fine-tuner."""
    
    def __init__(self, model_name='distilbert-base-uncased', output_dir='./models/distilbert_instruction_qa'):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.answer_to_id = {}
        self.id_to_answer = {}
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_model_and_tokenizer(self, num_labels):
        """Initialize DistilBERT for classification."""
        print(f"Loading DistilBERT model: {self.model_name}")
        
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        
        print(f"Model loaded with {num_labels} output classes")
    
    def train_instruction_style(self, data_file, epochs=8, batch_size=8, learning_rate=2e-5):
        """Train DistilBERT with instruction-style approach."""
        
        print("Preparing instruction dataset...")
        dataset = DistilBERTInstructionDataset(
            tokenizer=DistilBertTokenizerFast.from_pretrained(self.model_name),
            data_file=data_file
        )
        
        if len(dataset) == 0:
            print("No valid examples found!")
            return
        
        # Store answer mappings
        self.answer_to_id = dataset.answer_to_id
        self.id_to_answer = dataset.id_to_answer
        
        # Setup model with correct number of labels
        self.setup_model_and_tokenizer(len(self.answer_to_id))
        
        # Split dataset
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        
        if val_size == 0:
            val_size = max(1, len(dataset) // 10)
            train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Number of answer classes: {len(self.answer_to_id)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=100,
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            
            # Logging
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=25,
            
            # Optimization
            fp16=torch.cuda.is_available(),
            
            # Reproducibility
            seed=42,
        )
        
        # Custom metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = np.mean(predictions == labels)
            return {"accuracy": accuracy}
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
        
        print("Starting instruction-style DistilBERT training...")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        
        # Train
        trainer.train()
        
        # Save model and mappings
        print(f"Saving instruction-tuned model to {self.output_dir}...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save answer mappings
        mappings = {
            'answer_to_id': self.answer_to_id,
            'id_to_answer': self.id_to_answer
        }
        with open(f"{self.output_dir}/answer_mappings.json", 'w') as f:
            json.dump(mappings, f, indent=2)
        
        print("Instruction-style DistilBERT training completed!")
    
    def load_model(self, model_path=None):
        """Load the instruction-tuned model."""
        if model_path is None:
            model_path = self.output_dir
        
        try:
            # Load answer mappings
            with open(f"{model_path}/answer_mappings.json", 'r') as f:
                mappings = json.load(f)
            self.answer_to_id = mappings['answer_to_id']
            self.id_to_answer = {int(k): v for k, v in mappings['id_to_answer'].items()}
            
            # Load model
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            
            print(f"Instruction-tuned model loaded from {model_path}")
            print(f"Answer vocabulary size: {len(self.answer_to_id)}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_answer(self, question, top_k=3):
        """Predict answer using instruction-style approach."""
        if not self.model or not self.tokenizer:
            return "Model not loaded", 0.0
        
        # Format as instruction
        instruction = "You are a financial analyst. Answer this question about TCS based on the data provided."
        formatted_input = f"Instruction: {instruction}\n\nQuestion: {question}\n\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_input,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities[0], k=top_k)
        
        # Return best prediction
        best_id = top_indices[0].item()
        best_prob = top_probs[0].item()
        best_answer = self.id_to_answer.get(best_id, "Unknown answer")
        
        return best_answer, best_prob

def main():
    """Main function for instruction-style DistilBERT fine-tuning."""
    print("DistilBERT Instruction-Style Financial Q&A Fine-tuning")
    print("=" * 65)
    
    # Initialize trainer
    trainer = DistilBERTInstructionFineTuner()
    
    # Train with instruction approach
    trainer.train_instruction_style(
        data_file='../data/distilbert_simple_format.json',
        epochs=8,
        batch_size=8,
        learning_rate=2e-5
    )
    
    print("\nInstruction-style DistilBERT fine-tuning completed!")
    print("Test with: python test_distilbert_instruction.py")

if __name__ == "__main__":
    main()