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

class SimpleInstructionDataset(Dataset):
    """Dataset for simple instruction-output format."""
    
    def __init__(self, tokenizer, data_file, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.answer_to_id = {}
        self.id_to_answer = {}
        
        # Load simple instruction data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loading {len(data)} simple instruction examples...")
        
        # Create answer vocabulary
        unique_answers = set()
        for item in data:
            answer = item['output'].strip()
            unique_answers.add(answer)
        
        # Create answer mappings
        sorted_answers = sorted(unique_answers)
        for idx, answer in enumerate(sorted_answers):
            self.answer_to_id[answer] = idx
            self.id_to_answer[idx] = answer
        
        print(f"Created vocabulary of {len(unique_answers)} unique answers")
        
        # Process examples
        for item in data:
            instruction = item['instruction']
            output = item['output'].strip()
            
            if output in self.answer_to_id:
                self.examples.append({
                    'instruction': instruction,
                    'output': output,
                    'label': self.answer_to_id[output]
                })
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize just the instruction (no context needed!)
        encoding = self.tokenizer(
            example['instruction'],
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

class SimpleInstructionFineTuner:
    """Simple instruction fine-tuner without context."""
    
    def __init__(self, model_name='distilbert-base-uncased', output_dir='./models/simple_instruction_qa'):
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
    
    def train_simple_instruction(self, data_file, epochs=5, batch_size=16, learning_rate=2e-5):
        """Train DistilBERT with simple instruction format."""
        
        print("Preparing simple instruction dataset...")
        dataset = SimpleInstructionDataset(
            tokenizer=DistilBertTokenizerFast.from_pretrained(self.model_name),
            data_file=data_file
        )
        
        if len(dataset) == 0:
            print("No valid examples found!")
            return
        
        # Store answer mappings
        self.answer_to_id = dataset.answer_to_id
        self.id_to_answer = dataset.id_to_answer
        
        # Setup model
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
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=50,
            
            eval_strategy="steps",
            eval_steps=25,
            save_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=10,
            
            fp16=torch.cuda.is_available(),
            seed=42,
        )
        
        # Metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = torch.argmax(torch.from_numpy(predictions), dim=1)
            accuracy = torch.mean((predictions == torch.from_numpy(labels)).float())
            return {"accuracy": accuracy.item()}
        
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
        
        print("Starting simple instruction training...")
        print(f"No context needed - just instruction → output!")
        
        # Train
        trainer.train()
        
        # Save model and mappings
        print(f"Saving simple instruction model to {self.output_dir}...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save answer mappings
        mappings = {
            'answer_to_id': self.answer_to_id,
            'id_to_answer': self.id_to_answer
        }
        with open(f"{self.output_dir}/answer_mappings.json", 'w') as f:
            json.dump(mappings, f, indent=2)
        
        print("Simple instruction training completed!")
    
    def load_model(self, model_path=None):
        """Load the trained model."""
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
            
            print(f"Simple instruction model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, instruction):
        """Predict answer for instruction."""
        if not self.model or not self.tokenizer:
            return "Model not loaded", 0.0
        
        # Tokenize instruction only
        inputs = self.tokenizer(
            instruction,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=True
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get best prediction
        best_id = torch.argmax(probabilities[0]).item()
        confidence = probabilities[0][best_id].item()
        answer = self.id_to_answer.get(best_id, "Unknown")
        
        return answer, confidence

def main():
    """Main function for simple instruction fine-tuning."""
    print("Simple Instruction-Output Fine-tuning (No Context Needed)")
    print("=" * 60)
    
    # First create the simple format
    from create_instruction_format import create_simple_instruction_format
    create_simple_instruction_format()
    
    # Initialize trainer
    trainer = SimpleInstructionFineTuner()
    
    # Train
    trainer.train_simple_instruction(
        data_file='../data/simple_instruction_format.json',
        epochs=5,
        batch_size=16,
        learning_rate=2e-5
    )
    
    print("\nSimple instruction fine-tuning completed!")
    print("Test with: python test_simple_instruction.py")

if __name__ == "__main__":
    main()