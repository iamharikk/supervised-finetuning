#!/usr/bin/env python3
"""
Complete pipeline for DistilBERT fine-tuning on financial Q&A data.
This script runs the entire process from data preprocessing to model testing.
"""

import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        if e.stderr:
            print("Error details:", e.stderr)
        return False

def check_prerequisites():
    """Check if required files exist."""
    required_files = [
        '../data/q-and-a/fin-data-qa-pairs.txt',
        'preprocess_distilbert.py',
        'finetune_distilbert.py',
        'test_distilbert.py',
        '../requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True

def update_requirements():
    """Add DistilBERT specific requirements."""
    distilbert_requirements = [
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "accelerate>=0.12.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0"
    ]
    
    try:
        with open('../requirements.txt', 'r') as f:
            existing = f.read()
        
        # Add missing requirements
        for req in distilbert_requirements:
            package = req.split('>=')[0]
            if package not in existing:
                with open('../requirements.txt', 'a') as f:
                    f.write(f"\n{req}")
        
        print("Requirements updated for DistilBERT")
        return True
    except Exception as e:
        print(f"Error updating requirements: {e}")
        return False

def main():
    """Run the complete DistilBERT pipeline."""
    print("DistilBERT Financial Q&A Fine-tuning Pipeline")
    print("="*55)
    
    # Change to distilbert_approach directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check prerequisites
    if not check_prerequisites():
        print("Please ensure all required files are present.")
        return
    
    # Update requirements
    update_requirements()
    
    # Step 1: Install requirements
    print("\nInstalling requirements...")
    install_success = run_command(
        "pip install -r ../requirements.txt",
        "Installing Python dependencies"
    )
    
    if not install_success:
        print("Failed to install requirements. Please install manually.")
        return
    
    # Step 2: Preprocess data
    preprocess_success = run_command(
        "python preprocess_distilbert.py",
        "Preprocessing Q&A data for DistilBERT"
    )
    
    if not preprocess_success:
        print("Data preprocessing failed.")
        return
    
    # Step 3: Fine-tune DistilBERT model
    print("\nStarting DistilBERT fine-tuning...")
    print("DistilBERT is more efficient than GPT-2 - this should be faster!")
    
    finetune_success = run_command(
        "python finetune_distilbert.py",
        "Fine-tuning DistilBERT model"
    )
    
    if not finetune_success:
        print("Model fine-tuning failed.")
        return
    
    # Step 4: Test model
    test_success = run_command(
        "python test_distilbert.py",
        "Testing the fine-tuned DistilBERT model"
    )
    
    if test_success:
        print("\n" + "="*60)
        print("🎉 DISTILBERT PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Your DistilBERT Q&A model is ready!")
        print("Model location: ./models/distilbert_financial_qa")
        print("\nAdvantages of DistilBERT over GPT-2:")
        print("✅ Better accuracy for factual Q&A")
        print("✅ Less hallucination")
        print("✅ Faster inference")
        print("✅ More reliable financial answers")
        print("\nYou can now use test_distilbert.py to interact with your model.")
    else:
        print("Model testing encountered issues, but fine-tuning completed.")

def run_individual_step():
    """Run individual pipeline steps."""
    print("\nDistilBERT Pipeline - Individual Steps")
    print("="*40)
    print("1. Preprocess data")
    print("2. Fine-tune model")
    print("3. Test model")
    print("4. Run full pipeline")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        run_command("python preprocess_distilbert.py", "Data preprocessing")
    elif choice == '2':
        run_command("python finetune_distilbert.py", "Model fine-tuning")
    elif choice == '3':
        run_command("python test_distilbert.py", "Model testing")
    elif choice == '4':
        main()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--individual":
        run_individual_step()
    else:
        main()