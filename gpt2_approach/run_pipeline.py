#!/usr/bin/env python3
"""
Complete pipeline for GPT-2 fine-tuning on financial Q&A data.
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
        'data/q-and-a/fin-data-qa-pairs.txt',
        'preprocess_data.py',
        'finetune_gpt2.py',
        'test_model.py',
        'requirements.txt'
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

def main():
    """Run the complete pipeline."""
    print("GPT-2 Financial Q&A Fine-tuning Pipeline")
    print("="*50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("Please ensure all required files are present.")
        return
    
    # Step 1: Install requirements
    print("\nInstalling requirements...")
    install_success = run_command(
        "pip install -r requirements.txt",
        "Installing Python dependencies"
    )
    
    if not install_success:
        print("Failed to install requirements. Please install manually.")
        return
    
    # Step 2: Preprocess data
    preprocess_success = run_command(
        "python preprocess_data.py",
        "Preprocessing Q&A data to JSON format"
    )
    
    if not preprocess_success:
        print("Data preprocessing failed.")
        return
    
    # Step 3: Fine-tune model
    print("\nStarting model fine-tuning...")
    print("This may take a while depending on your hardware...")
    
    finetune_success = run_command(
        "python finetune_gpt2.py",
        "Fine-tuning GPT-2 model"
    )
    
    if not finetune_success:
        print("Model fine-tuning failed.")
        return
    
    # Step 4: Test model
    test_success = run_command(
        "python test_model.py",
        "Testing the fine-tuned model"
    )
    
    if test_success:
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Your fine-tuned model is ready!")
        print("Model location: ./models/financial_qa_gpt2")
        print("You can now use test_model.py to interact with your model.")
    else:
        print("Model testing encountered issues, but fine-tuning completed.")

if __name__ == "__main__":
    main()