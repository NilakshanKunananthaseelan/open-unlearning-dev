#!/usr/bin/env python3
"""
Example script demonstrating the usage of HyperbolicBusemannTrainer
for machine unlearning with hyperbolic geometry.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import sys
import os

# Add the src directory to the path to import the trainer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from trainer.unlearn.hbul import HyperbolicBusemannTrainer


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    # Sample conversations for unlearning
    conversations = [
        {
            "text": "Human: What is the capital of France?\nAssistant: The capital of France is Paris.",
            "labels": [-100] * 10 + list(range(50256, 50256 + 20))  # -100 for human text, token IDs for assistant
        },
        {
            "text": "Human: How do you make coffee?\nAssistant: To make coffee, you need coffee beans, hot water, and a coffee maker.",
            "labels": [-100] * 12 + list(range(50256, 50256 + 25))
        },
        {
            "text": "Human: What is machine learning?\nAssistant: Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "labels": [-100] * 11 + list(range(50256, 50256 + 30))
        }
    ]
    
    return Dataset.from_list(conversations)


def main():
    """Main function demonstrating the HyperbolicBusemannTrainer."""
    
    # 1. Load a pre-trained model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "microsoft/DialoGPT-small"  # Small model for demonstration
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Create sample dataset
    print("Creating sample dataset...")
    dataset = create_sample_dataset()
    
    # 3. Define concepts to retain (these will be used to create ideal prototypes)
    retain_prompts = [
        "What is the capital of France?",
        "How do you make coffee?",
        "What is machine learning?"
    ]
    
    # 4. Set up training arguments
    training_args = TrainingArguments(
        output_dir="./hbul_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./hbul_logs",
        logging_steps=1,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        max_seq_length=128,
    )
    
    # 5. Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 6. Initialize the HyperbolicBusemannTrainer
    print("Initializing HyperbolicBusemannTrainer...")
    trainer = HyperbolicBusemannTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        retain_prompts=retain_prompts,
        lambda_hyp=1.0,      # Weight for hyperbolic loss
        lambda_ot=0.5,       # Weight for optimal transport loss
        lambda_rep=0.3,      # Weight for repulsive loss
        margin=0.1,          # Margin for repulsive hinge loss
    )
    
    # 7. Train the model
    print("Starting training...")
    trainer.train()
    
    # 8. Monitor losses
    print("\nFinal loss breakdown:")
    final_losses = trainer.get_current_losses()
    for loss_name, loss_value in final_losses.items():
        print(f"{loss_name}: {loss_value}")
    
    # 9. Save the model
    print("\nSaving model...")
    trainer.save_model("./hbul_final_model")
    tokenizer.save_pretrained("./hbul_final_model")
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
