# =====================================================
# Optimized BART-LoRA Training for T4 GPU
# Uses effective batch size 8 via gradient accumulation
# Includes padding, length updates, and stable T4 config
# =====================================================

# !pip install -q datasets transformers peft wandb accelerate

import os
import wandb
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# -------------------------
# Config
# -------------------------
MODEL_NAME = "facebook/bart-large-cnn"
OUTPUT_DIR = "/content/llmed_certification_FineTuneFlow/ft_outputs/bart_lora_highlightsum"

#OUTPUT_DIR = "./ft_outputs/bart_lora_highlightsum"


N_SAMPLES = 2000
EPOCHS = 3
MICRO_BATCH_SIZE = 4
GRAD_ACC = 2                     # → effective batch size = 8
LEARNING_RATE = 2e-4

MAX_INPUT_LENGTH = 768           # reduced from 1024 → faster
MAX_TARGET_LENGTH = 192          # increased summary length

WANDB_PROJECT = "highlightsum_bart_lora"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# -------------------------
# W&B (optional)
# -------------------------
# Comment out the next line to disable W&B logging
wandb.login()

# -------------------------
# Load Dataset
# -------------------------
dataset = load_dataset("knkarthick/highlightsum")["train"].select(range(N_SAMPLES))
print(f"Loaded {len(dataset)} samples.")

# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

# -------------------------
# Model + LoRA
# -------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],   # correct for BART
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)
model.to(device)

# -------------------------
# Tokenization
# (NOW INCLUDES STATIC PADDING → faster & stable)
# -------------------------
def tokenize_fn(example):
    inputs = tokenizer(
        example["dialogue"],
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length"
    )
    label_ids = tokenizer(
        example["summary"],
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
        padding="max_length"
    ).input_ids

    inputs["labels"] = label_ids
    return inputs

tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

# -------------------------
# Data Collator
# -------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# -------------------------
# Training Arguments
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=50,
    save_strategy="no",              # save ONLY final model → MUCH faster
    report_to="wandb",
    run_name="bart_lora_highlightsum_t4_optimized",
)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# -------------------------
# Train
# -------------------------
trainer.train()

# -------------------------
# Save final model
# -------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n🎉 Fine-tuned model saved to: {OUTPUT_DIR}")
