# =====================================================
!pip install -q datasets transformers peft wandb

import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
import torch
import wandb

# -------------------------
# Config
# -------------------------
MODEL_NAME = "facebook/bart-large-cnn"
OUTPUT_DIR = "./ft_outputs/bart_lora_highlightsum"
N_SAMPLES = 2000   # first 2k samples for fine-tuning
EPOCHS = 1         # set 1 epoch for debugging
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
WANDB_PROJECT = "highlightsum_bart_lora"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# -------------------------
# W&B login
# -------------------------
wandb.login()  # prompts for API key in Colab

# -------------------------
# Load dataset
# -------------------------
dataset = load_dataset("knkarthick/highlightsum")["train"].select(range(N_SAMPLES))
print(f"Loaded {len(dataset)} samples for training.")

# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# ☢ Ensure pad_token is set
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

# -------------------------
# Model + LoRA
# -------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # attention proj layers
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)
model.to(device)

# -------------------------
# Tokenization function
# -------------------------
def tokenize_fn(example):
    inputs = tokenizer(example["dialogue"], truncation=True, max_length=MAX_INPUT_LENGTH)
    labels = tokenizer(example["summary"], truncation=True, max_length=MAX_TARGET_LENGTH).input_ids
    inputs["labels"] = labels
    return inputs

tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

# -------------------------
# Training Arguments
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=50,
    save_strategy="no",
    gradient_accumulation_steps=1,
    report_to="wandb",   # enables W&B logging
    run_name="bart_lora_highlightsum"
)

# -------------------------
# Data Collator
# -------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# -------------------------
# Run training
# -------------------------
trainer.train()

# -------------------------
# Save model & tokenizer
# -------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nℹ️ Fine-tuned model saved to {OUTPUT_DIR}")
