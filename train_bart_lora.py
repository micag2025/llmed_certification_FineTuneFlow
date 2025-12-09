# =====================================================
# Optimized BART-LoRA Training for T4 GPU
# Transformers 4.57.3 Compatible
# =====================================================

import os
import wandb
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# -------------------------
# Config
# -------------------------
MODEL_NAME = "facebook/bart-large-cnn"
OUTPUT_DIR = "/content/llmed_certification_FineTuneFlow/ft_outputs/bart_lora_highlightsum"

TRAIN_SAMPLES = 2000
VAL_SAMPLES = 200

EPOCHS = 3
MICRO_BATCH_SIZE = 4
GRAD_ACC = 2   # effective batch size = 8

LEARNING_RATE = 2e-4

MAX_INPUT_LENGTH = 768
MAX_TARGET_LENGTH = 192

WANDB_PROJECT = "highlightsum_bart_lora"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# -------------------------
# W&B Logging
# -------------------------
wandb.login()
wandb.init(project=WANDB_PROJECT, name="bart_lora_t4_transformers457")

# -------------------------
# Dataset
# -------------------------
dataset = load_dataset("knkarthick/highlightsum")

train_data = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
val_data = dataset["validation"].shuffle(seed=42).select(range(VAL_SAMPLES))

print(f"📊 Training samples: {len(train_data)}")
print(f"📊 Validation samples: {len(val_data)}")

# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Model + LoRA
# -------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
model.to(device)

# -------------------------
# Tokenization
# -------------------------
def tokenize_fn(example):
    enc = tokenizer(
        example["dialogue"],
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
    )

    labels = tokenizer(
        example["summary"],
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
    ).input_ids

    enc["labels"] = labels
    return enc

train_dataset = train_data.map(tokenize_fn, remove_columns=train_data.column_names)
val_dataset = val_data.map(tokenize_fn, remove_columns=val_data.column_names)

# -------------------------
# Data Collator
# -------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# -------------------------
# Training Arguments (Transformers 4.57)
# -------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,

    learning_rate=LEARNING_RATE,
    fp16=True,

    logging_steps=50,

    # ⚠️ Correct argument name for Transformers 4.57 → eval_strategy
    eval_strategy="epoch",
    save_strategy="epoch",

    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=4,

    report_to="wandb",
    run_name="bart_lora_highlightsum_t457",
)

# -------------------------
# Trainer
# -------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# -------------------------
# Train
# -------------------------
print("\n🚀 Starting training...")
trainer.train()

# -------------------------
# Save Model
# -------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n🎉 Model saved to {OUTPUT_DIR}")
