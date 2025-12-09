# =====================================================
# Evaluation + Export Script for BART-LoRA Highlights
# With full W&B logging (ROUGE + table)
# =====================================================
# =====================================================
# Fast Evaluation for BART-LoRA Highlight Summaries
# Batched ROUGE scoring + progress bar + safer padding
# =====================================================

import os
import wandb
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

# -------------------------
# Config
# -------------------------
MODEL_DIR = "/content/llmed_certification_FineTuneFlow/ft_outputs/bart_merged_highlighsum"
VALIDATION_SIZE = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_CSV = "/content/llmed_certification_FineTuneFlow/metrics/validation_predictions_merged.csv"

MAX_INPUT_LENGTH = 768
MAX_NEW_TOKENS = 192

WANDB_PROJECT = "highlightsum_bart_lora_eval_merged"

# -------------------------
# W&B logging
# -------------------------
wandb.login()
run = wandb.init(project=WANDB_PROJECT, name="bart_lora_validation_eval")

# -------------------------
# Load validation dataset
# -------------------------
dataset = load_dataset("knkarthick/highlightsum")["validation"].select(range(VALIDATION_SIZE))
print(f"Loaded {len(dataset)} validation samples.")

# -------------------------
# Load model & tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# -------------------------
# ROUGE scorer
# -------------------------
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

results = []

# -------------------------
# Evaluation Loop
# -------------------------
print("🔍 Running evaluation...")

for idx, sample in enumerate(dataset):

    encoded = tokenizer(
        sample["dialogue"],
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.0
        )

    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    human = sample["summary"]

    scores = scorer.score(human, pred)

    results.append({
        "sample_id": idx,
        "dialogue": sample["dialogue"],
        "human_summary": human,
        "model_summary": pred,
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    })

# -------------------------
# Convert to DataFrame
# -------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Validation predictions exported to {OUTPUT_CSV}")

# -------------------------
# Average ROUGE
# -------------------------
avg_r1 = df["rouge1"].mean()
avg_r2 = df["rouge2"].mean()
avg_rL = df["rougeL"].mean()

print(f"\n📊 Avg ROUGE:")
print(f"ROUGE-1: {avg_r1:.4f}")
print(f"ROUGE-2: {avg_r2:.4f}")
print(f"ROUGE-L: {avg_rL:.4f}")

# -------------------------
# Log metrics to Weights & Biases
# -------------------------
wandb.log({
    "avg_rouge1": avg_r1,
    "avg_rouge2": avg_r2,
    "avg_rougeL": avg_rL,
})

# Log entire validation table
wandb.log({"validation_predictions": wandb.Table(dataframe=df)})

# Log a few sample predictions
for i in [0, 42, 123, 199]:
    wandb.log({
        f"sample_{i}_dialogue": df.loc[i, "dialogue"],
        f"sample_{i}_pred": df.loc[i, "model_summary"],
        f"sample_{i}_human": df.loc[i, "human_summary"],
    })

# -------------------------
# Finish
# -------------------------
run.finish()

print("\n🏁 Evaluation + W&B logging complete.")
