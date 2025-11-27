
# =====================================================
# Evaluate fine-tuned BART LoRA model on HighlightSUM
# with sample predictions
# =====================================================

!pip install -q datasets transformers rouge-score torch

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

# -------------------------
# Config
# -------------------------
MODEL_DIR = "./ft_outputs/bart_lora_highlightsum"
N_VAL = 200    # number of validation samples for quick evaluation
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
NUM_SAMPLES_TO_DISPLAY = 5  # number of sample predictions to show
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# -------------------------
# Load validation data
# -------------------------
dataset = load_dataset("knkarthick/highlightsum")["test"].select(range(N_VAL))
print(f"Loaded {len(dataset)} validation samples.")

# -------------------------
# Load fine-tuned model & tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)

# -------------------------
# ROUGE scorer
# -------------------------
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def compute_rouge(preds, refs):
    agg = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        for k in agg:
            agg[k] += scores[k].fmeasure
    n = len(preds)
    return {k: v / n * 100 for k, v in agg.items()}

# -------------------------
# Generate summaries
# -------------------------
preds = []
refs = dataset["summary"]

for text in dataset["dialogue"]:
    inputs = tokenizer(
        text,
        truncation=True,
        padding="longest",
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    ).to(device)
    output_ids = model.generate(**inputs, max_new_tokens=MAX_TARGET_LENGTH)
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    preds.append(pred)

# -------------------------
# Compute ROUGE
# -------------------------
scores = compute_rouge(preds, refs)
print("\n📊 ROUGE scores on validation set:")
for k, v in scores.items():
    print(f"{k}: {v:.2f}")

# -------------------------
# Display some sample predictions
# -------------------------
print(f"\n📝 Sample predictions (showing {NUM_SAMPLES_TO_DISPLAY} examples):\n")
for i in range(NUM_SAMPLES_TO_DISPLAY):
    print(f"--- Example {i+1} ---")
    print("Dialogue:\n", dataset[i]["dialogue"])
    print("\nReference Summary:\n", dataset[i]["summary"])
    print("\nPredicted Summary:\n", preds[i])
    print("------------------------------\n")
