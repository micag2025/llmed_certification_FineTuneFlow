# =====================================================
# Fast Evaluation for BART-LoRA Highlight Summaries
# Batched ROUGE scoring + progress bar + safer padding
# =====================================================

import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

# -------------------------
# Config
# -------------------------
MODEL_DIR = "/content/llmed_certification_FineTuneFlow/ft_outputs/bart_lora_highlightsum"
OUTPUT_CSV = "/content/llmed_certification_FineTuneFlow/metrics/validation_predictions.csv"

VALIDATION_SIZE = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_INPUT_LENGTH = 768
MAX_NEW_TOKENS = 192
BATCH_SIZE = 8       # 🔥 batched inference = much faster

print(f"Using device: {DEVICE}")

# -------------------------
# Load validation
# -------------------------
dataset = load_dataset("knkarthick/highlightsum")["validation"].select(range(VALIDATION_SIZE))
print(f"Loaded {len(dataset)} validation samples.")

dialogues = [ex["dialogue"] for ex in dataset]
human_summaries = [ex["summary"] for ex in dataset]

# -------------------------
# Load model + tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# -------------------------
# ROUGE scorer
# -------------------------
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# -------------------------
# Batched inference
# -------------------------
predictions = []

for i in tqdm(range(0, len(dialogues), BATCH_SIZE), desc="Generating"):
    batch = dialogues[i:i + BATCH_SIZE]

    inputs = tokenizer(
        batch,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding="longest",
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )

    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outs]
    predictions.extend(decoded)

# -------------------------
# Compute ROUGE
# -------------------------
rouge1_list, rouge2_list, rougeL_list = [], [], []

for human, pred in zip(human_summaries, predictions):
    scores = scorer.score(human, pred)
    rouge1_list.append(scores["rouge1"].fmeasure)
    rouge2_list.append(scores["rouge2"].fmeasure)
    rougeL_list.append(scores["rougeL"].fmeasure)

# -------------------------
# Save CSV
# -------------------------
df = pd.DataFrame({
    "dialogue": dialogues,
    "human_summary": human_summaries,
    "model_summary": predictions,
    "rouge1": rouge1_list,
    "rouge2": rouge2_list,
    "rougeL": rougeL_list,
})
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved evaluation results to {OUTPUT_CSV}")

# -------------------------
# Average ROUGE
# -------------------------
print("\n📊 Average ROUGE on validation:")
print(f"ROUGE-1: {df['rouge1'].mean():.3f}")
print(f"ROUGE-2: {df['rouge2'].mean():.3f}")
print(f"ROUGE-L: {df['rougeL'].mean():.3f}")

# -------------------------
# Inspect one example
# -------------------------
example_id = 42
print("\n📝 Dialogue:")
print(dialogues[example_id])

print("\n🟡 Model summary:")
print(predictions[example_id])

print("\n🟢 Human summary:")
print(human_summaries[example_id])
