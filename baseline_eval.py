# =====================================================
# Baseline Evaluation for BART (No LoRA)
# Purpose: Compare pre-finetuning performance
# =====================================================

import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
import wandb

# -------------------------
# Config
# -------------------------
BASE_MODEL = "facebook/bart-large-cnn"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_INPUT_LENGTH = 768
MAX_NEW_TOKENS = 192
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3
BATCH_SIZE = 8

PRED_CSV = "/content/llmed_certification_FineTuneFlow/metrics/baseline_predictions.csv"
METRICS_CSV = "/content/llmed_certification_FineTuneFlow/metrics/baseline_predictions_metrics.csv"

WANDB_PROJECT = "highlightsum_bart_lora"
WANDB_RUN = "baseline_evaluation"

print("🔥 Using device:", DEVICE)

# -------------------------
# Initialize W&B
# -------------------------
try:
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN)
    USE_WANDB = True
except:
    print("⚠️ W&B not available, continuing without logging.")
    USE_WANDB = False

# -------------------------
# Load tokenizer & model
# -------------------------
print("\n🔧 Loading BASE model:", BASE_MODEL)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE
).to(DEVICE)

model.eval()

# -------------------------
# Batch inference function
# -------------------------
def batch_generate(dialogues):
    preds = []

    for i in range(0, len(dialogues), BATCH_SIZE):
        batch = dialogues[i:i + BATCH_SIZE]

        tok = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            padding="longest",
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outs = model.generate(
                **tok,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                early_stopping=True
            )

        preds.extend([tokenizer.decode(o, skip_special_tokens=True) for o in outs])

    return preds

# -------------------------
# Load validation dataset
# -------------------------
print("\n📥 Loading validation dataset: 200 samples")

ds = load_dataset("knkarthick/highlightsum")["validation"]
N = min(200, len(ds))

dialogues = [ds[i]["dialogue"] for i in range(N)]
human = [ds[i]["summary"] for i in range(N)]

# -------------------------
# Generate baseline summaries
# -------------------------
print("\n🚀 Running baseline inference...")
preds = batch_generate(dialogues)

df = pd.DataFrame({
    "id": list(range(N)),
    "dialogue": dialogues,
    "human_summary": human,
    "model_summary": preds
})

os.makedirs(os.path.dirname(PRED_CSV), exist_ok=True)
df.to_csv(PRED_CSV, index=False)

print(f"📁 Saved baseline predictions → {PRED_CSV}")

# =====================================================
# Metrics
# =====================================================

print("\n📊 Computing metrics...")

# -------------------------
# ROUGE
# -------------------------
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge1_list, rouge2_list, rougeL_list = [], [], []

for pred, ref in zip(df['model_summary'], df['human_summary']):
    scores = scorer.score(ref, pred)
    rouge1_list.append(scores['rouge1'].fmeasure)
    rouge2_list.append(scores['rouge2'].fmeasure)
    rougeL_list.append(scores['rougeL'].fmeasure)

df["rouge1"] = rouge1_list
df["rouge2"] = rouge2_list
df["rougeL"] = rougeL_list

# -------------------------
# BERTScore
# -------------------------
P, R, F1 = bert_score(
    cands=df["model_summary"].tolist(),
    refs=df["human_summary"].tolist(),
    lang="en",
    rescale_with_baseline=True
)

df["bert_f1"] = F1.numpy()

# -------------------------
# BLEU
# -------------------------
references = [[ref.split()] for ref in df["human_summary"]]
candidates = [pred.split() for pred in df["model_summary"]]
bleu = corpus_bleu(references, candidates)

# -------------------------
# Save Metrics CSV
# -------------------------
df.to_csv(METRICS_CSV, index=False)

print(f"📁 Saved baseline metrics → {METRICS_CSV}")

# -------------------------
# Log to W&B
# -------------------------
if USE_WANDB:
    wandb.log({
        "baseline/rouge1": sum(rouge1_list) / len(rouge1_list),
        "baseline/rouge2": sum(rouge2_list) / len(rouge2_list),
        "baseline/rougeL": sum(rougeL_list) / len(rougeL_list),
        "baseline/bert_f1": float(df["bert_f1"].mean()),
        "baseline/bleu": bleu
    })

    wandb.finish()

print("\n✔ Baseline evaluation complete.")
