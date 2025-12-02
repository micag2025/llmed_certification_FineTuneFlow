# =====================================================
# Inference for BART-LoRA Highlight Summarizer
# Clean, GPU-optimized, LoRA-safe loading
# =====================================================


import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# -------------------------
# Config
# -------------------------
MODEL_DIR = "/content/llmed_certification_FineTuneFlow/ft_outputs/bart_lora_highlightsum"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_INPUT_LENGTH = 768        # matches training
MAX_NEW_TOKENS = 192          # matches evaluation script
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3
BATCH_SIZE = 8

OUTPUT_CSV = "/content/llmed_certification_FineTuneFlow/metrics/inference_predictions.csv"

print("🔥 Using device:", DEVICE)

# -------------------------
# Load tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Load model (merged OR LoRA separate)
# -------------------------
print("\n🔧 Loading model...")

try:
    # Try loading as a merged model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE
    )
    print("✅ Loaded merged model.")

except Exception as err:
    # If merged model fails, load base + LoRA adapter
    print("ℹ️ Merged model not found — loading base BART + LoRA adapter.")
    print("   Reason:", err)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/bart-large-cnn",
        torch_dtype=DTYPE
    )
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    print("✅ Loaded LoRA adapter on top of base model.")

model.to(DEVICE)
model.eval()


# -------------------------
# Generation helpers
# -------------------------
def generate_summary(text):
    """Single-sample inference."""
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding="longest",
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            length_penalty=1.0,
            early_stopping=True
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def batch_generate(dialogues):
    """Fast batch inference."""
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
# 1) Quick example
# -------------------------
sample_dialogue = (
    "A: Hi Tom, are you busy tomorrow afternoon?\n"
    "B: I think I am. Why?\n"
    "A: I want to go to the animal shelter.\n"
    "B: For what?\n"
    "A: I'm getting a puppy for my son."
)

print("\n=== Single Example ===")
print("➡️", generate_summary(sample_dialogue))


# -------------------------
# 2) Batch inference
# -------------------------
print("\n=== Batch inference on HighlightSUM validation (N=200) ===")

try:
    ds = load_dataset("knkarthick/highlightsum")["validation"]
    N = min(200, len(ds))

    dialogues = [ds[i]["dialogue"] for i in range(N)]
    human_summaries = [ds[i]["summary"] for i in range(N)]
    predictions = batch_generate(dialogues)

    df = pd.DataFrame({
        "id": range(N),
        "dialogue": dialogues,
        "human_summary": human_summaries,
        "model_summary": predictions
    })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"📁 Saved predictions to: {OUTPUT_CSV}")

except Exception as e:
    print("⚠️ Dataset could not load (offline).")
    print("Error:", e)

print("\n✔ Inference complete.")
