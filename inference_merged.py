# =====================================================
# Show Dialogue, Human Summary, and Merged Model Output
# for the first item of HighlightSum
# =====================================================

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# Config
# -------------------------
MERGED_MODEL_PATH = "/content/llmed_certification_FineTuneFlow/ft_outputs/bart_merged_highlightsum"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_INPUT_LENGTH = 768
MAX_NEW_TOKENS = 192
NUM_BEAMS = 4

# -------------------------
# Load dataset (first item)
# -------------------------
ds = load_dataset("knkarthick/highlightsum")
dialogue = ds["test"][0]["dialogue"]
human_summary = ds["test"][0]["summary"]

# -------------------------
# Load merged model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MERGED_MODEL_PATH,
    torch_dtype=DTYPE
).to(DEVICE)
model.eval()

# -------------------------
# Generate model summary
# -------------------------
inputs = tokenizer(
    dialogue,
    truncation=True,
    max_length=MAX_INPUT_LENGTH,
    return_tensors="pt"
).to(DEVICE)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        no_repeat_ngram_size=3,
        length_penalty=1.0, # Added missing length_penalty
        early_stopping=True
    )

merged_summary = tokenizer.decode(output[0], skip_special_tokens=True)

# -------------------------
# Create table
# -------------------------
df = pd.DataFrame([{
    "Dialogue": dialogue,
    "Human Gold Summary": human_summary,
    "Merged Model Summary": merged_summary
}])

df