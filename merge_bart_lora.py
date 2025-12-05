import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# -------------------------
# Paths
# -------------------------
BASE_MODEL = "facebook/bart-large-cnn"
LORA_PATH = "/content/llmed_certification_FineTuneFlow/ft_outputs/bart_lora_highlightsum"
OUTPUT_PATH = "/content/llmed_certification_FineTuneFlow/ft_outputs/bart_merged_highlightsum"

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("🔄 Loading base BART model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

print("🔌 Attaching LoRA adapter...")
lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("🧬 Merging LoRA → base model (this may take a moment)...")
merged_model = lora_model.merge_and_unload()  # applies LoRA deltas + removes adapters

print("💾 Saving merged model...")
merged_model.save_pretrained(OUTPUT_PATH)

print("📁 Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)  # safer (matches training tokenizer)
tokenizer.save_pretrained(OUTPUT_PATH)

print("\n✅ Merge complete!")
print(f"🚀 Final merged model saved to:\n{OUTPUT_PATH}")
