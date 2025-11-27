
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL = "BART-large"
DATASET_PATH = "./highlightsum_train.jsonl"
OUTPUT_DIR = "./ft_outputs/BART-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05)
model = get_peft_model(model, lora_cfg)

ds = load_dataset("json", data_files={"train": DATASET_PATH})["train"]

def tokenize_fn(example):
    out = tokenizer(example["dialogue"], truncation=True, max_length=1024)
    labels = tokenizer(example["summary"], truncation=True, max_length=128).input_ids
    out["labels"] = labels
    return out

train_ds = ds.map(tokenize_fn, remove_columns=ds.column_names)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=0.0002,
    fp16=True,
    save_strategy="no",
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
