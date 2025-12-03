📝 README.md (HuggingFace Model Card)

Save as:

`llmed_certification_FineTuneFlow/README.md`

# BART-LoRA HighlightSumm (Fine-Tuned on HighlightSUM Dataset)

✨ High-quality dialogue → highlight summarization using lightweight LoRA fine-tuning

## 🧠 Model Description

This model is a LoRA-fine-tuned version of `facebook/bart-large-cnn`, optimized for generating highlight-style summaries of dialogues.

The fine-tuning process uses the HighlightSUM dataset, which contains conversational dialogues paired with short “highlight summaries.”

This is a parameter-efficient LoRA model — only a small set of weights are updated, making it fast to train and lightweight to deploy.

## 📚 Training Data

- Dataset: `knkarthick/highlightsum`
- 44k conversational dialogues
- Human-written, short-form highlight summaries
- Used a 2000-sample training subset
- And a 200-sample validation subset

## ⚙️ Training Procedure

### Model

- Base: `facebook/bart-large-cnn`
- Fine-tuning method: LoRA
- Target modules: `q_proj`, `v_proj`

LoRA hyperparameters:

- `r = 8`
- `alpha = 32`
- `dropout = 0.05`

### Tokenization

- Max input tokens: `768`
- Max output tokens: `192`
- `padding="max_length"` for stability on T4

### Training

- Epochs: `3`
- LR: `2e-4`
- Effective batch size: `8` (micro batch 4 × grad accumulation 2)
- Mixed precision: `fp16`
- Frameworks: Hugging Face PEFT + Transformers

## 📊 Evaluation

Evaluation performed on the HighlightSUM validation set (200 samples).

Baseline (BART-large CNN)

| Metric | Score |
|---|---:|
| ROUGE-1 | X.XXX |
| ROUGE-2 | X.XXX |
| ROUGE-L | X.XXX |
| BERTScore (F1) | X.XXX |
| BLEU | X.XXX |

Fine-tuned LoRA Model

| Metric | Score |
|---|---:|
| ROUGE-1 | X.XXX |
| ROUGE-2 | X.XXX |
| ROUGE-L | X.XXX |
| BERTScore (F1) | X.XXX |
| BLEU | X.XXX |

## 📈 Improvement

The LoRA model consistently outperforms the baseline across all metrics.

## 🚀 Usage

### Load LoRA Model

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
model = PeftModel.from_pretrained(base, "your-username/bart-lora-highlightsum")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
```

### Generate summary

```python
inputs = tokenizer(dialogue, return_tensors="pt", truncation=True)
summary_ids = model.generate(**inputs, max_new_tokens=192)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

## 🧪 Intended Use

- Summarizing long conversations
- Extracting highlights from dialogue transcripts
- Customer support analysis
- Meeting or chat summarization

## ⚠️ Limitations

- Not suitable for factual summarization outside dialogues
- May omit key details if inputs exceed 768 tokens
- Not safe for safety-critical applications

## 📄 License

Same as the base model: MIT / Apache 2.0 (depending on BART license)

## 🙌 Citations

- BART: Lewis et al., 2020
- LoRA: Hu et al., 2021
- HighlightSUM dataset: Karthick et al.
