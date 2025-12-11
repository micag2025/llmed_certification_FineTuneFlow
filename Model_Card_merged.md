# HuggingFace Model Card — BART-HighlightSum (Merged Model)

## BART-HighlightSum (Merged Model)
Fine-tuned BART-Large on the HighlightSum dialogue summarization dataset (Merged LoRA → Full Model)

**Author:** @dlaima  
**License:** MIT 
**Model type:** Seq2Seq Summarization  
**Base model:** facebook/bart-large-cnn  
**Dataset:** HighlightSum (dialogue summarization)  
**Finetuning method:** LoRA → merged into full FP16 BART  

## Model Summary

This model is a merged BART-Large fine-tuned on 2,000 training + 200 validation samples from the HighlightSum dataset.
It produces concise, accurate summaries of multi-turn dialogues.

- LoRA fine-tuning    
- LoRA weights merged into base BART    
- No PEFT required for inference    
- Lightweight, fast, and deployment-ready    

This version is recommended for production, as it scores highest among all variants (Baseline, LoRA, Merged).

## Performance
Evaluation on HighlightSum (Validation 200 samples)

| Metric | Baseline BART | LoRA Model | Merged Model |
|--------|---------------|------------|--------------|
| ROUGE-1 | 0.275 | 0.337 | 0.383 |
| ROUGE-2 | 0.090 | 0.152 | 0.179 |
| ROUGE-L | 0.204 | 0.252 | 0.301 |
| BERTScore (F1) | 0.163 | 0.298 | 0.335 |
| BLEU | 0.0052 | 0.0111 | 0.0014 |

### Conclusion

The merged model performs best, achieving the highest ROUGE-1, ROUGE-2, ROUGE-L and BERTScore among all variants.
It is therefore the recommended model for deployment, inference, and user-facing applications.

##  Example Input / Output

Screenshot of few items of the HighlightSum dataset showing the summary evaluations predictions of BART LoRa merged. In details, the above screenshot diplays the `Dialogue`, the `Human Summary` and the `Merged Model Summary` respectively. As it can be seen the `Merged Model` captures the intent, context, and key meaning with improved fluency and coherence.

![Screenshot_validation_predictions.csv](https://github.com/micag2025/llmed_certification_FineTuneFlow/blob/f3f644cf9f1b0e13115945a99aea3f3eb705c196/Screenshots/Screenshot_11-12-2025_131040_wandb.ai.jpeg)

## Intended Use

### Suitable for
- Dialogue summarization
- Customer service chat compression
- Meeting note extraction
- Educational tools

### Not suitable for
- Factual QA
- Domain-specific technical summaries without fine-tuning
- Safety-critical use

## How to Use

### Python Inference
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "dlaima/bart-highlightsum-merged"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """A: Hi Tom, are you busy tomorrow afternoon?
B: I think I am. Why?
A: I want to go to the animal shelter.
B: For what?
A: I'm getting a puppy for my son."""

inputs = tokenizer(text, return_tensors="pt", truncation=True)
summary = model.generate(**inputs, max_new_tokens=192)
print(tokenizer.decode(summary[0], skip_special_tokens=True))
```

## Training Details

- **Method:** LoRA (rank 8)
- **Model:** BART-Large
- **Batch size:** 8 (micro-batch 4 × grad-accumulation 2)
- **Epochs:** ~2.4 (capped by 2000 examples)
- **Max input length:** 768 tokens
- **Max summary length:** 192 tokens
- **Precision:** FP16
- **Optimizer:** AdamW
- **Learning rate:** 3e-4
- **Hardware:** NVIDIA T4

## 📚 Dataset: HighlightSum

A dataset of dialogue → summary pairs from multiple conversational sources.

- Multi-turn dialogues
- Short, medium, or long
- Realistic conversational structure
- Human-written summaries

### Subset used here:
- 2,000 samples for training
- 200 samples for validation

## Files Included in This Repo

| File | Description |
|------|-------------|
| pytorch_model.bin | Final merged FP16 BART model |
| config.json | Standard HuggingFace config |
| generation_config.json | Beam search config |
| tokenizer.json / tokenizer.model | Tokenizer files |
| README.md | This model card |

## Limitations & Recommendations

### Limitations
- May shorten overly long dialogues excessively
- Not designed for domain-specific jargon
- Occasionally omits rare names or details
- Not a factual QA model
- Can hallucinate minor details in complex dialogues

### Recommendations
- Use merged model for production
- Apply additional fine-tuning for domain-specific tasks
- For 100% reproducibility, fix random seeds and HF transformers version
- Consider quantization (INT8 or GGUF) for mobile deployment

## 🛠 Maintenance

This model will be updated as:
- Additional training data becomes available
- Larger LoRA variants are tested
- Better merging & evaluation pipelines are developed

## 📫 Contact

For questions, improvements, or collaboration, feel free to reach out via GitHub or HuggingFace (@dlaima).





