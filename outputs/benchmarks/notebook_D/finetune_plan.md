# Fine-tuning Plan (Auto-Generated)

This document summarizes top fine-tuning choices based on Notebook C.

### 1. BART-large
- Composite score: 1.0000
- ROUGE-L: 21.04%
- Inferred size: 0.4B
- Recommended method: **LoRA (PEFT) — encoder–decoder friendly**
- Hyperparameters: `{'epochs': 3, 'micro_batch_size': 8, 'lr': 0.0002}`

### 2. LLaMA-1B
- Composite score: 0.4514
- ROUGE-L: 16.05%
- Inferred size: 1B
- Recommended method: **LoRA or full fine-tune**
- Hyperparameters: `{'epochs': 3, 'micro_batch_size': 8, 'lr': 0.0002}`
