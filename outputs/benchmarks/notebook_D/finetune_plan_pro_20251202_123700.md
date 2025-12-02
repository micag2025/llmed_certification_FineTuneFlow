# Fine-tuning Plan (Auto-Generated)

Generated: 20251202_123700

### 1. BART-large
- Composite score: 1.2307
- ROUGE-L: 21.06%
- Inferred size: 0.4B
- Detected GPU mem (GB): 15.828320256
- Recommended method: **LoRA (PEFT) — encoder–decoder friendly**
- Hyperparameters: `{'epochs': 3, 'micro_batch_size': 8, 'lr': 0.0001}`

### 2. LLaMA-1B
- Composite score: 0.4632
- ROUGE-L: 21.21%
- Inferred size: 1B
- Detected GPU mem (GB): 15.828320256
- Recommended method: **LoRA or full fine-tune**
- Hyperparameters: `{'epochs': 3, 'micro_batch_size': 8, 'lr': 0.0001}`
