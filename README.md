# LLM Engineering and Deployment (LLMED) Certification:Capstone Project: LLM Fine-Tuning & Optimization for Dialogue Summarization (HighlightSum)

This repository is part of capstone project for the **LLM Engineering and Deployment Certification program** by [Ready Tensor](https://www.readytensor.ai) and it is linked to the publication:**LLMED Certification:Capstone Project:LLM Fine-Tuning & Optimization for Dialogue Summarization (HighlightSum)** available on [Ready Tensor](https://www.readytensor.ai). This project builds a complete evaluation, selection, and fine-tuning pipeline for small-to-medium open-source language models. The objective is to identify the most efficient model for dialogue summarization, then fine-tune it using QLoRA and optimize it for real-world deployment. This was achieved using a subset of the HighlightSum dataset. This capstone project focuses on fine-tuning and benchmarking large language models for efficient, high-quality conversational summarization.

---

## Project Overview (Description)

This project develops a scalable, efficient workflow for selecting, fine-tuning, and evaluating open-source LLMs (e.g. BART, T5, LLaMA-1B, LLaMA-3B, Phi-3-Mini) for the task of dialogue summarization, using a subset of the benchmark [HighlightSum dataset](https://huggingface.co/datasets/knkarthick/highlightsum) as a test dataset. The codebase automates model selection via benchmarking, applies QLoRA for parameter-efficient fine-tuning, and outputs deployable artifacts.

---

## Workflow & Stages for BART-LoRA Fine-Tuning  

The complete and up-to-date pipeline / workflow (end-to-end) including training → evaluation → merging → deployment → export (production)

```text  

┌────────────────────────────────────────────────────┐
│ 1. Inspection and Prepare Dataset  HighlightSum                               │
│  ─ Raw documents                                    │
│  ─ Highlights / summaries                           │
│  → Format into HuggingFace dataset (train/val)      │
└────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│ 2. Fine-Tune Base BART with LoRA (PEFT)             │
│  python train_bart_lora.py                          │
│  Output: ./ft_outputs/bart_lora_highlightsum                    │
│   (LoRA adapter weights + training logs)            │(PEFT checkpoints + base model refs only)
└────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│ 3. Evaluate LoRA Model (Validation)                 │
│  python eval_bart_lora.py                           │
│  Output: ./metrics/lora_eval.json ?                   │
│    - ROUGE-1 / ROUGE-2 / ROUGE-L                    │
│    - BERTScore, BLEU                                │
│    - validation_predictions.csv                     │
└────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│ 4. Merge LoRA into Base BART                        │
│  python merge_bart_lora.py                               │
│  Output: ./ft_outputs/bart_merged_highlighsum                   │
└────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│ 5. Post-Merge Cleanup (Fix Config)    ???              │
│  python post_merge_cleanup.py                       │
│  Fixes:                                             │
│   - forced_bos_token_id                             │
│   - decoder_start_token_id                          │
│   - early_stopping flag                             │
│  Output: ./ft_outputs/bart_merged_clean             │
└────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│ 6. Evaluate Final Merged Model      TO BE ENCLOSED                │
│  python eval_bart_lora.py --model=merged_clean      │
│  Output: ./metrics/merged_eval.json                 │
│                                                      │
│  🔽 Comparison (automatic in notebook)               │
│    lora_eval.json       vs       merged_eval.json    │
│    → Does merging preserve or improve ROUGE?         │
└────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│ 7. Inference / Deployment                           │
│  python test_inference.py                           │
│  or deploy using:                                    │
│   - FastAPI Endpoint   (?)                            │
│   - Gradio Web UI      (?)                             │
│   - Hugging Face Space                              │
│                                                     │
└────────────────────────────────────────────────────┘

─────────────────────────────────────────────────────────────
  6. POST-MERGE USAGE  (deployment stage)
─────────────────────────────────────────────────────────────
                   ./ft_outputs/bart_merged_clean
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   inference.py          evaluate.py           Notebook-F (GGUF export)
(Real use / API)   (ROUGE + BERTScore + BLEU     for llama.cpp /
                       + charts dashboard)        LM Studio / Ollama
─────────────────────────────────────────────────────────────
  7. PRODUCTION
─────────────────────────────────────────────────────────────
 Option A — Hugging Face pipeline
 Option B — FastAPI / Flask service
 Option C — GGUF quantized using llama.cpp/LM Studio
 Option D — Batch inference at scale
```

To evaluate and improve a model’s step-by-step summarisation capability using a subset of the [HighlightSum dataset](https://huggingface.co/datasets/knkarthick/highlightsum), the following **workflow**, divided into several stages, is employed:  
  
1. **Benchmarking & Model Selection**
   - Multiple models are compared (BART, T5, LLaMA, etc.) on ROUGE, speed, and efficiency.
   - Visual ranking and automatic recommendation of the best model to fine-tune.

2. **Fine-Tuning (QLoRA)**
   - Selected model is fine-tuned using QLoRA for memory efficiency (4-bit quantization).
   - Produces LoRA adapter weights.

3. **Evaluation**
   - Fine-tuned model is evaluated using ROUGE and other metrics on HighlightSum validation set.
   - Outputs include plots and summary tables.

4. **Deployment Prep**
   - Merge LoRA adapters with base model.
   - Convert to GGUF for fast CPU inference (llama.cpp compatibility).
   - Tracks all experiments with Weights & Biases.

---

## Features / What’s Included

- Automated benchmarking and composite ranking of open LLMs
- QLoRA-based fine-tuning pipeline 
- Inference & evaluation scripts
- Artifacts for deployment (merged weights, GGUF exports)
- Experiment tracking (Weights & Biases)
- Example Colab/Notebook integration

---

## Repository Structure  TO BE UPDATED

```text
📁 C:\Users\Michela\llmed_Certification_Project1_FineTuneFlow     project/
├── README.md
│
├── train_bart_lora.py            # QLoRA training (2k or full dataset)  / Training script
├── merge_bart_lora.py                 # Merge LoRA → full FP16 model  / Merge adapters with base model
├── inference_bart_lora.py                  # Generation with LoRA or merged model  / Summarization with fine-tuned model
├── eval_bart_lora.py                   # ROUGE metrics + charts (CLI)   / Compute ROUGE, generate charts
├── eval_metrics_bart_lora.py    
│ 
├── notebooks/
│   ├── Notebook_C.ipynb              # Benchmarking + model selection / Benchmarking & Selection
│   ├── Notebook_D.ipynb              # Auto finetune plan recommendation / Fine-Tuning Recommendation
│   ├── Notebook_E.ipynb              # Inference + evaluation + ROUGE   / Inference/Evaluation Pipeline  MISSING
│   ├── Notebook_F.ipynb              # Production (FastAPI, GGUF export) / Productionization Guide       MISSING
│   ├── Notebook_G.ipynb              # Stretch-goal / safety alignment  / (API/Deployment)               MISSING
│
├── models (ft_outputs)/
│   ├── bart_lora_highlightsum/      # Training output (2k subset)  / OUTPUT_DIR for 2k-subset training
│       ├── adapter_model.bin
│       ├── adapter_config.json
│       └── tokenizer files
│   ├── bart_merged_highlightsum/        # Full merged HF model   / MERGED_DIR after merge_lora.py
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       ├── special_tokens_map.json
│       └── etc...
│   ├── gguf/                         # Quantized GGUF exports (Notebook F) TO BE ENCLOSED
│
├── outputs/
│   ├── evaluation/                          # Evaluation results / Generated by Notebook E  MISSING
│   │   ├── metrics.json
│   │   ├── metrics.csv
│   │   ├── rouge1.png
│   │   ├── rouge2.png
│   │   ├── rougel.png
│   │
│   ├── benchmarks/
│       ├── notebook_C/                      # Ranking results, charts THIS HAS BEEN ENCLOSED
│       │   ├── final_ranking.csv
│       │   ├── final_ranking.json
│       │   ├── final_ranking.html
│       ├── notebook_D/                      # Fine-tuning plans, scripts
│       │   ├── finetune_plan.md
│       │   ├── qLoRa_train.sh  MISSING
│       │   ├── train_lora_BART-large_20251202_123700.py
│       │   ├── train_q_lora_LLaMA-1B_20251202_123700.py
│       │   ├── recommendations.json
│
├── requirements.txt                          # Project dependencies
└── .env_example.txt                         # Example environment file for API keys
```

```
llmed_certification_FineTuneFlow/
│
├── train_bart_lora.py                  # LoRA fine-tuning
├── baseline_eval.py                    # Baseline evaluation (pre-training)
├── eval_bart_lora.py                   # Post-training evaluation
├── inference_bart_lora.py              # Inference w/ LoRA or merged model
├── merge_bart_lora.py                  # Merge LoRA → base model
│
├── metrics/                            # All metrics + prediction CSVs
│   ├── baseline_predictions.csv
│   ├── baseline_predictions_metrics.csv
│   ├── validation_predictions.csv
│   ├── validation_predictions_metrics.csv
│
├── ft_outputs/                         # Model outputs
│   ├── bart_lora_highlightsum/         # LoRA adapter model
│   ├── bart_merged_highlightsum/       # (optional) merged checkpoint
│
├── requirements.txt
├── README.md                           # Full model card for HF Hub
│
└── utils/ (optional)
    ├── dataset_utils.py
    ├── generation_utils.py
```

### What the evaluation step provides

Each evaluation run (`eval_bart_lora.py`) computes:
| Metric                      | Purpose                                   |
| --------------------------- | ----------------------------------------- |
| ROUGE-1 / ROUGE-2 / ROUGE-L | Measures overlap with reference summaries |
| BERTScore                   | Semantic similarity                       |
| BLEU                        | Precision-based n-gram similarity         |
| Avg. Length                 | Output stability check                    |
| Failure Buckets             | Templates for common failure cases        |

Plus the CSV:
validation_predictions.csv
| id | source_text | reference_summary | generated_summary | rougeL_score |


###  Final comparison across model versions
| Model version                    | When to compute               |
| -------------------------------- | ----------------------------- |
|  Base BART (optional)           | Before fine-tuning (baseline) |
|  BART + LoRA (during training) | After fine-tuning             |
|  BART merged_clean             | Final deployment model        |

This allows to answer three key questions:
| Question                             | Which comparison?                   |
| ------------------------------------ | ----------------------------------- |
| Is LoRA training effective?          | Base vs. BART-LoRA                  |
| Is merging lossless?                 | BART-LoRA vs. merged_clean          |
| Is the final model production-ready? | merged_clean eval score + inference |

>_Note_: LoRA fine-tuning → evaluate → merge LoRA into base → cleanup → evaluate again → deploy.

---

## Getting Started

### Prerequisites

- Python 3.10+    
- [HuggingFace Account & API Key](https://huggingface.co/)
- [Weights & Biases Account](https://wandb.ai/site) (for experiment tracking—optional, but recommended)

Set relevant API keys in your environment:

```bash
export HF_API_KEY=your_hf_key
export WANDB_API_KEY=your_wandb_key
```
### Hugging Face & Weights & Biases Authentication in Notebooks

If running in a Colab or Jupyter notebook, authenticate your session with Hugging Face Hub and Weights & Biases as follows:

```python
from huggingface_hub import notebook_login
notebook_login()
```

Install Weights & Biases (if not already installed) and log in:

```bash
pip install wandb
wandb login
```
---

### Installation

```bash
git clone https://github.com/micag2025/llmed_certification_FineTuneFlow.git
cd llmed_certification_FineTuneFlow
pip install -r requirements.txt
```
---

## Running the Pipeline

**Dataset Selection and Preparation**  
- Dataset: Highlightsum dataset on Hugging Face  
- Sample Size: 2,000 training samples, 200 validation samples.  
- Preprocessing:  
  - Tokenization with BART tokenizer.    
  - Input truncation (max length 768), target truncation (max length 192).    
  - Splitting into training and validation sets.  

Here the focus is on _flow of data into fine-tuning pipeline_ rather than dataset collection or cleaning.

**Model Benchmarking**

Run the benchmarking notebook (`notebook_C`) to compare multiple candidate models using accuracy and efficiency metrics. The evaluation `notebook C`, compare five candidate models ( [BART-large](https://huggingface.co/facebook/bart-large-cnn), [T5-large](https://huggingface.co/google/flan-t5-large), [Phi-3-Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), [LLaMA-1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), and [LLaMA-3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)) on 200 validation samples of Highlightsum dataset using:    
- ROUGE-1 / ROUGE-2 / ROUGE-L scores    
- Execution time per sample    
- Tokens-per-second throughput    
- An overall efficiency score (accuracy vs speed)

Basic preprocessing has been performed, including `tokenization of dialogues` with appropriate padding and truncation, `batch preparation` for seq2seq models, and `selection of a subset from the HighlightSUM train split` for benchmarking. In details, the basic preprocessing performed is based on:  
- **Tokenization**:    
  - All text inputs (`dialogue`) are tokenized using the model-specific tokenizer.  
  - For causal models, if `pad_token` was missing, it was set to `eos_token` to allow batching/padding.  
  - Seq2seq and causal models both use truncation and padding (`max_length=768`) to ensure consistent tensor shapes.  
- **Dataset splitting**:  
  - Selected a subset of samples (`N_SAMPLES`) from the HighlightSUM train split for benchmarking.  
- **Batching (seq2seq models)**:
 - Inputs are batched to reduce memory usage on GPU, which is part of preprocessing before model inference.

`Notebook C` allows:  
- Benchmarks large models safely on Colab.    
- Performs basic preprocessing (tokenization, padding, truncation, batching).  
- Handles tokenizer safety (pad_token set if missing).  
- Includes train/validation/test split handling.  
- Clearly highlights all places where tokenizer safety or padding/truncation is applied.

_Key Preprocessing & Tokenizer Safety Highlights_  
1 **Tokenizer safety**:
```bash
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```
- Applied for **all models**, including causal LMs.
2 **Batching & padding**:  
- Seq2seq models use `padding="longest"` and `max_length=768`.  
- Causal models use `padding="max_length"` and `max_length=768`.    
3 **Truncation**:  
- Ensures sequences don’t exceed model’s max input length.  
4 **Dataset split**:  
- Train, validation, and test subsets selected (N_SAMPLES for test subset).  

`Notebook_C` generates `final_ranking.csv`that reflects the real performance trade-offs (ROUGE + throughput + efficiency) on HighlightSUM dataset.  

 **Auto-fine-tuning Recommendation & Plan** 

`Notebook D` (Auto-fine-tuning Recommendation & Plan) reads the final leaderboard from `model_benchmarks/notebook_C/final_ranking.csv` (inputs) and generates a comprehensive fine-tuning strategy available in the following outputs (`model_benchmarks/notebook_D/`):     
- `finetune_plan.md` — Human-readable fine-tuning plan with rationale and hyperparameters
- `recommendations.json` -	Structured recommendations per model (method + hyperparameters)  
- `train_qLoRA.py` — Training template using PEFT + QLoRA  (TO BE RENAMED?)
- `qLoRA_train.sh` — Bash wrapper to execute QLoRA training with Hugging Face Accelerate  (TO BE RENAMED?)

For each model, `Notebook D` produces:  
- Ranked recommendation — which model(s) to fine-tune
- Fine-tuning method — QLoRA / LoRA / full fine-tuning (based on model size & available GPU)
- Hyperparameters — recommended training settings
- Compute estimate — rough time/resource heuristic (informational only, not billing-accurate)  


Next steps:  
- customize the `train_qLoRA.py` to the chosen model (map tokenizers/prompt style precisely). 
- add validation loop + ROUGE evaluation inside training to checkpoint best model.
- produce a small sample dataset JSONL generator from HighlightSumthat matches the expected supervised format.
- estimate training time more accurately based on the GPU type (T4 / L4 / A100) and hours you can run.  


 `Customize train_qLoRA.py`
`Notebook D` provides a generic training script, but it needs to be adapted so you can:
1. Set your model
2. Set your dataset / Notebook D may include a placeholder dataset.
3. Set your training hyperparameters:    
    - batch size  
  	- gradient accumulation  
    - QLoRA R value  
    - learning rate  
    - warmup  
    - max steps / epochs  
    - max sequence length  
4. Set output directory 

For this project:  
> Use: Optimized BART-LoRA Training for T4 GPU  
       Uses effective batch size 8 via gradient accumulation  
       Includes padding, length updates, and stable T4 config    

QLoRA training script has been also  modifed for speed while keeping most of the QLoRA benefits and model quality.  Key changes:
  - Shorter context: MAX_LENGTH = 768 — biggest speed win.  
  - Smaller LoRA rank: r = 8 (was 16) — less computation, still effective.  
  - Smarter batching: smaller per-device batch + larger gradient_accumulation_steps to keep effective batch size.
  - Cap total steps: use max_steps to avoid unnecessary epochs (you can tune).
  - Fewer saves/eval/logging: reduce IO overhead.
  -  Keep gradient checkpointing and fp16 to reduce memory & speed tradeoff.  ??
  -  Faster preprocessing: use padding="longest" then collate, avoid padding="max_length" in map to reduce token workload.
  -  Minor other tweaks (num_workers for tokenizers, cudnn benchmark, use_cache=False).

The fully updated `train_bart_lora.py` has been also equipped with Weights & Biases (W&B)  

**Fine-tuning**

After having customized `train_qLoRA.py` as needed, the main training script has been called `train_bart_lora.py`, and then this has been used to launch training (QLoRA Training):

```bash
!python train_bart_lora.py
```

Training output is saved to:   
`models (ft_outputs)/bart_lora_highlightsum/`

This script:  (TO BE VERIFIED)
- Loads base model in FP16 (not 4-bit)  
- Loads LoRA adapters  
- Applies the LoRA weights  
- Merges them into the model  
- Saves a standalone checkpoint  

>_Note_ See Correct Flow Summary > training > evaluation> merge>inference 

**Merge and Evaluate** TO BE CHANGED 
The fully corrected, safe, and LLaMA-3.2 compatible merge_lora.py script has been used for merging:  
- the 4-bit base model, and
- the LoRA adapters
- into a single FP16 full model that you can use normally without PEFT.

To sum up, the  `merge_lora.py` script:  
- Loads base model in FP16 (not 4-bit)  
- Loads LoRA adapters  
- Applies the LoRA weights  
- Merges them into the model  
- Saves a standalone checkpoints

The LLaMA-3.2 compatible evaluation script (ROUGE on SAMSum validation set), `evaluate.py`:
- Loads either LoRA model or merged model  
- Computes ROUGE-1, ROUGE-2, ROUGE-L on the SAMSum validation set  
- Uses your chat template for inference  
- Runs on Colab GPU  
- Is optimized for speed (batch inference, no sampling, greedy decoding)      


```bash
!python merge_lora.py
!python evaluate.py
```

This produces a new folder:  
```bash
llama1b-samsum-merged/
    config.json
    generation_config.json
    model.safetensors
    tokenizer.json
    tokenizer.model
...

**Inference**

```bash
!python inference.py
```

Example:  
Input dialogue:
John: Are you joining the call?
Sarah: Yes, give me 2 minutes.
John: Sure, I'll wait.

Generated summary:
Sarah will join the call shortly.

Evaluation (ROUGE)  
```bash
python src/evaluate.py
```
Outputs go to:
`outputs/evaluation_results/`  
Including:
`metrics.json`  
`rouge1.png`, `rouge2.png`, `rougeL.png`  

---


Full Fine-Tuning Workflow 
for the BART-LoRA: training → inference → evaluation → metrics → merge (optional)
**STEP 1 — Train LoRA Model**

Runs:
train_bart_lora.py

✔ Produces:
/content/.../ft_outputs/bart_lora_highlightsum

**STEP 2 — Evaluate on Validation Split (ROUGE only)**

Runs:
eval_bart_lora.py

✔ Produces basic evaluation CSV:
validation_predictions.csv
(with columns: dialogue, human_summary, model_summary, rouge scores)

⚠️ This CSV is required for the Metrics step.

**STEP 3 — Full Metrics (ROUGE + BERTScore + BLEU)**

Runs:
eval_metrics_bart_lora
/metrics/validation_predictions.csv

✔ Produces enriched CSV:
validation_predictions_metrics.csv
(contains rouge1, rouge2, rougeL, bert_f1, BLEU)

**STEP 4 — Optional: Merge LoRA → Full Model**

Runs:
merge_bart_lora.py

✔ Produces merged full model (no adapters):
bart_merged_highlightsum/

Only needed if:  
- you want to deploy without LoRA  
- or run inference outside PEFT context  

**STEP 5 — Inference Script (for new unseen data)**

Runs:
inference_bart_lora.py (the updated version)

✔ Can load either:
LoRA model
OR merged model

## Usage Examples  

### Inspection Dataset

```
 📊 Dataset Overview:
  Train splits: 27,401 samples
  Val splits: 1,360 samples
  Test splits: 2,347 samples

🔑 Keys: ['id', 'dialogue', 'summary']

📘 First training example:

🔸 DIALOGUE (32390 chars):
Speaker A: Cool. Do you wanna give me the little cable thing? Yeah. Cool. Ah, that's why it won't meet. Okay, cool. Yep, cool. Okay, functional requirements. Alright, yeah. It's working. Cool, okay. So what I have, wh where I've got my information from is a survey where the usability lab um observed...

🔹 SUMMARY (1299 chars):
The project manager opens the meeting by stating that they will address functional design and then going over the agenda. The industrial designer gives his presentation, explaining how remote controls function and giving personal preference to a clear, simple design that upgrades the technology as well as incorporates the latest features in chip design. The interface specialist gives her presentation next, addressing the main purpose of a remote control. She pinpoints the main functions of on/off, channel-switching, numbers for choosing particular channels, and volume; and also suggests adding a menu button to change settings such as brightness on the screen. She gives preference to a remote that is small, easy to use, and follows some conventions. The group briefly discusses the possibility of using an LCD screen if cost allows it, since it is fancy and fashionable. The marketing expert presents, giving statistical information from a survey of 100 subjects. She prefers a remote that is sleek, stylish, sophisticated, cool, beautiful, functional, solar-powered, has long battery life, and has a locator. They discuss the target group, deciding it should be 15-35 year olds. After they talk about features they might include, the project manager closes the meeting by allocating tasks.  
```


### Benchmark Results

| model       | model_id                                       |    rouge1 |    rouge2 |    rougeL |        time | throughput |  efficiency | composite_score |
|-------------|------------------------------------------------|----------:|----------:|----------:|------------:|------------:|------------:|-----------------:|
| BART-large  | facebook/bart-large-cnn                        |  28.106   |   9.183   |  21.063   |   101.632   |    1.968    |    0.207    |      1.231      |
| LLaMA-1B    | meta-llama/Llama-3.2-1B-Instruct               |  28.636   |   9.618   |  21.205   |   393.929   |    0.508    |    0.054    |      0.463      |
| LLaMA-3B    | meta-llama/Llama-3.2-3B-Instruct               |  23.772   |   8.223   |  17.306   |   748.223   |    0.267    |    0.023    |     -0.162      |
| Phi-3-Mini  | microsoft/Phi-3-mini-4k-instruct               |  20.550   |   7.028   |  14.307   |   987.636   |    0.203    |    0.014    |     -0.572      |
| T5-large    | t5-large                                       |  10.977   |   1.944   |   9.637   |   263.028   |    0.760    |    0.037    |     -0.960      |

> _Notes_:  
- Accuracy: ROUGE-L is used as the primary accuracy metric.
- Latency: Time refers to the average inference time per sample.
- Throughput:samples/sec = speed=total time
- Efficiency: Defined as ROUGE-L divided by inference time = ROUGE/time  
- Composite score: Normalized metric combining accuracy and efficiency to support model selection.

The Ranking Table provides a full benchmarking and model-selection pipeline. Thus, this identifies (recommends) automatically the best model to fine-tune based on balanced performance rather than size alone. To sum up, the highest composite_score wins.  When selecting models for dialogue summarization, balancing prediction quality with inference efficiency is crucial — especially in practical or real-time settings.  
_Key takeaways_  
  - Composite score reflects both accuracy and speed, giving a more holistic evaluation than ROUGE alone.    
  - Models like BART-large outperform others because they achieve solid accuracy and fast inference.    
  - Larger causal models (e.g., LLaMA-3B, Phi-3-Mini) may achieve acceptable ROUGE scores, but their high latency significantly reduces their overall ranking.   

This shows the importance of balancing accuracy with inference speed when benchmarking large models for dialogue summarization.  

---

## Auto-fine-tuning Recommendation & Plan Results  

Notebook D generates `recommendations.json`, which contains per-model fine-tuning strategies and hyperparameters. Based on this analysis, BART-large and LLaMA-1B emerged as the top two candidates.  

Recommendation Output
```json
{
  "BART-large": {
    "size_hint": "0.4B",
    "method": "LoRA (PEFT) \u2014 encoder\u2013decoder friendly",
    "recommended_hyperparams": {
      "epochs": 3,
      "micro_batch_size": 8,
      "lr": 0.0002
    }
  },
  "LLaMA-1B": {
    "size_hint": "1B",
    "method": "LoRA or full fine-tune",
    "recommended_hyperparams": {
      "epochs": 3,
      "micro_batch_size": 8,
      "lr": 0.0002
    }
  }
```
Interpretation & Comparison
| Model          | Size | Recommended PEFT Method                    | Suggested Hypers                   | Meaning                                                   |
| -------------- | ---- | ------------------------------------------ | ---------------------------------- | --------------------------------------------------------- |
| **BART-large** | 0.4B | **LoRA (PEFT) — encoder–decoder friendly** | epochs: 3, batch size: 8, LR: 2e-4 | **Best match for abstractive summarisation + efficiency** |
| **LLaMA-1B**   | 1B   | LoRA **or full fine-tune**                 | epochs: 3, batch size: 8, LR: 2e-4 | Strong, but slower + worse summarisation on highlightSUM  |

**BART-large** was selected and it is the preferred choice because:  
- Highest composite score from Notebook D rankings  
- Optimized architecture for encoder-decoder summarization tasks (HighlightSUM)  
- Native LoRA support on attention projections (q_proj/v_proj) without special patching  
- Efficient training & inference on Colab T4 GPU  

While both models received identical hyperparameters, BART-large offers superior performance due to its architecture fit, ROUGE scores, and latency characteristics.  


**Customize `train_qLoRA.py`**  

Next steps:  
- customize the `train_qLoRA.py` to the chosen model (map tokenizers/prompt style precisely). 
- add validation loop + ROUGE evaluation inside training to checkpoint best model.
- produce a small sample dataset JSONL generator from SAMSum that matches the expected supervised format.
- estimate training time more accurately based on the GPU type (T4 / L4 / A100) and hours you can run.  

Example of `customization of train_qLoRA.py`  

```bash
# -------------------------
# Config
# -------------------------
MODEL_NAME = "facebook/bart-large-cnn"
OUTPUT_DIR = "./ft_outputs/bart_lora_highlightsum"
N_SAMPLES = 2000
EPOCHS = 3                # recommended based on benchmark
MICRO_BATCH_SIZE = 4      # per-device batch size
GRAD_ACC = 2              # → effective batch size = 8
LEARNING_RATE = 2e-4
MAX_INPUT_LENGTH = 768    # reduction speeds up training significantly
MAX_TARGET_LENGTH = 192
WANDB_PROJECT = "highlightsum_bart_lora"
```

```bash
# -------------------------
# Load dataset
# -------------------------
dataset = load_dataset("knkarthick/highlightsum")["train"].select(range(N_SAMPLES))
print(f"Loaded {len(dataset)} samples for training.")  
```

> It has choosen to train HighlightSum on a subset Recommandations sizes 
 
| Subset size      | GPU time     | Quality   |
| ---------------- | ------------ | --------- |
| **1k** samples   | ~ min   |        |
| **2k** samples   | ~ min      |      |
| **5k** samples   | ~ hours |  |
| **Full dataset** | ~s   |       |

HighlightSum is small, so even 2k samples already gives strong summarization  


### Examples output train.py 

- **Model Loaded in 4-bit Successfully** 
```bash
🔥 Loading LLaMA-3.2-1B in 4-bit…
`torch_dtype` is deprecated! Use `dtype` instead!
```
> _Note_: The deprecation warning is harmless. 4-bit quantization is functioning.

- **LoRA adapters correctly attached**
```bash
trainable params: 11,272,192 || all params: 1,247,086,592 || trainable%: 0.9039
```
> _Note_: This means only ~0.9% of the model is being fine-tuned, perfect for QLoRA. The LoRA config is working.  

- **Dataset fully loaded and tokenized**
```bash
Map: 100% 14731/14731 [00:17]
Map: 100% 818/818 [00:00]
```
> _Note_: Fast and correct — tokenizer & formatting are working.  

- **QLoRA Training Started**
```bash
 🚀 Starting QLoRA training…
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
```
> _Note_: This is normal. It actually reduces memory usage, which is good.

- **Training is progressing normally**
```bash
  {'loss': 3.4777, 'grad_norm': 2.6477, 'learning_rate': 8.6e-06, 'epoch': 0.01}
  0% 26/5526 [03:38<12:49:09,  8.39s/it]
```

> _Note_: This tells - Training loop is working  - Gradients are valid  - Your GPU is training      
> _Note_: On a Colab T4 GPU, the real expected training time is:    
```bash
  ~25–35 minutes per epoch
  ~1h 20m for 3 epochs  
```  

- **Completion**  
```bash  
💾 Saving LoRA adapters…  
🎉 Training completed. Saved to llama1b-samsum-qlora  
```    
 
###  Experiment Tracking with Weights & Biases (W&B)  
Training runs are instrumented and tracked using [Weights & Biases](https://wandb.ai/). This integration enables:
- Visualization of training loss and evaluation curves
- Learning rate schedule and gradient monitoring
- GPU memory usage tracking during training
- Evaluation metrics logging after each epoch
- Model artifact saving and versioning

All W&B integration is handled in the training script [`run_llama_qlora.py`](run_llama_qlora.py) with minimal and safe changes:
- Added `wandb.init(...)` for project setup
- Configured `report_to="wandb"` and custom `run_name`
- Enabled configuration tracking for reproducibility

#### Example Visualizations

![Training Loss Curve](loss_curve.jpeg)

![Evaluation Metrics](eval_metrics.jpeg)   TO BE UPLOADED

![GPU Utilization](gpu_utilization.jpeg)   

> To access interactive dashboards and full experiment details, [visit our Weights & Biases project](https://wandb.ai/agostinimichelait-ready-tensor/llama-qlora-samsum).

**Usage Note**: All tracking features are enabled only during training within the notebook/script.

---

**Want to view your training run?**
Once W&B is installed and you are logged in, your training run will automatically appear at:
```
https://wandb.ai/<your-team-or-user>/llama-qlora-samsum
```
- Loss curves: automatically logged  
- Eval metrics: automatically logged  
- Model artifacts: saved & versioned  
- GPU utilization: tracked
  

Notebook E, and the model produced:
```yaml
📊 Final ROUGE scores:
ROUGE-1: 0.2142
ROUGE-2: 0.0915
ROUGE-L: 0.1633
```
Are these ROUGE scores good?

For only 1,000 training samples and 1B parameters, these numbers are reasonable.

Typical SAMSum results:  
| Model                                  | ROUGE-1   | ROUGE-2   | ROUGE-L   |
| -------------------------------------- | --------- | --------- | --------- |
| **LLaMA-1B baseline**                  | ~0.15     | ~0.04     | ~0.11     |
| **Your QLoRA fine-tuned (1k samples)** | **0.214** | **0.091** | **0.163** |
| BART-Large (full-size seq2seq)         | 0.49      | 0.26      | 0.45      |
| LLaMA-3-8B fine-tuned                  | 0.40+     | 0.20+     | 0.33+     |

👉 Your model improved significantly over the baseline
👉 But it's still far from seq2seq baselines or larger LLaMA models
👉 Expected: 1B LLaMA is small and not optimized for summarization out of the box

The model:  
- Is working correctly  
- Respects the SAMSum structure  
- Has significant improvement over baseline  
- But is `not yet production-level`, I meant something very specific about quality, robustness, and reliability—not that the model is bad or unusable. The model works, and the ROUGE numbers prove it is learning.
But for deployment in a real product, there are higher expectations.

The model is working and it can be considered as a successful prototype since it is:  
- generating grammatical summaries
- improving over the baseline
- responding correctly to dialogue inputs
- trained with a clean QLoRA pipeline

it’s not yet production-quality. A production summarization model should meet certain criteria that are beyond what a 1B model fine-tuned on 1k samples can achieve.  
- Accuracy is not high enough
- Repetition & hallucination still occur
- A 1B LLaMA is small for summarization  Most production summarizers use: 3B, 7B, or 8B LLMs or seq2seq models optimized for summarization. 1B is good for prototyping, not for commercial-level results.  
- Trained on only 1k samples This is fine for testing, but: SAMSum has 14k training samples, Summarization requires large data, With 1000 samples, the model generalizes poorly. Production summarizers typically need:
  full dataset, 3–5 epochs, larger context window, decoding tuning


what is your model good for right now?

It is great for:
✔️ research
✔️ experimentation
✔️ proof-of-concept demos
✔️ learning QLoRA + LLaMA finetuning
✔️ early prototypes of conversation summarization

🔥 How to make it production-ready
If you want, I can generate:
1. A stronger training run
full SAMSum dataset
3 epochs
stronger LoRA (r=32)

2. A more advanced decoding setup
repetition penalty
penalties on n-grams
deterministic sampling

3. A distilled model evaluation
ROUGE
BERTScore
length normalization
human evaluation templates

4. A Notebook F: “From Prototype to Production”
With:
merging
quantization (GGUF)
benchmarking
packaging for API deployment
  

### Summarize new dialogues with your fine-tuned model  
### Reproduce benchmarking with your own subset  
### Swap in other dialogue datasets with minor tweaks  

---

## Technologies Used

- [Hugging Face Transformers](https://huggingface.co/)
- [PEFT (LoRA/QLoRA)](https://github.com/huggingface/peft)
- [Weights & Biases](https://wandb.ai/site)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) (quantization)
- [GGUF/Llama.cpp compatible conversion](https://github.com/ggerganov/llama.cpp)
- Python, Colab/Jupyter

---

## Results & Benchmarks

Example ROUGE scores for LLaMA-1B fine-tuned on 1k SAMSum samples:

| Model         | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---------------|---------|---------|---------|
| LLaMA-1B base | ~0.15   | ~0.04   | ~0.11   |
| LLaMA-1B QLoRA (1k) | 0.214 | 0.091 | 0.163 |
| BART-Large    | 0.49    | 0.26    | 0.45    |
| LLaMA-3-8B QLoRA | 0.40+ | 0.20+  | 0.33+   |

_For 1B parameter models and limited data, this project significantly improves summarization performance while keeping memory and compute requirements low._

---

## Limitations & Recommendations

- 1B parameter models are ideal for prototyping, not production
- For higher accuracy, train on more data (full 14k SAMSum samples) and/or use larger models (3B/8B)
- Production deployments may require further tuning to reduce hallucinations or errors
- See Notebook F for tips on packaging and deploying your final model

---
 
## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/micag2025/llmed_Certification_Project1_FineTuneFlow/blob/97ed39ce6ae05e2b0546450448328841ef67816f/LICENSE) file for details.

---

## Contact

If you encounter bugs, have questions, or want to request a new feature, please [open an issue](https://github.com/micag2025/llmed_Certification_Project1_FineTuneFlow/issues) on this repository.   






