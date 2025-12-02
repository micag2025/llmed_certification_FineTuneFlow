# =====================================================
# Evaluation Metrics: ROUGE, BERTScore, BLEU
# For BART-LoRA Highlight Summarization
# =====================================================


import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import corpus_bleu

# -------------------------
# Load predictions
# -------------------------
CSV_PATH = "/content/llmed_certification_FineTuneFlow/metrics/validation_predictions.csv"
OUTPUT_PATH = "/content/llmed_certification_FineTuneFlow/metrics/validation_predictions_metrics.csv"

df = pd.read_csv(CSV_PATH)

print(f"📄 Loaded {len(df)} predictions from:")
print(f"   {CSV_PATH}")

# -------------------------
# ROUGE (1, 2, L)
# -------------------------
print("\n🔍 Computing ROUGE scores...")

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)

rouge1_list, rouge2_list, rougeL_list = [], [], []

for pred, ref in zip(df['model_summary'], df['human_summary']):
    scores = scorer.score(ref, pred)
    rouge1_list.append(scores['rouge1'].fmeasure)
    rouge2_list.append(scores['rouge2'].fmeasure)
    rougeL_list.append(scores['rougeL'].fmeasure)

df['rouge1'] = rouge1_list
df['rouge2'] = rouge2_list
df['rougeL'] = rougeL_list

print("✅ ROUGE completed.")


# -------------------------
# BERTScore (F1)
# -------------------------
print("\n🔍 Computing BERTScore (F1)...")

P, R, F1 = bert_score(
    cands=df['model_summary'].tolist(),
    refs=df['human_summary'].tolist(),
    lang='en',
    rescale_with_baseline=True
)

df['bert_f1'] = F1.numpy()

print("✅ BERTScore completed.")


# -------------------------
# BLEU (corpus-level)
# -------------------------
print("\n🔍 Computing BLEU (corpus-level)...")

references = [[ref.split()] for ref in df['human_summary']]
candidates = [pred.split() for pred in df['model_summary']]

bleu = corpus_bleu(references, candidates)

print(f"✅ BLEU Score: {bleu:.4f}")


# -------------------------
# Save CSV with Metrics
# -------------------------
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n💾 Metrics saved to:\n   {OUTPUT_PATH}")

print("\n📊 Average scores:")
print(f"ROUGE-1: {df['rouge1'].mean():.3f}")
print(f"ROUGE-2: {df['rouge2'].mean():.3f}")
print(f"ROUGE-L: {df['rougeL'].mean():.3f}")
print(f"BERTScore-F1: {df['bert_f1'].mean():.3f}")
print(f"BLEU: {bleu:.4f}")

print("\n✔ Evaluation metrics complete.")
