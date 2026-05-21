# Paper 5 — LinguoMT-Transfer: Experiment Guidelines

Cross-lingual transfer: Niger-Congo (Yoruba, Igbo, Swahili) vs Afro-Asiatic (Hausa).
Zero-shot baselines come from Paper 1. Fine-tuning uses full fine-tune on small data budgets.

**Deliverables:** Table 1 (typological similarity), Table 2 (transfer strategies), Table 3 (data scaling), Figures 1–2, paper_outline.filled.md.

---

## Key files

| File | Purpose |
|------|---------|
| `papers/experiment_setup.yaml` | Language families, codes — shared reference |
| `papers/paper5_transfer/config.yaml` | Transfer pairs, few-shot budgets, URIEL setup |
| `papers/paper5_transfer/baselines.csv` | Zero-shot baselines (from Paper 1) |
| `papers/paper5_transfer/paper_outline.md` | Paper skeleton with `[RESULT:key]` placeholders |
| `papers/fill_results.py` | Fills placeholders from experiment output CSVs |

---

## Prerequisites

- [ ] Paper 1 zero-shot results (WER and BLEU for all three languages)
- [ ] `lang2vec` installed: `pip install lang2vec`
- [ ] FLEURS train split accessible
- [ ] `PAPER_MODE = "transfer"` in run scripts

---

## Step 1 — Compute typological similarity (Table 1)

```python
import lang2vec.lang2vec as l2v
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

langs    = ["yor", "ibo", "hau", "eng", "fra"]
features = l2v.get_features(langs, "syntax_knn")
vecs     = np.array([features[l] for l in langs])
sim      = cosine_similarity(vecs)

# Report pairwise similarity for the paper:
# Yoruba ↔ Igbo:    sim[0, 1]   (expected: high — both Volta-Niger)
# Yoruba ↔ Hausa:   sim[0, 2]   (expected: low — cross-family)
# Igbo ↔ Hausa:     sim[1, 2]   (expected: low)
# Yoruba ↔ English: sim[0, 3]   (reference)
```

---

## Step 2 — Import zero-shot baselines from Paper 1

Copy SeamlessM4T-v2 BLEU and Whisper WER (zero-shot) from Paper 1 into `baselines.csv`.
These are your `transfer_type: zero_shot, num_ft_samples: 0` entries.

---

## Step 3 — Configure run scripts

```python
DEBUG_MODE        = False
PAPER_MODE        = "transfer"
ENABLE_FINETUNING = True
FINETUNING_METHOD = "full"      # NOT lora — use full FT to isolate typological effect
SOTA_FILE         = "papers/paper5_transfer/baselines.csv"
```

Language settings:
- `FLEURS__SeamlessM4Tv2`: `MANUAL_LANGUAGES = ["igbo", "yoruba", "swahili"]`
- `FLEURS__WhisperNLLB`:   `MANUAL_LANGUAGES = ["yoruba", "hausa", "swahili"]`

---

## Step 4 — Run few-shot fine-tuning

Run at data budgets: 25, 50, 100, 200 samples per language.

```python
# For each budget, set:
TEXT_FINETUNE_SAMPLES = 100    # then 25, 50, 200
ASR_FINETUNE_SAMPLES  = 50     # proportionally
```

Record post-adaptation WER (ASR) and BLEU (S2TT) per language per budget.

---

## Step 5 — Run cross-lingual fine-tuning

For each transfer pair in `config.yaml`:

```python
FT_LANGUAGE   = "yoruba"   # language to train on
EVAL_LANGUAGE = "igbo"     # language to evaluate on (different)
FT_SAMPLES    = 1000
```

This requires modifying the data cache or running a standalone fine-tuning script.
Record `source_lang`, `target_lang`, WER/BLEU for each pair.

---

## Step 6 — Data scaling and interaction analysis

Using few-shot results from Step 4:

```python
# Transfer efficiency per language
efficiency = (wer_zero_shot - wer_after_N) / N_samples

# Regression to test family × data interaction
import statsmodels.formula.api as smf
df["lang_family"] = df["language"].map({"yoruba": 0, "igbo": 0, "hausa": 1})
model = smf.ols("wer ~ np.log(num_ft_samples) * lang_family", data=df).fit()
print(model.summary())
# Report interaction term coefficient and p-value
```

---

## Step 7 — Fill in paper_outline.md

```bash
python papers/fill_results.py paper5_transfer
```

Output: `papers/paper5_transfer/paper_outline.filled.md`

---

## Step 8 — Validate

```bash
python -c "
from framework.sota import load_and_validate_sota
load_and_validate_sota('papers/paper5_transfer/baselines.csv', 'papers/paper5_transfer/schema.json')
print('Schema OK')
"
```

---

## Reporting checklist

- [ ] Table 1: Pairwise URIEL cosine similarity for all language pairs
- [ ] Table 2: Zero-shot vs few-shot vs cross-lingual FT × 3 languages
- [ ] Table 3: Data scaling — WER/BLEU at 25/50/100/200 samples, by family
- [ ] Figure 1: Learning curves per language
- [ ] Figure 2: Transfer efficiency frontier — WER gain/sample, coloured by family
- [ ] Interaction term coefficient and p-value reported
- [ ] Convergence sample count identified (cross-lingual FT reaches within X% of monolingual)
- [ ] `paper_outline.filled.md` generated

---

## Scope reminder

- Do NOT run LoRA/adapter here → Paper 2 (this paper uses full FT on small data)
- Do NOT vary audio conditions → Paper 3
- Do NOT compare cascade vs E2E systematically → Paper 4
