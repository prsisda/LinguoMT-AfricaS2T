# Paper 5 — Experiment Guidelines: LinguoMT-Transfer

Step-by-step workflow for running and reporting cross-lingual transfer experiments.

<!-- TOC -->
- [Paper 5 — Experiment Guidelines: LinguoMT-Transfer](#paper-5-experiment-guidelines-linguomt-transfer)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Step 1 — Compute typological similarity scores (Table 1)](#step-1-compute-typological-similarity-scores-table-1)
  - [Step 2 — Record zero-shot baselines (do not re-run)](#step-2-record-zero-shot-baselines-do-not-re-run)
  - [Step 3 — Few-shot fine-tuning (< 1 hour of data)](#step-3-few-shot-fine-tuning-1-hour-of-data)
  - [Step 4 — Cross-lingual fine-tuning](#step-4-cross-lingual-fine-tuning)
  - [Step 5 — Data scaling analysis per language family (Table 3, Figure 1)](#step-5-data-scaling-analysis-per-language-family-table-3-figure-1)
  - [Step 6 — Convergence analysis](#step-6-convergence-analysis)
  - [Step 7 — Interaction analysis (Table 3)](#step-7-interaction-analysis-table-3)
  - [Step 8 — Generate tables and figures](#step-8-generate-tables-and-figures)
  - [Reporting checklist](#reporting-checklist)
  - [Scope reminder](#scope-reminder)
<!-- /TOC -->

---

## Overview

This paper asks: does linguistic family membership predict cross-lingual transfer efficiency, and which transfer strategy (zero-shot, few-shot, cross-lingual fine-tuning) is best for each language? The zero-shot baselines come from Paper 1. Paper 2 owns PEFT efficiency — this paper focuses on the typological dimension: Yoruba and Igbo (Niger-Congo, Volta-Niger) vs Hausa (Afro-Asiatic, Chadic).

**Deliverables:** Table 1 (typological similarity scores), Table 2 (transfer strategies comparison), Table 3 (data scaling per language family), Figure 1 (learning curves per language), Figure 2 (transfer efficiency frontier by language family).

---

## Prerequisites

- [ ] Paper 1 zero-shot baselines available (WER and BLEU for all three languages)
- [ ] `lang2vec` installed (`pip install lang2vec`) for URIEL typological features
- [ ] FLEURS train split accessible for few-shot and fine-tuning experiments
- [ ] `PAPER_MODE = "transfer"` in your run script

---

## Step 1 — Compute typological similarity scores (Table 1)

Use URIEL via `lang2vec` to compute cosine similarity between language feature vectors. This gives an objective typological distance measure to interpret transfer results.

```python
import lang2vec.lang2vec as l2v

langs = ["yor", "ibo", "hau", "eng", "fra"]
features = l2v.get_features(langs, "syntax_knn")

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vecs = np.array([features[l] for l in langs])
sim_matrix = cosine_similarity(vecs)
```

Report pairwise cosine similarity for:
- Yoruba ↔ Igbo (expected: high — both Niger-Congo, Volta-Niger)
- Yoruba ↔ Hausa (expected: low — cross-family)
- Igbo ↔ Hausa (expected: low — cross-family)
- Yoruba ↔ English (reference point)

Save `lang_similarity_score` for each language pair in `references.yaml`.

---

## Step 2 — Record zero-shot baselines (do not re-run)

Import zero-shot WER and BLEU from Paper 1 into `references.yaml` as entries with `transfer_type: zero_shot` and `num_ft_samples: 0`.

| Entry | Source |
|-------|--------|
| SeamlessM4T-v2 zero-shot BLEU × 3 languages | Paper 1, Table 2 |
| mSLAM zero-shot BLEU (Yoruba, Hausa) | Paper 1, Table 2 |
| MMS-300M WER × 3 languages | Paper 1, Table 1 |
| XLS-R-1B WER (Yoruba, Hausa) | Paper 1, Table 1 |

---

## Step 3 — Few-shot fine-tuning (< 1 hour of data)

Run fine-tuning on XLS-R-1B and MMS-300M with very small data budgets. The goal is to find the minimum amount of target-language data that gives meaningful improvement.

**Data budgets:** 25, 50, 100, 200 samples per language  
**Languages:** Yoruba, Hausa, Igbo  
**Models:** XLS-R-1B, MMS-300M (both support all three languages)

```python
PAPER_MODE        = "transfer"
ENABLE_FINETUNING = True
FT_METHOD         = "full"          # full fine-tune on small data
FT_SAMPLES        = 50              # repeat for each budget
TRANSFER_TYPE     = "few_shot"
```

Record for each model × language × budget:
- Post-adaptation WER
- `transfer_type: few_shot`
- `num_ft_samples` (actual count used)

---

## Step 4 — Cross-lingual fine-tuning

Fine-tune on a **source language** then evaluate on a **target language** without any target-language data. This tests whether linguistic relatedness enables free transfer.

**Transfer pairs to run:**

| Source (train on) | Target (eval on) | Expected |
|-------------------|-----------------|---------|
| Yoruba | Igbo | Good transfer — related family |
| Igbo | Yoruba | Good transfer — related family |
| Yoruba | Hausa | Poor transfer — cross-family |
| Hausa | Yoruba | Poor transfer — cross-family |
| English | Yoruba | Reference baseline |
| English | Hausa | Reference baseline |

```python
FT_LANGUAGE    = "yoruba"    # language to train on
EVAL_LANGUAGE  = "igbo"      # language to evaluate on
TRANSFER_TYPE  = "cross_lingual_finetune"
FT_SAMPLES     = 1000        # use full available training data for source language
```

Record: `source_lang_family`, `target_lang_family`, `transfer_type: cross_lingual_finetune`.

---

## Step 5 — Data scaling analysis per language family (Table 3, Figure 1)

Using the few-shot results from Step 3, build learning curves for each language. Plot WER vs number of training samples for Yoruba, Igbo, and Hausa on the same axes.

Compute **transfer efficiency** for each language:
```
efficiency = (WER_zero_shot − WER_after_N_samples) / N_samples
```

Higher efficiency = more WER reduction per training sample.

Compare efficiency between the Niger-Congo group (Yoruba, Igbo) and Hausa:
- If Niger-Congo efficiency > Hausa efficiency → linguistic relatedness explains the gap
- If they are similar → model pretraining data imbalance is the more likely explanation

---

## Step 6 — Convergence analysis

Find the data size at which cross-lingual fine-tuning converges to full monolingual fine-tuning performance (using MasakhaSpeech full fine-tuning WER as the target).

Extend the data scaling from Step 5 to larger budgets: 500, 1 000, 2 000, all-train.

Report: "cross-lingual fine-tuning reaches within X% of full fine-tuning performance with N samples for Niger-Congo languages, vs M samples for Hausa."

---

## Step 7 — Interaction analysis (Table 3)

Fit a simple regression to test whether source language family and data size interact:

```python
import statsmodels.formula.api as smf

df["lang_family"] = df["language"].map({"yoruba": 0, "igbo": 0, "hausa": 1})
model = smf.ols("wer ~ np.log(num_ft_samples) * lang_family", data=df).fit()
print(model.summary())
```

Report the interaction term coefficient and its significance. This is the statistical evidence for or against the linguistic relatedness hypothesis.

---

## Step 8 — Generate tables and figures

```bash
python run_transfer.py --output-only
```

Verify:
- Table 2 shows clear ranking: cross-lingual FT > few-shot > zero-shot for all three languages
- Figure 1 learning curves show faster improvement for Niger-Congo languages (if hypothesis holds)
- Figure 2 efficiency scatter shows Niger-Congo languages clustered together, Hausa separated

---

## Reporting checklist

- [ ] Table 1: Pairwise typological similarity scores (URIEL cosine) for all language pairs
- [ ] Table 2: Zero-shot vs few-shot vs cross-lingual FT × 3 languages × 2 models
- [ ] Table 3: Data scaling — WER at each budget per language and family
- [ ] Figure 1: Learning curves (samples vs WER) per language
- [ ] Figure 2: Transfer efficiency frontier — WER gain per sample, coloured by family
- [ ] Transfer efficiency values computed and compared across families
- [ ] Convergence data size identified and stated
- [ ] Interaction term significance reported
- [ ] `transfer_type`, `source_lang_family`, `target_lang_family`, `num_ft_samples`, `lang_similarity_score` filled in `references.yaml` for all entries
- [ ] `paper_references.csv` updated with URIEL/lang2vec citation

---

## Scope reminder

Do **not** run LoRA or adapter experiments here — that is Paper 2. Fine-tuning in this paper uses full fine-tuning on small data to isolate the typological effect, not to demonstrate PEFT efficiency.  
Do **not** vary audio conditions — that is Paper 3.  
Do **not** compare cascade vs E2E architectures — that is Paper 4.
