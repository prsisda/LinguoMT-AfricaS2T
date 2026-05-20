# Paper 2 — Experiment Guidelines: LinguoMT-Adapt

Step-by-step workflow for running and reporting all PEFT experiments. Follow in order — data scaling (Step 4) depends on the single fine-tuning run (Step 3).

<!-- TOC -->
- [Paper 2 — Experiment Guidelines: LinguoMT-Adapt](#paper-2-experiment-guidelines-linguomt-adapt)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Step 1 — Record pre-adaptation baselines](#step-1-record-pre-adaptation-baselines)
  - [Step 2 — Prepare fine-tuning data](#step-2-prepare-fine-tuning-data)
  - [Step 3 — Run LoRA fine-tuning (main experiment)](#step-3-run-lora-fine-tuning-main-experiment)
  - [Step 4 — Run adapter fine-tuning (comparison)](#step-4-run-adapter-fine-tuning-comparison)
  - [Step 5 — Data scaling experiment (Table 3, Figure 1)](#step-5-data-scaling-experiment-table-3-figure-1)
  - [Step 6 — Parameter efficiency analysis (Table 2)](#step-6-parameter-efficiency-analysis-table-2)
  - [Step 7 — Generate tables and figures](#step-7-generate-tables-and-figures)
  - [Step 8 — Validate](#step-8-validate)
  - [Reporting checklist](#reporting-checklist)
  - [Scope reminder](#scope-reminder)
<!-- /TOC -->

---

## Overview

This paper asks: how much of the full-fine-tuning gain can LoRA recover, with how few trainable parameters, and with how little data? The zero-shot baselines come from Paper 1 — do not re-run them here, cite them. Your job is to run and compare three adaptation conditions: LoRA, adapter modules, and full fine-tuning.

**Deliverables:** Table 1 (before/after per method), Table 2 (parameter budget comparison), Table 3 (data scaling), Figure 1 (learning curves), Figure 2 (PEFT efficiency frontier).

---

## Prerequisites

- [ ] Paper 1 zero-shot results available (BLEU for SeamlessM4T-v2, WER for Whisper)
- [ ] FLEURS **train** split accessible (used for fine-tuning data)
- [ ] GPU with at least 24 GB VRAM for LoRA on SeamlessM4T-v2-large; 16 GB sufficient for smaller variants
- [ ] `peft` library installed (`pip install peft`)
- [ ] `PAPER_MODE = "adaptation"` and `ENABLE_FINETUNING = True` in your run script

---

## Step 1 — Record pre-adaptation baselines

Copy the zero-shot scores from Paper 1 into `references.yaml` as entries with `ft_method: none`. These are your "before" column.

| Entry | Score source |
|-------|-------------|
| SeamlessM4T-v2-large zero-shot BLEU | Paper 1, Table 2 |
| SeamlessM4T-v2-large zero-shot WER | Paper 1, Table 1 |

Also fill in the full fine-tuning upper bounds from published work:

| Entry | Where to find the score |
|-------|------------------------|
| Wav2Vec2-XLSR full FT WER (Hausa, Yoruba) | Table 2 of `arXiv:2206.00253` (MasakhaSpeech) |
| Whisper-large-v2 full FT WER (AfriSpeech) | Table 4 of `arXiv:2104.02010` |

These upper bounds define the ceiling your PEFT methods are measured against.

---

## Step 2 — Prepare fine-tuning data

Use the FLEURS **train** split only. Keep the test split clean — never touch it during training.

Prepare three data budgets for the data scaling experiment in Step 4:

| Budget name | Samples per language | Approximate audio hours |
|-------------|---------------------|------------------------|
| `tiny` | 100 | ~0.5 h |
| `small` | 500 | ~2.5 h |
| `medium` | 1 000 | ~5 h |
| `full` | all train | ~8–10 h |

Save each split as a reproducible subset (fix the random seed to 42):
```python
FT_SEED    = 42
FT_SAMPLES = 500   # vary per budget
```

---

## Step 3 — Run LoRA fine-tuning (main experiment)

Use the `medium` budget (1 000 samples) as the main result. Run for all three languages independently.

```python
PAPER_MODE        = "adaptation"
ENABLE_FINETUNING = True
FT_METHOD         = "lora"    # framework.finetuning FinetuneConfig.finetuning_method
LORA_R            = 8
LORA_ALPHA        = 16
LORA_DROPOUT      = 0.05
FT_SAMPLES        = 1000
SOTA_FILE         = "papers/paper2_adaptation/references.yaml"
```

Record for each language after fine-tuning:
- Post-adaptation BLEU (S2TT, Source → English)
- Post-adaptation WER (ASR)
- Number of trainable parameters (absolute and as % of total)
- Training wall-clock time (GPU hours)
- Fill `pretrained_score` in `references.yaml` with the Paper 1 zero-shot score

---

## Step 4 — Run adapter fine-tuning (comparison)

Repeat Step 3 with `FT_METHOD = "adapter"` (Bapna & Firat 2019 style). Use the same `medium` budget and seed.

Record the same metrics as Step 3. This is the direct comparison point against LoRA in Table 2.

---

## Step 5 — Data scaling experiment (Table 3, Figure 1)

Run LoRA fine-tuning (best configuration from Step 3) at all four data budgets defined in Step 2. One run per language × budget = 12 runs total.

```python
for samples in [100, 500, 1000, "all"]:
    FT_SAMPLES = samples
    # run LoRA fine-tuning
```

Record post-adaptation WER and BLEU for each budget. Plot learning curves: x-axis = number of training samples, y-axis = WER (ASR) and BLEU (S2TT).

Identify the **minimum data threshold**: the smallest budget at which PEFT shows a statistically significant improvement over the zero-shot baseline (paired t-test, p < 0.05 on FLEURS test).

---

## Step 6 — Parameter efficiency analysis (Table 2)

For each method (LoRA, adapter, full fine-tune), report:

| Column | How to compute |
|--------|---------------|
| Trainable params (M) | count parameters where `requires_grad=True` |
| % of total params | trainable / total × 100 |
| BLEU gain over zero-shot | post_score − pretrained_score |
| BLEU gain per 1M trainable params | BLEU gain / trainable_M |
| GPU hours | wall-clock training time |

Full fine-tuning numbers come from published baselines (Step 1) — you do not need to run full fine-tuning yourself unless compute allows.

---

## Step 7 — Generate tables and figures

```bash
python run_adaptation.py --output-only
```

Check:
- Table 1 has non-null before/after scores for all three languages
- Table 2 shows all three methods with parameter counts
- Figure 1 shows learning curves with error bars (if multiple seeds were run)

---

## Step 8 — Validate

```bash
python -c "
from framework.sota import load_and_validate_sota
load_and_validate_sota('papers/paper2_adaptation/references.yaml', 'papers/paper2_adaptation/schema.json')
print('Schema OK')
"
```

Confirm that every entry has `ft_method` set and `pretrained_score` filled where available.

---

## Reporting checklist

- [ ] Table 1: Before/after BLEU and WER for LoRA and adapter, per language
- [ ] Table 2: Parameter budget comparison (LoRA vs adapter vs full FT upper bound)
- [ ] Table 3: Data scaling — BLEU/WER at each budget, per language
- [ ] Figure 1: Learning curves (samples vs BLEU/WER) for all three languages
- [ ] Figure 2: Efficiency frontier — BLEU gain vs trainable parameters (scatter plot)
- [ ] Minimum data threshold identified and stated in the text
- [ ] `references.yaml` has `pretrained_score` filled for all fine-tuned entries
- [ ] `paper_references.csv` updated

---

## Scope reminder

Do **not** run audio preprocessing experiments here — that is Paper 3.  
Do **not** compare cascade vs E2E architectures here — that is Paper 4.  
Do **not** vary source language for cross-lingual transfer here — that is Paper 5. Adaptation in this paper always uses target-language data only.
