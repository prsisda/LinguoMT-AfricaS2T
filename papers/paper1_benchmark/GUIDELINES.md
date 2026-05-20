# Paper 1 — Experiment Guidelines: LinguoMT Benchmark

Step-by-step workflow for running and reporting all experiments for this paper. Follow the steps in order — later steps depend on earlier outputs.

<!-- TOC -->
- [Paper 1 — Experiment Guidelines: LinguoMT Benchmark](#paper-1-experiment-guidelines-linguomt-benchmark)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Step 1 — Fill in published SOTA baselines](#step-1-fill-in-published-sota-baselines)
  - [Step 2 — Run ASR evaluation (Table 1)](#step-2-run-asr-evaluation-table-1)
  - [Step 3 — Run S2TT evaluation (Table 2)](#step-3-run-s2tt-evaluation-table-2)
  - [Step 4 — Run the cascade baseline (Table 3)](#step-4-run-the-cascade-baseline-table-3)
  - [Step 5 — Run high-resource gap analysis (Table 4, Figure 1)](#step-5-run-high-resource-gap-analysis-table-4-figure-1)
  - [Step 6 — Metric sensitivity check (Table 5)](#step-6-metric-sensitivity-check-table-5)
  - [Step 7 — Generate tables and figures](#step-7-generate-tables-and-figures)
  - [Step 8 — Validate schema and results](#step-8-validate-schema-and-results)
  - [Reporting checklist](#reporting-checklist)
  - [What to hand off to other papers](#what-to-hand-off-to-other-papers)
<!-- /TOC -->

---

## Overview

This paper is purely zero-shot evaluation. You run pre-trained models as-is, record scores, compare them against each other and against published baselines, and quantify the performance gap between African and high-resource languages. No fine-tuning. No audio augmentation.

**Deliverables:** Table 1 (ASR), Table 2 (S2TT), Table 3 (cascade baseline), Table 4 (gap analysis), Table 5 (metric sensitivity), Figure 1 (gap bar chart).

---

## Prerequisites

- [ ] GPU with at least 16 GB VRAM (SeamlessM4T-v2-large requires it)
- [ ] Python environment with `transformers`, `datasets`, `sacrebleu`, `jiwer` installed
- [ ] `PAPER_MODE = "benchmark"` set in your run script
- [ ] `SOTA_FILE` pointing to `papers/paper1_benchmark/references.yaml`

---

## Step 1 — Fill in published SOTA baselines

Before running anything, collect the published numbers that will anchor your comparison tables.

Open `references.yaml` and fill in the `score` field for each entry using the sources listed in the table below. Leave `score: null` only for entries you cannot locate.

| Entry | Where to find the score |
|-------|------------------------|
| SeamlessM4T-v2 BLEU (Yoruba, Hausa, Igbo) | Table B.1 of `arXiv:2312.05187` |
| mSLAM BLEU (Yoruba, Hausa) | Supplementary S2TT table of `arXiv:2202.01374` |
| mSLAM-CTC BLEU (FLEURS baseline) | Table 3 of `arXiv:2205.12446` |
| Whisper-large-v3 WER (Yoruba, Hausa) | HuggingFace model card — FLEURS ASR eval |
| MMS-300M WER (Yoruba, Hausa, Igbo) | Per-language appendix of `arXiv:2305.13516` |
| XLS-R-1B WER (Yoruba, Hausa) | Table 4 of `arXiv:2111.09296` |
| NLLB-200-600M spBLEU (Flores-200) | Table 2 of `arXiv:2207.04672` |

Validate the file:
```bash
python -c "
from framework.sota import load_and_validate_sota
entries = load_and_validate_sota('papers/paper1_benchmark/references.yaml', 'papers/paper1_benchmark/schema.json')
print(f'{len(entries)} valid entries')
"
```

Expected: no schema errors, at least one entry with a non-null score per language.

---

## Step 2 — Run ASR evaluation (Table 1)

Evaluate all ASR-capable models on FLEURS test split. Run one model at a time to avoid VRAM contention.

**Models to run:** Whisper-large-v3, MMS-300M, XLS-R-1B, SeamlessM4T-v2-large (ASR head)  
**Languages:** Yoruba, Hausa, Igbo  
**Metrics to record:** WER, CER

```python
PAPER_MODE = "benchmark"
MODEL_ID   = "openai/whisper-large-v3"   # repeat for each model
DATASET_ID = "google/fleurs"
ENABLE_FINETUNING = False
```

Record for each model × language:
- WER (word error rate, %)
- CER (character error rate, %)
- Number of test samples used
- Any languages the model does not support (e.g., Whisper does not support Igbo)

**Expected output:** `results/benchmark/asr_<model>_<language>.json`

---

## Step 3 — Run S2TT evaluation (Table 2)

Evaluate end-to-end speech-to-text translation models.

**Models to run:** SeamlessM4T-v2-large, mSLAM (if accessible via HuggingFace)  
**Direction:** Source language → English  
**Languages:** Yoruba, Hausa, Igbo  
**Metrics to record:** BLEU (sacrebleu, tokenize=13a), spBLEU (tokenize=flores101), ChrF++

```python
PAPER_MODE = "benchmark"
MODEL_ID   = "facebook/seamless-m4t-v2-large"
DATASET_ID = "google/fleurs"
ENABLE_FINETUNING = False
```

**Expected output:** `results/benchmark/s2tt_<model>_<language>.json`

---

## Step 4 — Run the cascade baseline (Table 3)

Build and evaluate the Whisper + NLLB-200 cascade as a zero-shot S2TT baseline. This also provides the starting point for Paper 4's architecture comparison — record enough detail here so Paper 4 can cite these numbers directly.

```python
PAPER_MODE = "benchmark"
MODEL_ID   = "whisper_nllb"   # virtual cascade model id
DATASET_ID = "google/fleurs"
ENABLE_FINETUNING = False
```

Record for each language:
- Intermediate ASR WER (Whisper transcripts before translation)
- Cascade BLEU / spBLEU (Whisper → NLLB-200 output)
- Text-MT ceiling BLEU: run NLLB-200-600M on the **gold reference transcripts** from FLEURS (not audio) to establish the upper bound for any cascade

NLLB-200 language codes: `yor_Latn` (Yoruba), `hau_Latn` (Hausa), `ibo_Latn` (Igbo)

---

## Step 5 — Run high-resource gap analysis (Table 4, Figure 1)

Repeat Step 2 and Step 3 for three high-resource reference languages to quantify how far African languages lag.

**Reference languages:** French (`fr_fr`), German (`de_de`), Spanish (`es_419`) on FLEURS  
**Models:** Whisper-large-v3 (ASR), SeamlessM4T-v2-large (S2TT)

Compute for each African language:
- Relative WER gap = (WER_african − WER_reference) / WER_reference × 100
- Absolute BLEU gap = BLEU_reference − BLEU_african

**Expected output:** gap table + bar chart comparing all languages side by side.

---

## Step 6 — Metric sensitivity check (Table 5)

Confirm that BLEU, spBLEU, and ChrF rank systems consistently. Reuse the model outputs already generated in Steps 3 and 4 — no new inference needed.

For SeamlessM4T-v2-large and the Whisper+NLLB cascade:
1. Score each system's output with all three metrics using the same references
2. Compute Kendall's τ between BLEU and spBLEU rankings and between BLEU and ChrF rankings
3. If τ < 0.8 for any pair, investigate and note it in the paper

```python
import sacrebleu, scipy.stats
# bleu_scores, spbleu_scores, chrf_scores are lists of per-system scores
tau_bleu_spbleu, _ = scipy.stats.kendalltau(bleu_scores, spbleu_scores)
```

---

## Step 7 — Generate tables and figures

Run the output generation step to produce camera-ready tables:

```bash
python run_benchmark.py --output-only   # regenerate tables from cached results
```

Check that every table has been populated and contains no `null` cells. Cross-check any score you reported in Step 1 against your own run — flag any discrepancy > 1 BLEU point.

---

## Step 8 — Validate schema and results

```bash
python -c "
from framework.sota import load_and_validate_sota
load_and_validate_sota('papers/paper1_benchmark/references.yaml', 'papers/paper1_benchmark/schema.json')
print('Schema OK')
"
```

Then run the full test suite:
```bash
python -m pytest tests/ -k benchmark -v
```

---

## Reporting checklist

- [ ] Table 1: WER/CER for all ASR models × 3 languages (note unsupported language pairs)
- [ ] Table 2: BLEU/spBLEU/ChrF for all S2TT models × 3 languages
- [ ] Table 3: Cascade vs E2E vs text-MT ceiling × 3 languages
- [ ] Table 4: High-resource vs African-language gap for WER and BLEU
- [ ] Table 5: Metric rank correlation (Kendall's τ)
- [ ] Figure 1: Bar chart — WER and BLEU by language (African + reference)
- [ ] All `score` fields in `references.yaml` filled in or explicitly marked `null` with a reason in `notes`
- [ ] `paper_references.csv` updated with all newly cited papers

---

## What to hand off to other papers

| Output | Used by |
|--------|---------|
| SeamlessM4T-v2-large zero-shot BLEU (all 3 languages) | Paper 2 (pre-adaptation baseline), Paper 4 (E2E reference), Paper 5 (zero-shot reference) |
| Whisper-large-v3 zero-shot WER (Yoruba, Hausa) | Paper 4 (cascade starting WER) |
| Cascade BLEU + intermediate WER (Step 4) | Paper 4 |
| High-resource BLEU/WER (Step 5) | All papers for framing |
