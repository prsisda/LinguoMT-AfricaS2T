# LinguoMT — African Speech Translation Research

A journal series investigating automatic speech recognition and speech-to-text translation for low-resource African languages. The project spans five papers, each targeting a distinct research question, built on a shared Python evaluation framework.

<!-- TOC -->
- [LinguoMT — African Speech Translation Research](#linguomt-african-speech-translation-research)
  - [Research Goal](#research-goal)
  - [Framework](#framework)
    - [Running an experiment](#running-an-experiment)
  - [Models](#models)
    - [End-to-end speech translation](#end-to-end-speech-translation)
    - [ASR-only models](#asr-only-models)
    - [Text machine translation](#text-machine-translation)
    - [Cascade pipeline](#cascade-pipeline)
    - [NLLB language codes for target languages](#nllb-language-codes-for-target-languages)
  - [Papers](#papers)
    - [Paper 1 — LinguoMT: Benchmark](#paper-1-linguomt-benchmark)
    - [Paper 2 — LinguoMT-Adapt: PEFT Fine-Tuning](#paper-2-linguomt-adapt-peft-fine-tuning)
    - [Paper 3 — LinguoMT-Audio: Preprocessing & Robustness](#paper-3-linguomt-audio-preprocessing-robustness)
    - [Paper 4 — LinguoMT-Cascade: Architecture Comparison](#paper-4-linguomt-cascade-architecture-comparison)
    - [Paper 5 — LinguoMT-Transfer: Cross-Lingual Adaptation](#paper-5-linguomt-transfer-cross-lingual-adaptation)
  - [Scope boundaries across papers](#scope-boundaries-across-papers)
  - [Shared citation index](#shared-citation-index)
  - [TOC maintenance](#toc-maintenance)
<!-- /TOC -->

---

## Research Goal

Spoken language is the dominant communication mode for hundreds of millions of people across sub-Saharan Africa, yet modern speech translation systems are systematically evaluated and optimised for high-resource languages. LinguoMT aims to:

1. Establish a reproducible benchmark for African-language ASR and speech translation.
2. Identify where state-of-the-art multilingual models fail and by how much.
3. Evaluate practical adaptation strategies (PEFT, audio preprocessing, cascade vs end-to-end architectures, cross-lingual transfer) that are feasible in low-compute, low-data settings.

**Core languages:** Yoruba (Niger-Congo · Volta-Niger), Hausa (Afro-Asiatic · Chadic), Igbo (Niger-Congo · Volta-Niger)  
**Primary evaluation dataset:** FLEURS (`google/fleurs`)  
**Secondary dataset:** AfricanCeltic (`McGill-NLP/african_celtic_dataset`)

---

## Framework

The shared codebase lives in `framework/` and is reused across all five papers. Every paper runs the same evaluation pipeline; the `PAPER_MODE` flag controls which tables, plots, and interpretations are emphasised.

| Module | Purpose |
|--------|---------|
| `paper_modes.py` | Defines the five `PAPER_MODE` configs and which outputs each enables |
| `capabilities.py` | Model capability registry — declares what each model can do (ASR, S2TT, T2TT, cascade) |
| `languages.py` | African language registry — ISO codes, dataset configs, model-specific language codes |
| `dataset.py` | Dataset loading and preprocessing for FLEURS and AfricanCeltic |
| `experiments.py` | Core experiment runner — evaluates a model across languages, tasks, and strategies |
| `metrics.py` | BLEU, spBLEU, ChrF, WER, CER scoring via sacrebleu and jiwer |
| `audio.py` | Audio preprocessing strategies — VAD, normalisation, chunking, SpecAugment |
| `finetuning.py` | LoRA and full fine-tuning for all supported model families |
| `sota.py` | Loads and validates SOTA baselines from `references.yaml` / `sota_results.csv` |
| `tables.py` | Generates comparison and result tables |
| `plots.py` | Generates figures (BLEU by language, strategy comparison, learning curves) |
| `output.py` | Writes results to disk in structured format |
| `monitoring.py` | GPU/CPU monitoring during runs |
| `interpretations.py` | Auto-generated text summaries of key findings |
| `environment.py` | Environment detection (Colab, local, GPU availability) |

### Running an experiment

Each paper has its own run script at the project root. The key settings are:

```python
PAPER_MODE = "benchmark"          # selects which outputs to generate
SOTA_FILE  = "papers/paper1_benchmark/references.yaml"   # baseline references
```

---

## Models

### End-to-end speech translation

| Model | HuggingFace ID | ASR | S2TT | T2TT | Notes |
|-------|---------------|-----|------|------|-------|
| SeamlessM4T-v2-large | `facebook/seamless-m4t-v2-large` | ✓ | ✓ | ✓ | Primary E2E baseline; supports all three target languages |
| SeamlessM4T-large | `facebook/seamless-m4t-large` | ✓ | ✓ | ✓ | Older v1 variant |

### ASR-only models

| Model | HuggingFace ID | Languages supported |
|-------|---------------|-------------------|
| Whisper-large-v3 | `openai/whisper-large-v3` | Yoruba (`yo`), Hausa (`ha`) — Igbo not supported |
| Whisper-large-v2 | `openai/whisper-large-v2` | Yoruba, Hausa |

### Text machine translation

| Model | HuggingFace ID | Direction | Notes |
|-------|---------------|-----------|-------|
| NLLB-200-distilled-600M | `facebook/nllb-200-distilled-600M` | Any → Any | Used as MT component in cascade and text-MT ceiling |
| NLLB-200-1.3B | `facebook/nllb-200-1.3B` | Any → Any | Larger variant |

### Cascade pipeline

| Pipeline | Components | Notes |
|----------|-----------|-------|
| Whisper + NLLB-200 | `openai/whisper-large-v3` → `facebook/nllb-200-distilled-600M` | Virtual model ID: `whisper_nllb`; used for cascade vs E2E analysis |

### NLLB language codes for target languages

| Language | NLLB code | Whisper code | Seamless code |
|----------|-----------|--------------|---------------|
| Yoruba | `yor_Latn` | `yo` | `yor` |
| Hausa | `hau_Latn` | `ha` | `hau` |
| Igbo | `ibo_Latn` | — (not supported) | `ibo` |

---

## Papers

Each paper builds on the previous one's results. The scope boundaries are strict — no paper duplicates experiments owned by another.

### Paper 1 — LinguoMT: Benchmark

> Zero-shot evaluation of all model families on FLEURS. Establishes the baseline numbers that all other papers compare against.

**Mode:** `benchmark` | **Folder:** `paper1_benchmark/`

| File | Purpose |
|------|---------|
| [README.md](paper1_benchmark/README.md) | Abstract, RQs, data field reference |
| [GUIDELINES.md](paper1_benchmark/GUIDELINES.md) | Step-by-step experiment workflow |
| [references.yaml](paper1_benchmark/references.yaml) | SOTA baselines |
| [schema.json](paper1_benchmark/schema.json) | Validation schema |

---

### Paper 2 — LinguoMT-Adapt: PEFT Fine-Tuning

> Parameter-efficient fine-tuning (LoRA, adapters) applied to SeamlessM4T-v2. Answers how much PEFT recovers vs full fine-tuning and at what parameter budget.

**Mode:** `adaptation` | **Folder:** `paper2_adaptation/`

| File | Purpose |
|------|---------|
| [README.md](paper2_adaptation/README.md) | Abstract, RQs, data field reference |
| [GUIDELINES.md](paper2_adaptation/GUIDELINES.md) | Step-by-step experiment workflow |
| [references.yaml](paper2_adaptation/references.yaml) | SOTA baselines |
| [schema.json](paper2_adaptation/schema.json) | Validation schema |

---

### Paper 3 — LinguoMT-Audio: Preprocessing & Robustness

> Audio preprocessing ablation (VAD, normalisation, SpecAugment) and robustness under SNR degradation. No model fine-tuning except augmentation-trained robustness probes.

**Mode:** `audio` | **Folder:** `paper3_audio/`

| File | Purpose |
|------|---------|
| [README.md](paper3_audio/README.md) | Abstract, RQs, data field reference |
| [GUIDELINES.md](paper3_audio/GUIDELINES.md) | Step-by-step experiment workflow |
| [references.yaml](paper3_audio/references.yaml) | SOTA baselines |
| [schema.json](paper3_audio/schema.json) | Validation schema |

---

### Paper 4 — LinguoMT-Cascade: Architecture Comparison

> Controlled comparison of the Whisper + NLLB-200 cascade against end-to-end SeamlessM4T-v2. Includes error propagation analysis and inference latency.

**Mode:** `cascade` | **Folder:** `paper4_cascade/`

| File | Purpose |
|------|---------|
| [README.md](paper4_cascade/README.md) | Abstract, RQs, data field reference |
| [GUIDELINES.md](paper4_cascade/GUIDELINES.md) | Step-by-step experiment workflow |
| [references.yaml](paper4_cascade/references.yaml) | SOTA baselines |
| [schema.json](paper4_cascade/schema.json) | Validation schema |

---

### Paper 5 — LinguoMT-Transfer: Cross-Lingual Adaptation

> How linguistic family membership (Niger-Congo vs Afro-Asiatic) moderates cross-lingual transfer efficiency across zero-shot, few-shot, and fine-tuning regimes.

**Mode:** `transfer` | **Folder:** `paper5_transfer/`

| File | Purpose |
|------|---------|
| [README.md](paper5_transfer/README.md) | Abstract, RQs, data field reference |
| [GUIDELINES.md](paper5_transfer/GUIDELINES.md) | Step-by-step experiment workflow |
| [references.yaml](paper5_transfer/references.yaml) | SOTA baselines |
| [schema.json](paper5_transfer/schema.json) | Validation schema |

---

## Scope boundaries across papers

The table below shows which topic is owned by each paper to prevent duplication.

| Topic | P1 | P2 | P3 | P4 | P5 |
|-------|----|----|----|----|-----|
| Zero-shot multi-model evaluation | ✓ | — | — | — | — |
| PEFT efficiency (LoRA vs adapters) | — | ✓ | — | — | — |
| Audio preprocessing & noise robustness | — | — | ✓ | — | — |
| Cascade vs end-to-end architecture | — | — | — | ✓ | — |
| Cross-lingual transfer & typology | — | — | — | — | ✓ |
| FLEURS zero-shot baselines (numbers) | ✓ | cites P1 | cites P1 | cites P1 | cites P1 |
| Fine-tuning any model | — | ✓ | augmentation only | — | few-shot only |

---

## Shared citation index

`paper_references.csv` — lightweight index of all cited papers across the series.  
Columns: `citation_key, paper_title, authors, year, venue, url, notes`

Add one row here whenever you add a new entry to any paper's `references.yaml`.

---

## TOC maintenance

The table of contents at the top of this file and all `EXPERIMENTS.md` files is kept up to date automatically. Run:

```bash
# Update all TOC blocks once
python update_toc.py

# Watch for changes while editing
python update_toc.py --watch
```
