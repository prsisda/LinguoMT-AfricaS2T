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
  - [Generating the paper draft from Colab outputs](#generating-the-paper-draft-from-colab-outputs)
    - [Step 1 — Download experiment outputs from Colab](#step-1--download-experiment-outputs-from-colab)
    - [Step 2 — Copy outputs into the repo](#step-2--copy-outputs-into-the-repo)
    - [Step 3 — Extract results into results.csv](#step-3--extract-results-into-resultscsv)
    - [Step 4 — Fill the paper draft](#step-4--fill-the-paper-draft)
    - [Step 5 — Review and complete the draft](#step-5--review-and-complete-the-draft)
    - [Step 6 — Regenerate after re-running experiments](#step-6--regenerate-after-re-running-experiments)
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

## Documentation structure

Each paper follows a three-layer documentation system:

```
papers/
├── experiment_setup.yaml          ← shared: all model IDs, language codes, dataset splits
├── fill_results.py                ← script: fills paper_outline.md from experiment CSVs
└── paper1_benchmark/
    ├── config.yaml                ← what to run + expected table structure
    ├── baselines.csv              ← published SOTA scores (SOTA_FILE for run scripts)
    ├── paper_outline.md           ← paper skeleton with [RESULT:key] placeholders
    ├── paper_outline.filled.md    ← auto-generated after running experiments
    ├── GUIDELINES.md              ← step-by-step workflow
    ├── references.yaml            ← BibTeX-style reference records
    └── schema.json                ← validation schema
```

**Workflow for each paper:**
1. Fill `baselines.csv` with published SOTA scores (Step 1 of GUIDELINES.md)
2. Set script config from `config.yaml` and run experiments
3. Run `python papers/fill_results.py paper1_benchmark` → produces `paper_outline.filled.md`
4. Fill `[NARRATIVE:key]` sections manually in the filled outline

---

## Generating the paper draft from Colab outputs

This section walks through converting a finished Colab run into a filled paper draft.  
**Target length:** ≤ 10 pages / ≤ 4 500 words including tables, figures, and references.

### Step 1 — Download experiment outputs from Colab

After the Colab notebook finishes all experiments, run the download cell (Step 11 in `run_on_colab.ipynb`) to zip the outputs and download them:

```python
# Colab cell — create the archive
import shutil
shutil.make_archive("/content/colab_outputs", "zip", "/content/outputs")
```

Then in the Colab sidebar open **Files**, right-click `colab_outputs.zip`, and select **Download**.

### Step 2 — Copy outputs into the repo

Unzip the archive locally and copy each experiment folder into `results/<paper_id>/from_colab/`:

```
results/
└── paper1_benchmark/
    └── from_colab/
        ├── AfricanCeltic__SeamlessM4Tv2/   ← copy from colab_outputs/
        ├── AfricanCeltic__WhisperNLLB/
        ├── FLEURS__SeamlessM4Tv2/
        └── FLEURS__WhisperNLLB/
```

Each experiment folder must contain a run directory named `<mode>_full/` (e.g. `full_full/`) with at minimum:

| File | Contents |
|------|----------|
| `text_metrics.csv` | BLEU, spBLEU, ChrF per language and direction |
| `asr_metrics.csv` | WER, CER per language |
| `audio_metrics.csv` | Audio S2TT scores (if applicable) |

> **Note:** `results/**/from_colab/` is git-ignored. Do not commit raw Colab output files.

### Step 3 — Extract results into results.csv

```bash
python papers/extract_results.py paper1_benchmark
```

This scans all `from_colab/<experiment>/<mode>_full/` folders, maps metrics to standard result keys
(e.g. `fleurs_seamless.yoruba.bleu`, `ac_whisper.igbo.wer`), and writes:

```
results/paper1_benchmark/results.csv
```

Run with `--all` to process every paper at once.

### Step 4 — Fill the paper draft

```bash
python papers/fill_results.py paper1_benchmark
```

This reads `results.csv` and `papers/paper1_benchmark/baselines.csv`, substitutes all `[RESULT:key]`
and `[BASELINE:key]` placeholders in the master template, and writes:

```
papers/paper1_benchmark/paper_draft_filled.md
```

Placeholders with no matching result remain as `[TBD]`.

### Step 5 — Review and complete the draft

Open `papers/paper1_benchmark/paper_draft_filled.md` and do the following:

1. **Fill `[NARRATIVE:key]` sections** — these require human interpretation: analysis of score trends,
   discussion of why models succeed or fail per language, limitations.
2. **Replace remaining `[TBD]` markers** — results from experiments that have not run yet; re-run
   Steps 3–4 once the missing experiments complete.
3. **Check the length target:**
   ```bash
   wc -w papers/paper1_benchmark/paper_draft_filled.md
   ```
   Aim for ≤ 4 500 words. Trim table rows or merge sections if the count is higher.
4. **Do not edit `paper_draft.md`** — it is the master template; only `paper_draft_filled.md` is the
   working document.

### Step 6 — Regenerate after re-running experiments

If you re-run any experiment and copy new output folders in, repeat Steps 3–4.  
Write final narrative edits directly in `paper_draft_filled.md` only **after** all experiments are
complete — running `fill_results.py` again will overwrite `paper_draft_filled.md` and lose any
manually written text.

---

## Papers

Each paper builds on the previous one's results. The scope boundaries are strict — no paper duplicates experiments owned by another.

### Paper 1 — LinguoMT: Benchmark

> Zero-shot evaluation of all model families on FLEURS. Establishes the baseline numbers that all other papers compare against.

**Mode:** `benchmark` | **Folder:** `paper1_benchmark/`

| File | Purpose |
|------|---------|
| [config.yaml](paper1_benchmark/config.yaml) | Run settings, expected tables, result key mapping |
| [baselines.csv](paper1_benchmark/baselines.csv) | Published SOTA scores — partially verified, rest to fill |
| [paper_outline.md](paper1_benchmark/paper_outline.md) | Paper skeleton with placeholders |
| [GUIDELINES.md](paper1_benchmark/GUIDELINES.md) | Step-by-step experiment workflow |
| [references.yaml](paper1_benchmark/references.yaml) | BibTeX-style reference records |
| [schema.json](paper1_benchmark/schema.json) | Validation schema |

---

### Paper 2 — LinguoMT-Adapt: PEFT Fine-Tuning

> Parameter-efficient fine-tuning (LoRA, adapters) applied to SeamlessM4T-v2 and Whisper. Zero-shot baselines from Paper 1.

**Mode:** `adaptation` | **Folder:** `paper2_adaptation/`

| File | Purpose |
|------|---------|
| [config.yaml](paper2_adaptation/config.yaml) | Run settings, data budgets, fine-tuning hyperparams |
| [baselines.csv](paper2_adaptation/baselines.csv) | Pre-adaptation baselines (imported from Paper 1) |
| [paper_outline.md](paper2_adaptation/paper_outline.md) | Paper skeleton with placeholders |
| [GUIDELINES.md](paper2_adaptation/GUIDELINES.md) | Step-by-step experiment workflow |

---

### Paper 3 — LinguoMT-Audio: Audio Strategy Analysis

> Compare S2TT (direct), cascade (ASR+MT), and ASR-only audio processing strategies.

**Mode:** `audio` | **Folder:** `paper3_audio/`

| File | Purpose |
|------|---------|
| [config.yaml](paper3_audio/config.yaml) | Audio strategies defined, text-MT ceiling setup |
| [baselines.csv](paper3_audio/baselines.csv) | Clean-audio baselines (from Paper 1) |
| [paper_outline.md](paper3_audio/paper_outline.md) | Paper skeleton with placeholders |
| [GUIDELINES.md](paper3_audio/GUIDELINES.md) | Step-by-step experiment workflow |

---

### Paper 4 — LinguoMT-Cascade: Architecture Comparison

> Controlled comparison of cascade vs end-to-end. Error propagation, oracle, latency analysis.

**Mode:** `cascade` | **Folder:** `paper4_cascade/`

| File | Purpose |
|------|---------|
| [config.yaml](paper4_cascade/config.yaml) | Oracle setup, error propagation, latency config |
| [baselines.csv](paper4_cascade/baselines.csv) | Cascade and E2E baselines (from Paper 1) |
| [paper_outline.md](paper4_cascade/paper_outline.md) | Paper skeleton with placeholders |
| [GUIDELINES.md](paper4_cascade/GUIDELINES.md) | Step-by-step experiment workflow |

---

### Paper 5 — LinguoMT-Transfer: Cross-Lingual Adaptation

> Niger-Congo vs Afro-Asiatic: how linguistic family membership drives transfer efficiency.

**Mode:** `transfer` | **Folder:** `paper5_transfer/`

| File | Purpose |
|------|---------|
| [config.yaml](paper5_transfer/config.yaml) | Transfer pairs, few-shot budgets, URIEL setup |
| [baselines.csv](paper5_transfer/baselines.csv) | Zero-shot baselines (from Paper 1) |
| [paper_outline.md](paper5_transfer/paper_outline.md) | Paper skeleton with placeholders |
| [GUIDELINES.md](paper5_transfer/GUIDELINES.md) | Step-by-step experiment workflow |

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
