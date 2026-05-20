# Paper 1 — LinguoMT: Benchmarking Multilingual Speech Translation for Low-Resource African Languages

---

## Title

**LinguoMT: Benchmarking Multilingual Speech Translation for Low-Resource African Languages**

---

## Abstract

Despite rapid advances in multilingual speech and translation technology, African languages remain critically under-served by existing evaluation frameworks. No shared benchmark systematically compares modern speech translation systems on low-resource African languages under a unified, reproducible protocol. This paper introduces LinguoMT, a benchmark covering three typologically distinct African languages — Yoruba, Hausa, and Igbo — evaluated on the FLEURS dataset [conneau2022fleurs_baseline]. We assess state-of-the-art models including SeamlessM4T-v2 [barrault2023seamlessm4t_yoruba], Whisper-large-v3 [radford2023whisper_yoruba], MMS-300M [pratap2023mms_yoruba], mSLAM [bapna2022mslam_yoruba], XLS-R-1B [babu2022xlsr_yoruba], and NLLB-200 [costajussa2022nllb_yoruba] across ASR (WER/CER) and speech-to-text translation (BLEU [papineni2002bleu], spBLEU, ChrF) tasks. Our benchmark exposes systematic performance gaps between African and high-resource languages, provides a schema-validated result format for reproducible comparisons, and serves as a living baseline that future systems can be measured against.

---

## Chapter 1 — Introduction

### 1.1 Motivation

Spoken language is the dominant mode of communication for hundreds of millions of people across sub-Saharan Africa, yet automatic speech recognition and translation research has overwhelmingly focused on high-resource languages such as English, Mandarin, and major European languages. Massively multilingual models — SeamlessM4T [barrault2023seamlessm4t_yoruba], Whisper [radford2023whisper_yoruba], and MMS [pratap2023mms_yoruba] — claim coverage of hundreds of languages including several African ones, but published evaluations of these models on African languages are fragmented, use inconsistent metrics, and rarely compare more than two systems side-by-side. Without a shared evaluation framework, the field cannot reliably measure whether recent advances in multilingual speech technology genuinely extend to low-resource African languages or whether performance gaps persist undetected.

### 1.2 Problem Statement

There is no standardised, reproducible benchmark for evaluating multilingual speech translation on low-resource African languages. While FLEURS [conneau2022fleurs_baseline] provides a multilingual speech dataset, it does not prescribe a benchmark spanning multiple model families, tasks, and metrics simultaneously. Individual papers reporting results on Yoruba, Hausa, or Igbo — including mSLAM [bapna2022mslam_yoruba], XLS-R [babu2022xlsr_yoruba], and NLLB-200 [costajussa2022nllb_yoruba] — use heterogeneous evaluation setups, making cross-paper comparison unreliable. This lack of a common evaluation ground prevents researchers from identifying the largest performance gaps, hinders informed model selection for practitioners, and slows targeted investment in African-language speech technology.

### 1.3 Research Questions

**RQ1.** How do current state-of-the-art multilingual speech models — SeamlessM4T-v2 [barrault2023seamlessm4t_yoruba], Whisper-large-v3 [radford2023whisper_yoruba], and MMS-300M [pratap2023mms_yoruba] — compare in ASR word error rate and speech-to-text translation BLEU [papineni2002bleu] on Yoruba, Hausa, and Igbo under a unified evaluation protocol on the FLEURS dataset [conneau2022fleurs_baseline]?

**RQ2.** To what extent does performance on African languages lag behind performance on high-resource languages for the same models, and which language–task combinations exhibit the largest relative gaps across model families including mSLAM [bapna2022mslam_yoruba] and XLS-R [babu2022xlsr_yoruba]?

**RQ3.** Do cascade pipelines combining Whisper [radford2023whisper_yoruba] for ASR with NLLB-200 [costajussa2022nllb_yoruba] for translation achieve competitive speech-to-text translation BLEU [papineni2002bleu] compared to end-to-end models such as SeamlessM4T-v2 [barrault2023seamlessm4t_yoruba] on the three target African languages?

---

**Paper mode:** `benchmark`  
**Schema file:** `schema.json` — read by the framework to validate data before generating comparison tables.  
**Experiment guidelines:** [GUIDELINES.md](GUIDELINES.md) — step-by-step workflow for running and reporting all experiments.

---

## Data sources in this folder

| File | Format | Purpose |
|------|--------|---------|
| `references.yaml` | YAML list | Bibliography + comparison metadata (preferred) |
| `sota_results.csv` | CSV | Tabular baselines — one row per result |
| `published_baselines.json` | JSON | Tabular baselines — alternative format |

---

## Working with `references.yaml` (recommended)

Each entry in `references.yaml` is a bibliographic reference enriched with comparison fields.
Use one entry per paper × language combination.

**To activate a baseline in comparison tables: fill in `score`.**  
Entries with `score: null` are kept as bibliography context but excluded from tables.

### Entry format

```yaml
- citation_key: barrault2023seamlessm4t_yoruba   # unique key
  type: article                                   # article | inproceedings | misc
  author: "Barrault, Loïc and others"
  title: "SeamlessM4T: Massively Multilingual & Multimodal Machine Translation"
  year: 2023
  journal: "arXiv preprint arXiv:2308.11596"
  url: "https://arxiv.org/abs/2308.11596"

  # --- comparison fields ---
  model: SeamlessM4T-v2-large
  datasets: [FLEURS]                              # list — all datasets the paper evaluates on
  language: Yoruba                                # one entry per language
  directions: ["Source → English"]               # list — all directions reported
  metrics: [BLEU]                                # list — score corresponds to metrics[0]
  score: null                                    # fill in when you have the number
  summary: >
    End-to-end S2TT model trained on 100+ languages. Reports BLEU on FLEURS for African
    languages including Yoruba, Hausa, Igbo. Primary E2E baseline for the benchmark.
    Scores in Table B.1 of arXiv:2312.05187.
  notes: "Fill in BLEU from Table B.1 of arXiv:2312.05187"
```

### Reference fields

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `citation_key` | ★ Required | str | Unique identifier, one per paper × language |
| `type` | ★ Required | str | `article`, `inproceedings`, `misc` |
| `author` | ★ Required | str | Full author list as a single string |
| `title` | ★ Required | str | Full paper title |
| `year` | ★ Required | int | 4-digit publication year |
| `model` | ★ Required | str | Exact model name and size |
| `datasets` | ★ Required | list | All datasets the paper evaluates on, e.g. `[FLEURS]` |
| `language` | ★ Required | str | Target language name — `Yoruba`, `Hausa`, `Igbo` |
| `directions` | ★ Required | list | All task directions, e.g. `["Source → English", "ASR"]` |
| `metrics` | ★ Required | list | All metrics reported; `score` corresponds to `metrics[0]` |
| `score` | ★ Required | float\|null | Numeric score for `metrics[0]`; `null` = not yet filled |
| `summary` | ★ Required | str | 2–5 sentences on what the paper does and why it is relevant |
| `journal` | ○ Optional | str | Journal name (for `type: article`) |
| `booktitle` | ○ Optional | str | Conference name (for `type: inproceedings`) |
| `url` | ○ Optional | str | Link to the paper |
| `notes` | ○ Optional | str | Where to find the score, caveats, dataset differences |

> This paper uses only the base schema. No paper-specific comparison fields are required.

---

## Working with `sota_results.csv`

One row per system × language × metric result. All fields are flat (no lists).

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `paper_title` | ★ Required | str | Full paper title |
| `authors` | ★ Required | str | First author + et al. |
| `year` | ★ Required | int | 4-digit year |
| `model` | ★ Required | str | Model name and size |
| `dataset` | ★ Required | str | Dataset name (single value) |
| `language` | ★ Required | str | Display language name |
| `direction` | ★ Required | str | `Source → English` or `English → Source` |
| `metric` | ★ Required | str | `BLEU`, `spBLEU`, `ChrF`, `WER`, or `CER` |
| `score` | ★ Required | float | Numeric score — must not be empty |
| `citation_key` | ★ Required | str | Key matching an entry in `references.yaml` |
| `notes` | ○ Optional | str | Evaluation conditions, caveats |

---

## Published baselines to fill in

### Speech translation — BLEU

| Model | Authors | Year | Where to find scores |
|-------|---------|------|----------------------|
| SeamlessM4T-v2-large | Barrault et al. | 2023 | Table B.1 of `arXiv:2312.05187` (Yoruba, Hausa, Igbo) |
| mSLAM | Bapna et al. | 2022 | Supplementary FLEURS S2TT table of `arXiv:2202.01374` |
| mSLAM-CTC (FLEURS baseline) | Conneau et al. | 2022 | Table 3 of `arXiv:2205.12446` |

### ASR — WER

| Model | Authors | Year | Where to find scores |
|-------|---------|------|----------------------|
| Whisper-large-v3 | Radford et al. | 2023 | HuggingFace model card · FLEURS ASR eval |
| MMS-300M | Pratap et al. | 2023 | Per-language appendix of `arXiv:2305.13516` |
| XLS-R-1B | Babu et al. | 2022 | Table 4 of `arXiv:2111.09296` |

### Text-only MT — spBLEU (cascade upper bound, Flores-200 dataset)

| Model | Authors | Year | Where to find scores |
|-------|---------|------|----------------------|
| NLLB-200-distilled-600M | Costa-Jussà et al. | 2022 | Table 2 of `arXiv:2207.04672` |

**NLLB language codes:** `yor_Latn` (Yoruba) · `hau_Latn` (Hausa) · `ibo_Latn` (Igbo)  
**Note:** NLLB scores are on Flores-200 (written text), not FLEURS (speech) — label the direction as `Text MT (Source → English)`.

---

## Usage in run script

```python
PAPER_MODE = "benchmark"
SOTA_FILE  = "sota/paper1_benchmark/references.yaml"
```
