# Paper 3 — LinguoMT-Audio: Audio Preprocessing and Robustness for African Speech Translation

---

## Title

**LinguoMT-Audio: Audio Preprocessing and Robustness for African Speech Translation**

---

## Abstract

Real-world African-language speech recordings differ substantially from the clean studio conditions assumed by most multilingual ASR and speech translation benchmarks. Background noise, microphone variation, bandwidth-limited telephony audio, and code-switching introduce acoustic conditions that large pre-trained models — including Whisper [radford2023whisper_robustness_yoruba] and SeamlessM4T-v2 [barrault2023seamlessm4t_audio_yoruba] — have not been systematically evaluated against for African languages. This paper investigates how audio preprocessing pipelines — including voice activity detection, audio normalisation, noise augmentation via SpecAugment [park2019specaugment], and signal-to-noise ratio filtering — affect ASR word error rate and speech-to-text translation BLEU on Yoruba, Hausa, and Igbo. Using Whisper [radford2023whisper_robustness_yoruba] and SeamlessM4T-v2 [barrault2023seamlessm4t_audio_yoruba] as backbone models, we characterise the robustness of state-of-the-art models under varied acoustic conditions and identify which preprocessing steps yield the largest performance recovery on the FLEURS dataset.

---

## Chapter 1 — Introduction

### 1.1 Motivation

Multilingual speech translation benchmarks almost universally evaluate models on clean, studio-recorded audio. However, the practical deployment of speech technology in African-language contexts — community radio broadcasts, mobile voice messages, clinic recordings, and field interviews — involves substantially noisier acoustic conditions. Whisper [radford2023whisper_robustness_yoruba], trained on 680,000 hours of diverse web audio, claims inherent robustness to noise and channel variation, while SeamlessM4T-v2 [barrault2023seamlessm4t_audio_yoruba] similarly draws on large-scale multilingual training. Yet neither system has been systematically stress-tested under realistic African-language acoustic conditions. Audio preprocessing — voice activity detection, normalisation, and data augmentation methods such as SpecAugment [park2019specaugment] — is a practical lever that can recover significant performance without modifying model weights. Understanding which preprocessing steps matter, and how much, is essential for practitioners building speech pipelines for African-language communities.

### 1.2 Problem Statement

There is no systematic study of how acoustic condition and audio preprocessing affect speech recognition and translation quality specifically for African languages. Existing robustness evaluations focus on English or high-resource European languages, and the clean-audio baselines reported for Yoruba and Hausa by Whisper [radford2023whisper_robustness_yoruba] and SeamlessM4T-v2 [barrault2023seamlessm4t_audio_yoruba] do not characterise degradation under noise. At the same time, augmentation methods such as SpecAugment [park2019specaugment] and hybrid acoustic modelling approaches [watanabe2017hybrid_ctc] have been validated primarily on English benchmarks. Without a robustness study on African languages, it is unknown whether standard preprocessing recipes transfer, or whether African-language phonology and dataset characteristics require different treatment.

### 1.3 Research Questions

**RQ1.** How does ASR word error rate for Whisper [radford2023whisper_robustness_yoruba] and S2TT BLEU for SeamlessM4T-v2 [barrault2023seamlessm4t_audio_yoruba] degrade on Yoruba and Hausa as a function of signal-to-noise ratio, compared to the clean-audio baseline on FLEURS?

**RQ2.** Which audio preprocessing steps — voice activity detection, loudness normalisation, or spectral augmentation [park2019specaugment] — yield the largest reduction in WER degradation under noisy conditions for African-language ASR?

**RQ3.** Does applying SpecAugment-style augmentation [park2019specaugment] during fine-tuning improve model robustness to unseen noise types for Yoruba and Hausa, compared to models fine-tuned on clean audio only?

---

**Paper mode:** `audio`  
**Schema file:** `schema.json` — read by the framework to validate data before generating comparison tables.  
**Experiment guidelines:** [GUIDELINES.md](GUIDELINES.md) — step-by-step workflow for running and reporting all experiments.

---

## Data sources in this folder

| File | Format | Purpose |
|------|--------|---------|
| `references.yaml` | YAML list | Bibliography + comparison metadata (preferred) |
| `sota_results.csv` | CSV | Tabular baselines — one row per result |

---

## Working with `references.yaml` (recommended)

Each entry is a bibliographic reference enriched with comparison fields.
Use one entry per paper × language × audio condition combination.

**To activate a baseline in comparison tables: fill in `score`.**

### Entry format

```yaml
- citation_key: radford2023whisper_robustness_yoruba
  type: inproceedings
  author: "Radford, Alec and Kim, Jong Wook and others"
  title: "Robust Speech Recognition via Large-Scale Weak Supervision"
  year: 2023
  booktitle: "Proceedings of ICML 2023"
  url: "https://arxiv.org/abs/2212.04356"

  # --- comparison fields ---
  model: Whisper-large-v3
  datasets: [FLEURS]                             # list — all datasets evaluated on
  language: Yoruba
  directions: [ASR]                              # list — all task directions reported
  metrics: [WER]                                 # list — score corresponds to metrics[0]
  score: null                                    # fill in clean-audio WER
  audio_condition: clean                         # paper-specific required field
  summary: >
    Whisper trained on 680k hours of diverse audio. Reports ASR WER on FLEURS clean audio
    for 100+ languages including Yoruba and Hausa. Used as the clean-audio baseline in
    LinguoMT-Audio — we measure WER degradation under noisy conditions vs this reference.
  notes: "Fill in WER from HuggingFace model card FLEURS ASR eval"
```

### Reference fields

#### Standard bibliographic

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `citation_key` | ★ Required | str | Unique identifier, one per paper × language × condition |
| `type` | ★ Required | str | `article`, `inproceedings`, `misc` |
| `author` | ★ Required | str | Full author list as a single string |
| `title` | ★ Required | str | Full paper title |
| `year` | ★ Required | int | 4-digit publication year |
| `journal` | ○ Optional | str | Journal name |
| `booktitle` | ○ Optional | str | Conference name |
| `url` | ○ Optional | str | Link to the paper |

#### Comparison

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `model` | ★ Required | str | Model name |
| `datasets` | ★ Required | list | All datasets the paper evaluates on, e.g. `[FLEURS]` |
| `language` | ★ Required | str | Target language — `Yoruba`, `Hausa`, `Igbo` |
| `directions` | ★ Required | list | All task directions, e.g. `[ASR]` or `["Source → English"]` |
| `metrics` | ★ Required | list | All metrics reported; `score` corresponds to `metrics[0]` |
| `score` | ★ Required | float\|null | Numeric score for `metrics[0]`; `null` = not yet filled |
| `summary` | ★ Required | str | 2–5 sentences on what the paper does and why it is relevant |
| `audio_condition` | ★ Required | str | `clean`, `noisy`, `augmented`, `vad_filtered`, `normalized` |
| `notes` | ○ Optional | str | Where to find the score, noise setup details |
| `noise_type` | ○ Optional | str | `babble`, `music`, `street`, `white`, `gaussian` |
| `snr_db` | ○ Optional | float | Signal-to-noise ratio in dB, e.g. `10.0` |
| `vad_applied` | ○ Optional | bool | Whether VAD pre-processing was applied |

> **Tip:** Add one entry per audio condition so the framework can build a robustness table.
> Use `clean` for published clean-audio results (the baseline) and add our own noisy/augmented
> results in `sota_results.csv`.

---

## Working with `sota_results.csv`

One row per system × language × audio condition × metric. Flat fields — no lists.

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `paper_title` | ★ Required | str | Full paper title |
| `authors` | ★ Required | str | First author + et al. |
| `year` | ★ Required | int | 4-digit year |
| `model` | ★ Required | str | Model name |
| `dataset` | ★ Required | str | Dataset name |
| `language` | ★ Required | str | Display language name |
| `direction` | ★ Required | str | `Source → English`, `English → Source`, or `ASR` |
| `metric` | ★ Required | str | `BLEU`, `spBLEU`, `ChrF`, `WER`, or `CER` |
| `score` | ★ Required | float | Numeric score |
| `citation_key` | ★ Required | str | Key matching an entry in `references.yaml` |
| `audio_condition` | ★ Required | str | `clean`, `noisy`, `augmented`, `normalized`, `chunked` |
| `noise_type` | ○ Optional | str | Noise type |
| `snr_db` | ○ Optional | float | Signal-to-noise ratio in dB |
| `vad_applied` | ○ Optional | bool | `true` or `false` |
| `sample_rate_hz` | ○ Optional | int | Audio sample rate, typically `16000` |
| `notes` | ○ Optional | str | Audio condition details |

---

## Published baselines to fill in

### Clean-audio baselines (condition: `clean`)

| Model | Dataset | Languages | Metric | Where to find scores |
|-------|---------|-----------|--------|----------------------|
| Whisper-large-v3 | FLEURS | Yoruba, Hausa | WER | HuggingFace model card FLEURS ASR eval |
| SeamlessM4T-v2-large | FLEURS | Yoruba, Hausa | BLEU | Table B.1 of `arXiv:2312.05187` |

### Methodology references (cite, not compare directly)

| Paper | Relevance |
|-------|-----------|
| SpecAugment — Park et al. 2019 (`arXiv:1904.08779`) | Audio augmentation method |
| Whisper long-form — Radford et al. 2023 | Chunking and VAD methodology |

---

## Usage in run script

```python
PAPER_MODE = "audio"
SOTA_FILE  = "sota/paper3_audio/references.yaml"
```
