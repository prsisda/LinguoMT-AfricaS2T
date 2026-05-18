# Paper 3 — LinguoMT-Audio: Audio Preprocessing and Robustness for African Speech Translation

**Paper mode:** `audio`  
**Schema file:** `schema.json` — read by the framework to validate data before generating comparison tables.

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
  notes: "Fill in WER from HuggingFace model card FLEURS eval"
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
