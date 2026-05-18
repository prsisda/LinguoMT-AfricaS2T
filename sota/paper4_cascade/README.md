# Paper 4 — LinguoMT-Cascade: Cascade vs End-to-End Architectures for African Speech Translation

**Paper mode:** `cascade`  
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
Use one entry per paper × language × architecture combination.

**To activate a baseline in comparison tables: fill in `score`.**

### Entry format

```yaml
- citation_key: barrault2023seamlessm4t_e2e_yoruba
  type: article
  author: "Barrault, Loïc and others"
  title: "SeamlessM4T: Massively Multilingual & Multimodal Machine Translation"
  year: 2023
  journal: "arXiv preprint arXiv:2308.11596"
  url: "https://arxiv.org/abs/2308.11596"

  # --- comparison fields ---
  model: SeamlessM4T-v2-large
  datasets: [FLEURS]                             # list — all datasets evaluated on
  language: Yoruba
  directions: ["Source → English"]              # list — all task directions reported
  metrics: [BLEU]                               # list — score corresponds to metrics[0]
  score: null                                   # fill in E2E BLEU from Table B.1
  architecture: end_to_end                      # paper-specific required field
  summary: >
    SeamlessM4T-v2 E2E S2TT BLEU on FLEURS Yoruba. Used as the E2E upper bound for cascade
    comparison. The core question is whether our Whisper+NLLB cascade matches or exceeds
    this E2E performance on African languages.
  notes: "Fill in BLEU from Table B.1 of arXiv:2312.05187"
```

### Reference fields

#### Standard bibliographic

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `citation_key` | ★ Required | str | Unique identifier, one per paper × language × architecture |
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
| `model` | ★ Required | str | Full pipeline, e.g. `Whisper-large-v3 + NLLB-200-600M` |
| `datasets` | ★ Required | list | All datasets evaluated on, e.g. `[FLEURS]` or `[MuST-C]` |
| `language` | ★ Required | str | Target language — `Yoruba`, `Hausa`, `Igbo` |
| `directions` | ★ Required | list | All task directions, e.g. `["Source → English"]` |
| `metrics` | ★ Required | list | All metrics reported; `score` corresponds to `metrics[0]` |
| `score` | ★ Required | float\|null | Numeric score for `metrics[0]`; `null` = not yet filled |
| `summary` | ★ Required | str | 2–5 sentences on what the paper does and why it is relevant |
| `architecture` | ★ Required | str | `cascade`, `end_to_end`, or `direct` |
| `notes` | ○ Optional | str | Where to find the score, architecture details |
| `asr_model` | ○ Optional | str | ASR component in cascade, e.g. `Whisper-large-v3` |
| `mt_model` | ○ Optional | str | MT component in cascade, e.g. `NLLB-200-600M` |
| `asr_wer` | ○ Optional | float | Intermediate ASR WER before translation |
| `latency_ms` | ○ Optional | float | End-to-end inference latency in ms |

> **Architecture values:**
> - `cascade` — ASR + MT pipeline (separate components)
> - `end_to_end` — single model for direct speech-to-text translation (e.g. SeamlessM4T)
> - `direct` — direct ST without intermediate text (E2E non-autoregressive)

---

## Working with `sota_results.csv`

One row per system × language × architecture × metric. Flat fields — no lists.

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `paper_title` | ★ Required | str | Full paper title |
| `authors` | ★ Required | str | First author + et al. |
| `year` | ★ Required | int | 4-digit year |
| `model` | ★ Required | str | Full pipeline description |
| `dataset` | ★ Required | str | Dataset name |
| `language` | ★ Required | str | Display language name |
| `direction` | ★ Required | str | `Source → English` or `English → Source` |
| `metric` | ★ Required | str | `BLEU`, `spBLEU`, `ChrF`, `WER`, or `CER` |
| `score` | ★ Required | float | Numeric score |
| `citation_key` | ★ Required | str | Key matching an entry in `references.yaml` |
| `architecture` | ★ Required | str | `cascade`, `end_to_end`, or `direct` |
| `asr_model` | ○ Optional | str | ASR component (cascade only) |
| `mt_model` | ○ Optional | str | MT component (cascade only) |
| `asr_wer` | ○ Optional | float | Intermediate ASR WER |
| `latency_ms` | ○ Optional | float | Inference latency in ms |
| `notes` | ○ Optional | str | Architecture details |

---

## Published baselines to fill in

### End-to-end baselines (architecture: `end_to_end`)

| Model | Dataset | Languages | Metric | Where to find scores |
|-------|---------|-----------|--------|----------------------|
| SeamlessM4T-v2-large | FLEURS | Yoruba, Hausa | BLEU | Table B.1 of `arXiv:2312.05187` |

### MT ceiling — text-only (use as cascade upper bound)

| Model | Dataset | Languages | Metric | Where to find scores |
|-------|---------|-----------|--------|----------------------|
| NLLB-200-600M | Flores-200 | Igbo (`ibo_Latn`) | spBLEU | Table 2 of `arXiv:2207.04672` |

### Architecture framing references (cite, not compare directly)

| Paper | Relevance |
|-------|-----------|
| Bentivogli et al. 2021 (`arXiv:2106.01045`) | Cascade vs E2E analysis (European pairs) |
| ESPnet-ST — Inaguma et al. 2020 (`arXiv:2004.10234`) | Cascade pipeline toolkit reference |
| CoVoST-2 — Wang et al. 2021 (`arXiv:2007.10310`) | Multilingual ST cascade baseline (**check language coverage**) |

> **CoVoST-2 note:** Yoruba and Hausa are likely **not** in CoVoST-2. Verify Table 2 before
> filling in scores for those entries in `references.yaml`.

---

## Usage in run script

```python
PAPER_MODE = "cascade"
SOTA_FILE  = "sota/paper4_cascade/references.yaml"
```
