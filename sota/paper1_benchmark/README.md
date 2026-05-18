# Paper 1 — LinguoMT: Benchmarking Multilingual Speech Translation for Low-Resource African Languages

**Paper mode:** `benchmark`  
**Schema file:** `schema.json` — read by the framework to validate data before generating comparison tables.

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
| NLLB-200-600M | Costa-Jussà et al. | 2022 | Table 2 of `arXiv:2207.04672` |

**NLLB language codes:** `yor_Latn` (Yoruba) · `hau_Latn` (Hausa) · `ibo_Latn` (Igbo)  
**Note:** NLLB scores are on Flores-200 (written text), not FLEURS (speech) — label the direction as `Text MT (Source → English)`.

---

## Usage in run script

```python
PAPER_MODE = "benchmark"
SOTA_FILE  = "sota/paper1_benchmark/references.yaml"
```
