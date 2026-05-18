# Paper 2 — LinguoMT-Adapt: Parameter-Efficient Fine-Tuning for African Speech Translation

**Paper mode:** `adaptation`  
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
Use one entry per paper × language combination.

**To activate a baseline in comparison tables: fill in `score`.**

### Entry format

```yaml
- citation_key: dossou2022masakhaspeech_hausa
  type: inproceedings
  author: "Dossou, Bonaventure F. P. and Emezue, Chris Chinenye and others"
  title: "MasakhaSpeech: End-to-End Speech Recognition for 5 African Languages"
  year: 2022
  booktitle: "Proceedings of Interspeech 2022"
  url: "https://arxiv.org/abs/2206.00253"

  # --- comparison fields ---
  model: "Wav2Vec2-XLSR (fine-tuned)"
  datasets: [MasakhaSpeech]                      # list — all datasets evaluated on
  language: Hausa
  directions: [ASR]                              # list — all task directions reported
  metrics: [WER]                                 # list — score corresponds to metrics[0]
  score: null                                    # fill in WER from Table 2
  ft_method: full_finetune                       # paper-specific required field
  summary: >
    Fine-tunes Wav2Vec2-XLSR on 5 African languages (~10h each). Direct adaptation baseline
    for Hausa ASR fine-tuning. Shows what full fine-tuning achieves with limited data.
  notes: "Fill in Hausa WER from Table 2 of arXiv:2206.00253"
```

### Reference fields

#### Standard bibliographic

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `citation_key` | ★ Required | str | Unique identifier, one per paper × language |
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
| `model` | ★ Required | str | Base model + adaptation method, e.g. `Whisper-large-v3 + LoRA` |
| `datasets` | ★ Required | list | All datasets the paper evaluates on |
| `language` | ★ Required | str | Target language — `Yoruba`, `Hausa`, `Igbo` |
| `directions` | ★ Required | list | All task directions, e.g. `[ASR]` or `["Source → English"]` |
| `metrics` | ★ Required | list | All metrics reported; `score` corresponds to `metrics[0]` |
| `score` | ★ Required | float\|null | Numeric score for `metrics[0]`; `null` = not yet filled |
| `summary` | ★ Required | str | 2–5 sentences on what the paper does and why it is relevant |
| `ft_method` | ★ Required | str | Adaptation method: `lora`, `adapter`, `full_finetune`, `prefix`, `none` |
| `notes` | ○ Optional | str | Where to find the score, training conditions |
| `pretrained_score` | ○ Optional | float | Score **before** adaptation — enables before/after comparison |
| `num_ft_samples` | ○ Optional | int | Number of training samples used for adaptation |
| `trainable_params_pct` | ○ Optional | float | % of parameters trained, e.g. `0.5` for 0.5% |

> **Tip:** If you have both pre- and post-adaptation scores from the same paper, fill in
> `pretrained_score` alongside `score`. The framework builds a before/after table automatically.

> **Zero-shot baselines** (no fine-tuning, e.g. SeamlessM4T pre-trained only): set `ft_method: none`.

---

## Working with `sota_results.csv`

One row per system × language × metric. Flat fields — no lists.

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `paper_title` | ★ Required | str | Full paper title |
| `authors` | ★ Required | str | First author + et al. |
| `year` | ★ Required | int | 4-digit year |
| `model` | ★ Required | str | Base model + adaptation method |
| `dataset` | ★ Required | str | Dataset name |
| `language` | ★ Required | str | Display language name |
| `direction` | ★ Required | str | `Source → English` or `English → Source` |
| `metric` | ★ Required | str | `BLEU`, `spBLEU`, `ChrF`, `WER`, or `CER` |
| `score` | ★ Required | float | Score after adaptation |
| `citation_key` | ★ Required | str | Key matching an entry in `references.yaml` |
| `ft_method` | ★ Required | str | Adaptation method |
| `pretrained_score` | ○ Optional | float | Score before adaptation |
| `num_ft_samples` | ○ Optional | int | Training samples used |
| `trainable_params_pct` | ○ Optional | float | % of parameters trained |
| `training_hours` | ○ Optional | float | GPU hours |
| `notes` | ○ Optional | str | Training conditions |

---

## Published baselines to fill in

### Zero-shot baselines (no fine-tuning)

| Model | Dataset | Languages | Metric | Where to find scores |
|-------|---------|-----------|--------|----------------------|
| SeamlessM4T-v2-large | FLEURS | Yoruba, Hausa | BLEU | Table B.1 of `arXiv:2312.05187` |

### ASR fine-tuning baselines

| Model/Method | Dataset | Languages | Metric | Where to find scores |
|--------------|---------|-----------|--------|----------------------|
| Wav2Vec2-XLSR (full FT) | MasakhaSpeech | Hausa, Yoruba | WER | Table 2 of `arXiv:2206.00253` |
| Whisper-large-v2 (full FT) | AfriSpeech-200 | African-accented English | WER | Table 4 of `arXiv:2104.02010` |

### Methodology references (cite, not compare)

| Paper | Purpose |
|-------|---------|
| LoRA — Hu et al. 2022 (`arXiv:2106.09685`) | LoRA adaptation method |
| Adapter — Bapna & Firat 2019 (`arXiv:1909.08478`) | Adapter method baseline |

---

## Usage in run script

```python
PAPER_MODE        = "adaptation"
ENABLE_FINETUNING = True
SOTA_FILE         = "sota/paper2_adaptation/references.yaml"
```
