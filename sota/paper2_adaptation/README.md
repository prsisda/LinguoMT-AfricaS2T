# Paper 2 — LinguoMT-Adapt: Parameter-Efficient Fine-Tuning for African Speech Translation

**Paper mode:** `adaptation`  
**Schema file:** `schema.json` — read by the framework to validate your data before generating comparison tables.

---

## How to add results

1. Open `sota_results.csv` in this folder.
2. Add one row per system × language × direction × metric.
3. Fill in every **★ Required** field. Leave **○ Optional** fields blank if unknown.
4. Add the citation key to `sota/paper_references.csv`.
5. Set `SOTA_FILE = "sota/paper2_adaptation/sota_results.csv"` in your run script.

The framework validates against `schema.json` at load time. Rows with missing required fields
are skipped with a printed warning.

---

## Field reference

### Base fields (all papers)

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `paper_title` | ★ Required | str | Full paper title |
| `authors` | ★ Required | str | First author + et al. |
| `year` | ★ Required | int | 4-digit publication year |
| `model` | ★ Required | str | Base model + method, e.g. `Whisper-large-v3 + LoRA` |
| `dataset` | ★ Required | str | Evaluation dataset name |
| `language` | ★ Required | str | Display name matching our system — `Yoruba`, `Hausa`, `Igbo` |
| `direction` | ★ Required | str | `Source → English` or `English → Source` |
| `metric` | ★ Required | str | `BLEU`, `ChrF`, `WER`, or `CER` |
| `score` | ★ Required | float | Score **after** adaptation — must not be empty |
| `citation_key` | ★ Required | str | BibTeX key also listed in `sota/paper_references.csv` |
| `notes` | ○ Optional | str | Training conditions, data size, epochs |

### Paper-specific fields

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `ft_method` | ★ Required | str | Adaptation method: `LoRA`, `adapter`, `full`, `prefix`, `prompt` |
| `pretrained_score` | ○ Optional | float | Score **before** adaptation — enables before/after comparison in `T_SOTA4` |
| `num_ft_samples` | ○ Optional | int | Number of training samples used |
| `trainable_params_pct` | ○ Optional | float | % of parameters trained, e.g. `0.5` for 0.5% |
| `training_hours` | ○ Optional | float | GPU hours required for fine-tuning |

> **Tip:** If you have both pre- and post-adaptation scores from the same paper, fill in
> `pretrained_score` alongside `score`. The framework will build a before/after table automatically.

---

## What papers to look for

### PEFT for speech and translation

| Model/Method | Authors | Year | arXiv / Venue |
|--------------|---------|------|---------------|
| LoRA for Whisper (low-resource ASR) | Various | 2023–2024 | Search "LoRA Whisper low-resource" |
| Efficient Fine-Tuning of Whisper | Gandhe et al. | 2023 | INTERSPEECH 2023 |
| Adapter-based multilingual ASR | Thomas et al. | 2022 | INTERSPEECH 2022 |
| PEFT for SeamlessM4T | Various | 2024 | ArXiv search |

### African language-specific adaptation

| System | Authors | Year | arXiv / Venue |
|--------|---------|------|---------------|
| AfriSpeech-200 | Olatunji et al. | 2023 | `arXiv:2209.00670` — WER before/after FT |
| MasakhaSpeech | Ogunremi et al. | 2023 | `arXiv:2311.06023` |

---

## Usage in run script

```python
PAPER_MODE        = "adaptation"
ENABLE_FINETUNING = True
SOTA_FILE         = "sota/paper2_adaptation/sota_results.csv"
```
