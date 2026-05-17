# Paper 5 — LinguoMT-Transfer: Cross-Lingual Adaptation in African Speech Translation

**Paper mode:** `transfer`  
**Schema file:** `schema.json` — read by the framework to validate your data before generating comparison tables.

---

## How to add results

1. Open `sota_results.csv` in this folder.
2. Add one row per system × language × **transfer type** × metric.
   Zero-shot and few-shot results for the same system should be separate rows.
3. Fill in every **★ Required** field. Leave **○ Optional** fields blank if unknown.
4. Add the citation key to `sota/paper_references.csv`.
5. Set `SOTA_FILE = "sota/paper5_transfer/sota_results.csv"` in your run script.

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
| `model` | ★ Required | str | Model name |
| `dataset` | ★ Required | str | Evaluation dataset name |
| `language` | ★ Required | str | **Target** language display name — `Yoruba`, `Hausa`, `Igbo` |
| `direction` | ★ Required | str | `Source → English` or `English → Source` |
| `metric` | ★ Required | str | `BLEU`, `ChrF`, `WER`, or `CER` |
| `score` | ★ Required | float | Numeric score on the target language — must not be empty |
| `citation_key` | ★ Required | str | BibTeX key also listed in `sota/paper_references.csv` |
| `notes` | ○ Optional | str | Transfer conditions, data sources, training details |

### Paper-specific fields

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `transfer_type` | ★ Required | str | `zero-shot`, `few-shot`, `cross-lingual`, or `multilingual` |
| `source_lang_family` | ○ Optional | str | Language family of training data: `Niger-Congo`, `Afro-Asiatic`, `mixed` |
| `target_lang_family` | ○ Optional | str | Language family of the evaluation language |
| `num_ft_samples` | ○ Optional | int | Adaptation samples used — `0` for zero-shot |
| `pretrain_langs` | ○ Optional | str | Languages seen during pretraining, e.g. `100 languages incl. Yoruba` |
| `lang_similarity_score` | ○ Optional | float | Typological similarity score (e.g. URIEL cosine), range 0–1 |

> **Language families for our languages:**
> - Yoruba → Niger-Congo (Volta-Niger)
> - Igbo → Niger-Congo (Volta-Niger) — related to Yoruba
> - Hausa → Afro-Asiatic (Chadic) — typologically distant from Yoruba/Igbo

> **Tip:** Add one row per transfer condition so the framework can build a learning-curve table:
> ```
> language=Igbo, transfer_type=zero-shot,  num_ft_samples=0,   WER=0.82
> language=Igbo, transfer_type=few-shot,   num_ft_samples=100, WER=0.65
> language=Igbo, transfer_type=few-shot,   num_ft_samples=500, WER=0.51
> ```

---

## What papers to look for

### Zero-shot multilingual models

| Model | Authors | Year | arXiv | Coverage |
|-------|---------|------|-------|----------|
| MMS-300M | Pratap et al. | 2023 | `arXiv:2305.13516` | 1,100+ languages incl. Yoruba, Hausa, Igbo |
| mSLAM | Bapna et al. | 2022 | `arXiv:2202.01374` | 51 languages — FLEURS S2TT |
| USM | Zhang et al. | 2023 | `arXiv:2303.01037` | 300+ languages |
| XLS-R 1B | Babu et al. | 2022 | `arXiv:2111.09296` | 128 languages |

### African-specific transfer

| System | Authors | Year | arXiv |
|--------|---------|------|-------|
| MasakhaSpeech | Ogunremi et al. | 2023 | `arXiv:2311.06023` |
| AfriSpeech-200 | Olatunji et al. | 2023 | `arXiv:2209.00670` |

### Language similarity

| Resource | Description |
|----------|-------------|
| URIEL / lang2vec | Typological feature vectors — `github.com/antonisa/lang2vec` |
| ASJP | Phonological distance database |

---

## Usage in run script

```python
PAPER_MODE = "transfer"
SOTA_FILE  = "sota/paper5_transfer/sota_results.csv"
```
