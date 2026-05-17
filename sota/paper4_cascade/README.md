# Paper 4 — LinguoMT-Cascade: Cascade vs End-to-End Architectures for African Speech Translation

**Paper mode:** `cascade`  
**Schema file:** `schema.json` — read by the framework to validate your data before generating comparison tables.

---

## How to add results

1. Open `sota_results.csv` in this folder.
2. Add one row per system × language × **architecture type** × metric.
   Cascade and end-to-end results for the same language should be separate rows.
3. Fill in every **★ Required** field. Leave **○ Optional** fields blank if unknown.
4. Add the citation key to `sota/paper_references.csv`.
5. Set `SOTA_FILE = "sota/paper4_cascade/sota_results.csv"` in your run script.

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
| `model` | ★ Required | str | Full pipeline name, e.g. `Whisper-large-v3 + NLLB-200` |
| `dataset` | ★ Required | str | Evaluation dataset name |
| `language` | ★ Required | str | Display name matching our system — `Yoruba`, `Hausa`, `Igbo` |
| `direction` | ★ Required | str | `Source → English` or `English → Source` |
| `metric` | ★ Required | str | `BLEU`, `ChrF`, `WER`, or `CER` |
| `score` | ★ Required | float | Numeric score — must not be empty |
| `citation_key` | ★ Required | str | BibTeX key also listed in `sota/paper_references.csv` |
| `notes` | ○ Optional | str | Architecture details, beam size, rescoring |

### Paper-specific fields

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `architecture` | ★ Required | str | `cascade`, `end-to-end`, or `direct` |
| `asr_model` | ○ Optional | str | ASR component for cascade pipelines, e.g. `Whisper-large-v3` |
| `mt_model` | ○ Optional | str | MT component for cascade pipelines, e.g. `NLLB-200-600M` |
| `asr_wer` | ○ Optional | float | Intermediate ASR WER before translation (cascade only) |
| `latency_ms` | ○ Optional | float | End-to-end inference latency in milliseconds, if reported |

> **Tip:** The framework uses `architecture` to group rows in `T_SOTA3` (system ranking).
> Always set it so cascade and end-to-end results appear as separate groups.

---

## What papers to look for

### IWSLT shared task results (best direct comparison source)

| Source | Year | URL |
|--------|------|-----|
| IWSLT 2023 findings | 2023 | `aclanthology.org/2023.iwslt` |
| IWSLT 2022 findings | 2022 | `aclanthology.org/2022.iwslt` |
| IWSLT 2021 findings | 2021 | `aclanthology.org/2021.iwslt` |

### End-to-end speech translation

| Model | Authors | Year | arXiv |
|-------|---------|------|-------|
| SeamlessM4T-v2 | Barrault et al. | 2023 | `arXiv:2312.05187` |
| ESPnet-ST | Inaguma et al. | 2020 | `arXiv:2004.10234` |
| FairSeq S2T | Wang et al. | 2020 | `arXiv:2010.05171` |

### CoVoST-2 benchmark (cascade vs E2E)

| Model | Authors | Year | arXiv |
|-------|---------|------|-------|
| CoVoST 2 baselines | Wang et al. | 2021 | `arXiv:2007.10310` |

### Error propagation papers

| Paper | Authors | Year | Venue |
|-------|---------|------|-------|
| On the Impact of ASR Errors on ST | Ruiz et al. | 2023 | EACL 2023 |
| Error Propagation in Cascade ST | Sperber et al. | 2019 | EMNLP 2019 |

---

## Usage in run script

```python
PAPER_MODE = "cascade"
SOTA_FILE  = "sota/paper4_cascade/sota_results.csv"
```
