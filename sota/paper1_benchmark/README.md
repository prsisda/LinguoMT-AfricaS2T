# Paper 1 — LinguoMT: Benchmarking Multilingual Speech Translation for Low-Resource African Languages

**Paper mode:** `benchmark`  
**Schema file:** `schema.json` — read by the framework to validate your data before generating comparison tables.

---

## How to add results

1. Open `sota_results.csv` (or `published_baselines.json`) in this folder.
2. Add one row per system × language × direction × metric.
3. Fill in every **★ Required** field. Leave **○ Optional** fields blank if unknown.
4. Add the citation key to `sota/paper_references.csv`.
5. Set `SOTA_FILE = "sota/paper1_benchmark/sota_results.csv"` in your run script.

The framework will validate your file against `schema.json` at load time and print a warning
for any row with a missing required field. Those rows are skipped automatically.

---

## Field reference

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `paper_title` | ★ Required | str | Full paper title as it appears in the publication |
| `authors` | ★ Required | str | First author + et al., e.g. `Barrault et al.` |
| `year` | ★ Required | int | 4-digit publication year, e.g. `2023` |
| `model` | ★ Required | str | Exact model name and size, e.g. `SeamlessM4T-v2-large` |
| `dataset` | ★ Required | str | `FLEURS` or `African-Celtic` |
| `language` | ★ Required | str | Display name matching our system — `Yoruba`, `Hausa`, `Igbo` |
| `direction` | ★ Required | str | `Source → English` or `English → Source` |
| `metric` | ★ Required | str | `BLEU`, `ChrF`, `WER`, or `CER` |
| `score` | ★ Required | float | Numeric score — **must not be empty** |
| `citation_key` | ★ Required | str | BibTeX key also listed in `sota/paper_references.csv` |
| `notes` | ○ Optional | str | Evaluation split, conditions, or caveats |

> This paper uses only the base schema. No paper-specific fields are required.

---

## What papers to look for

### Speech translation — BLEU / ChrF

| Model | Authors | Year | Where to find scores |
|-------|---------|------|----------------------|
| SeamlessM4T-v2-large | Barrault et al. | 2023 | HuggingFace model card · paper appendix (`arXiv:2312.05187`) |
| Seamless (v1) | Barrault et al. | 2023 | `arXiv:2308.11596` |
| AudioPaLM | Rubenstein et al. | 2023 | `arXiv:2306.12925` |
| USM | Zhang et al. | 2023 | `arXiv:2303.01037` |
| mSLAM | Bapna et al. | 2022 | `arXiv:2202.01374` — FLEURS S2TT table |
| FLEURS baseline | Conneau et al. | 2022 | `arXiv:2205.12446` — Table 3 |

### ASR — WER / CER

| Model | Authors | Year | Where to find scores |
|-------|---------|------|----------------------|
| Whisper large-v3 | Radford et al. | 2023 | HuggingFace model card · ICML 2023 paper |
| MMS-300M | Pratap et al. | 2023 | `arXiv:2305.13516` — Appendix Table |
| XLS-R 1B | Babu et al. | 2022 | `arXiv:2111.09296` |

### Text-only MT — BLEU (cascade upper-bound)

| Model | Authors | Year | Where to find scores |
|-------|---------|------|----------------------|
| NLLB-200-600M | Costa-Jussà et al. | 2022 | `arXiv:2207.04672` — Flores-200 table |
| M2M-100 | Fan et al. | 2021 | `arXiv:2010.11125` |

**NLLB language codes:** Yoruba = `yor_Latn`, Hausa = `hau_Latn`, Igbo = `ibo_Latn`

---

## Usage in run script

```python
PAPER_MODE = "benchmark"
SOTA_FILE  = "sota/paper1_benchmark/sota_results.csv"
```
