# LinguoMT — SOTA Reference Data

This folder stores published baseline results and paper references for the LinguoMT journal series.
Each sub-folder corresponds to one paper and contains its own `schema.json`, `README.md`,
`references.yaml`, and tabular result files.

---

## Folder structure

```
sota/
├── README.md                    ← this file
├── paper_references.csv         ← shared citation index (all papers)
├── paper1_benchmark/
│   ├── schema.json              ← field schema (read by the framework)
│   ├── README.md                ← field reference and instructions
│   ├── references.yaml          ← comparison baselines in YAML format
│   ├── sota_results.csv         ← tabular results (CSV format)
│   └── published_baselines.json ← tabular results (JSON format)
├── paper2_adaptation/
├── paper3_audio/
├── paper4_cascade/
└── paper5_transfer/
```

---

## How the framework uses these folders

When you set `SOTA_FILE` in a run script, the framework does the following:

1. **Loads** the file you specified (`.yaml`, `.csv`, or `.json`).
2. **Finds** `schema.json` in the same folder.
3. **Validates** every entry: required fields must be present and `score` must be numeric.
4. **Skips** invalid entries and prints a warning for each one — never stops training.
5. **Generates** SOTA comparison tables from the valid entries.

```python
# In your run script (point at whichever format you prefer):
SOTA_FILE = "sota/paper1_benchmark/references.yaml"
# or:
SOTA_FILE = "sota/paper1_benchmark/sota_results.csv"
```

---

## Two complementary data sources

### 1. `references.yaml` — bibliography + comparison metadata

The preferred format. Each entry is a full bibliographic reference enriched with comparison
fields. One entry per paper × language combination.

**Advantages:** single source of truth for both citation and comparison data; readable in the
IDE; version-controlled alongside the code.

**Fill in `score` to activate comparison tables.** Entries with `score: null` are kept as
bibliography context but excluded from comparison tables.

### 2. `sota_results.csv` / `published_baselines.json` — tabular baselines

Flat tabular format, one row per system × language × metric result.
Use when copying results in bulk from a spreadsheet or structured source.

Both formats are validated against the same `schema.json`.

---

## `references.yaml` format

Each file is a YAML list. Every entry has:

### Standard bibliographic fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `citation_key` | ★ | str | Unique key, e.g. `barrault2023seamlessm4t_yoruba` |
| `type` | ★ | str | `article`, `inproceedings`, `book`, `misc` |
| `author` | ★ | str | Full author list as a single string |
| `title` | ★ | str | Full paper title |
| `year` | ★ | int | 4-digit publication year |
| `journal` | ○ | str | Journal name (for `type: article`) |
| `booktitle` | ○ | str | Conference/workshop name (for `type: inproceedings`) |
| `url` | ○ | str | Link to the paper |

### Comparison fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `model` | ★ | str | Exact model name, e.g. `SeamlessM4T-v2-large` |
| `datasets` | ★ | list | All datasets the paper evaluates on, e.g. `[FLEURS]` |
| `language` | ★ | str | Target language name. **One entry per language.** |
| `directions` | ★ | list | All task directions reported, e.g. `["Source → English", "ASR"]` |
| `metrics` | ★ | list | All metrics reported, e.g. `[BLEU, WER]`. `score` corresponds to `metrics[0]`. |
| `score` | ★ | float\|null | Numeric score for `metrics[0]`. Set `null` until you fill it in. |
| `summary` | ★ | str | 2–5 sentences: what the paper does, which languages/datasets/metrics it covers, why it is a relevant comparator. |
| `notes` | ○ | str | Where to find the score, caveats, dataset differences. |

Paper-specific comparison fields (e.g. `ft_method`, `audio_condition`, `architecture`,
`transfer_type`) are documented in each folder's `README.md` and declared in `schema.json`.

### Example entry

```yaml
- citation_key: barrault2023seamlessm4t_yoruba
  type: article
  author: "Barrault, Loïc and Chung, Yu-An and others"
  title: "SeamlessM4T: Massively Multilingual & Multimodal Machine Translation"
  year: 2023
  journal: "arXiv preprint arXiv:2308.11596"
  url: "https://arxiv.org/abs/2308.11596"
  model: SeamlessM4T-v2-large
  datasets: [FLEURS]
  language: Yoruba
  directions: ["Source → English"]
  metrics: [BLEU]
  score: null          # fill in when you have the number
  summary: >
    End-to-end S2TT model trained on 100+ languages. Reports BLEU on FLEURS for African
    languages including Yoruba, Hausa, Igbo. Relevant as the primary E2E baseline.
    Scores in Table B.1 of arXiv:2312.05187.
  notes: "Fill in BLEU from Table B.1 of arXiv:2312.05187"
```

---

## Schema system

Each paper folder has a `schema.json` with two sections:

| Section | Purpose |
|---------|---------|
| `field_definitions` | Validates tabular results (`sota_results.csv` / `.json`) |
| `reference_fields` | Validates entries in `references.yaml` |

Both sections declare required vs optional fields and their types. The framework validates
against these schemas and prints warnings but **never raises errors or stops training**.

---

## Shared citation index

`paper_references.csv` is a lightweight index of all cited papers across the series.

Columns: `citation_key, paper_title, authors, year, venue, url, notes`

Add one row here when you add a new entry to any `references.yaml`.
