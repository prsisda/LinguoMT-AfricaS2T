# LinguoMT — SOTA Reference Data

This folder stores published baseline results and paper references for the LinguoMT journal series.
Each sub-folder corresponds to one paper and contains its own `schema.json` and `README.md`.

---

## Folder structure

```
sota/
├── README.md                    ← this file
├── paper_references.csv         ← shared BibTeX-style reference list (all papers)
├── paper1_benchmark/
│   ├── schema.json              ← field schema (read by the framework)
│   ├── README.md                ← instructions and field reference
│   ├── sota_results.csv         ← fill in your published baselines here
│   └── published_baselines.json ← alternative JSON format
├── paper2_adaptation/
├── paper3_audio/
├── paper4_cascade/
└── paper5_transfer/
```

---

## How the framework uses these folders

When you set `SOTA_FILE` in a run script, the framework does the following:

1. **Loads** the CSV or JSON file you specified.
2. **Finds** `schema.json` in the same folder.
3. **Validates** every row: required fields must be non-empty and `score` must be numeric.
4. **Skips** invalid rows and prints a warning for each one.
5. **Generates** SOTA comparison tables from the valid rows.

```python
# In your run script:
SOTA_FILE = "sota/paper1_benchmark/sota_results.csv"
```

---

## Schema system

Each paper folder has a `schema.json` that declares:

| Key | Meaning |
|-----|---------|
| `paper_specific_required` | Extra fields that must be filled in for this paper |
| `paper_specific_optional` | Extra fields that are useful but can be blank |
| `field_definitions` | Full description of every field (type, description) |

The base required fields are the same for all papers:

| Field | Type | Notes |
|-------|------|-------|
| `paper_title` | str | Full title |
| `authors` | str | First author + et al. |
| `year` | int | 4-digit year |
| `model` | str | Model name and size |
| `dataset` | str | Dataset name |
| `language` | str | Display name matching our system output |
| `direction` | str | `Source → English` or `English → Source` |
| `metric` | str | `BLEU`, `ChrF`, `WER`, or `CER` |
| `score` | float | Numeric score — must not be empty |
| `citation_key` | str | BibTeX key listed in `paper_references.csv` |

The base optional field is:

| Field | Type | Notes |
|-------|------|-------|
| `notes` | str | Evaluation conditions, caveats, split name |

Paper-specific required/optional fields are documented in each folder's `README.md`.

---

## Accepted file formats

| Format | Filename | When to use |
|--------|----------|-------------|
| CSV | `sota_results.csv` | Easiest to edit in Excel / Google Sheets |
| JSON | `published_baselines.json` | When copying from a script or structured source |

Both formats are validated against the same `schema.json`.
Rows with `null` or empty `score` are always skipped.

---

## Shared reference list

`paper_references.csv` is a master list of all cited papers across the series.
Add one row here whenever you add results to any paper folder.

Columns: `citation_key, paper_title, authors, year, venue, url, notes`
