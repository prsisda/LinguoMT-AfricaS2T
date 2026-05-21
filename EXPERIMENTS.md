# Running LinguoMT Experiments

This guide covers the complete workflow from experiment execution to results report generation.

---

## Repository layout

```
run_on_colab.ipynb                                  # Central runner — all 4 experiments
AfricanCeltic__SeamlessM4Tv2/notebooks/
  run_experiment.py                                 # Script (patched and run by notebook)
  run_on_colab.ipynb                                # Single-experiment Colab notebook
AfricanCeltic__WhisperNLLB/notebooks/               # Same structure
FLEURS__SeamlessM4Tv2/notebooks/                    # Same structure
FLEURS__WhisperNLLB/notebooks/                      # Same structure
papers/
  generate_report.py                                # Markdown results report generator
  extract_results.py                                # Maps metrics to result keys (local workflow)
  fill_results.py                                   # Fills paper_draft.md placeholders (local workflow)
  paper<N>_*/paper_draft.md                         # Paper templates with [RESULT:key] placeholders
framework/                                          # Shared evaluation, metrics, and analysis code
  scaling.py                                        # Paper 2 — data scaling experiment
  cascade_analysis.py                               # Paper 4 — oracle cascade, error propagation, latency
  transfer.py                                       # Paper 5 — typological similarity, cross-lingual transfer
```

---

## Language support matrix

| Experiment | Igbo | Yoruba | Hausa | Swahili | Notes |
|---|:---:|:---:|:---:|:---:|---|
| AfricanCeltic × SeamlessM4T-v2 | ✓ | ✓ | — | — | Hausa audio absent from African-Celtic dataset |
| AfricanCeltic × Whisper+NLLB | — | ✓ | ✓ | — | Igbo excluded: no Whisper language token |
| FLEURS × SeamlessM4T-v2 | ✓ | ✓ | — | ✓ | Hausa excluded: `hau` not in SeamlessM4T S2TT list; Swahili substituted |
| FLEURS × Whisper+NLLB | — | ✓ | ✓ | ✓ | Igbo excluded: no Whisper token; Swahili substituted for parity |

---

## Complete workflow overview

```
Step 1  Mount Google Drive
Step 2  Clone / update repo
Step 3  Install dependencies
Step 4  Configure (PAPER_MODE, DEBUG_MODE, ENABLE_FINETUNING, SCALING_BUDGETS, SOTA_FILE)
Step 5  Patch run_experiment.py files with config → run all experiments
Step 6  Consolidate metric CSVs from all output directories
Step 7  Generate Markdown results report  ← papers/generate_report.py
Step 8  Package (ZIP of report + tables + plots) → save to Drive → download
```

The results report (`papers/<paper_id>/results_report.md`) is a structured document to
**facilitate writing** — it is not a finished paper section. See the report disclaimer for details.

---

## Colab notebooks

### Root notebook — all 4 experiments

Open `run_on_colab.ipynb` in the repository root.

1. Set Runtime → **T4 GPU** (or A100).
2. Run **Step 1** (mount Drive) and **Step 2** (clone repo).
3. In **Step 4**, configure:
   - `PAPER_MODE` — uncomment the paper you are running
   - `DEBUG_MODE` — `True` for a fast smoke test, `False` for the full paper run
   - `ENABLE_FINETUNING` — `True` for Papers 2 and 3 only
   - `SCALING_BUDGETS` — Paper 2 only; e.g. `[100, 500, 1000, 0]`
   - `SOTA_FILE` — path to a SOTA CSV, or `""` to skip SOTA comparison
   - `EXPERIMENTS` — comment out any experiments to skip
4. Click **Runtime → Run all**.

After all steps complete you will have:
- `/content/outputs/*/metrics/` — raw metric CSVs per experiment
- `/content/outputs/consolidated_*/` — merged metrics across all experiments
- `papers/<paper_id>/results_report.md` — Markdown analysis report
- A ZIP download containing the report, tables, plots, and interpretations

### Individual experiment notebooks

Each experiment folder contains its own `notebooks/run_on_colab.ipynb`.
Use these when you want to run one pipeline at a time (useful on limited Colab sessions).

The workflow is identical to the root notebook except only one `run_experiment.py` is run.
The report generation step will note which result keys are absent if other experiments have
not yet been run — this is expected.

---

## Configuration reference

All settings are in the `# %% --- configuration ---` block at the top of each `run_experiment.py`.
The Colab notebook patches these values automatically before running.

### Run mode

| Variable | Default | Effect |
|---|---|---|
| `DEBUG_MODE` | `True` | `True` → scan fewer rows, run fewer samples (~10 min/experiment). `False` → full evaluation. |
| `FAST_MODE` | `False` | Forces `DEBUG_MODE = True`. |
| `RUN_FULL_GRID` | `True` | `False` → run Experiment_1 only even in full mode. |
| `FORCE_RERUN` | `False` | Re-download and rebuild the dataset cache. |

### Paper mode

`PAPER_MODE` controls which analyses and report sections are produced.

| Value | Paper | What it produces |
|---|---|---|
| `"benchmark"` | Paper 1 — LinguoMT | Zero-shot BLEU/ChrF/WER baselines + SOTA gap analysis |
| `"adaptation"` | Paper 2 — LinguoMT-Adapt | Before/after fine-tuning comparison + data scaling curves |
| `"audio"` | Paper 3 — LinguoMT-Audio | Audio strategy analysis (S2TT, ASR+MT, normalise, trim, chunk) |
| `"cascade"` | Paper 4 — LinguoMT-Cascade | Oracle cascade, error propagation, break-even WER, latency |
| `"transfer"` | Paper 5 — LinguoMT-Transfer | Typological similarity, cross-lingual transfer, few-shot scaling |

### Language override

```python
MANUAL_LANGUAGES = ["yoruba", "hausa"]  # fixed list — recommended for reproducibility
MANUAL_LANGUAGES = None                  # auto-detect from model + dataset capabilities
```

Values currently set in each script are documented in the language support matrix above.

### Fine-tuning (Papers 2 & 3)

```python
ENABLE_FINETUNING = True
FINETUNING_METHOD = "lora"      # lora | adapter | full
FINETUNE_TEXT_TRANSLATION      = True
FINETUNE_REVERSE_TRANSLATION   = True
FINETUNE_ASR                   = True
FINETUNE_DIRECT_SPEECH_TRANSLATION = False  # SeamlessM4T only
TEXT_FINETUNE_SAMPLES = 1000
ASR_FINETUNE_SAMPLES  = 500
ST_FINETUNE_SAMPLES   = 200
```

### Data scaling (Paper 2)

```python
SCALING_BUDGETS = [100, 500, 1000, 0]   # 0 = full train set; [] = disabled
```

### SOTA file

```python
SOTA_FILE = "sota/paper1_benchmark/sota_results.csv"   # include SOTA tables
SOTA_FILE = ""                                          # skip SOTA tables
```

The SOTA CSV must have columns: `system`, `language`, `BLEU`, `venue`, `year`.

---

## Per-paper setup

### Paper 1 — Benchmark (zero-shot baselines)

```python
DEBUG_MODE        = False
PAPER_MODE        = "benchmark"
ENABLE_FINETUNING = False
SOTA_FILE         = "sota/paper1_benchmark/sota_results.csv"
```

Run all four experiment scripts. The report compares zero-shot results across models,
languages, and directions, and computes the gap versus published SOTA systems.

### Paper 2 — Adaptation (fine-tuning)

```python
DEBUG_MODE        = False
PAPER_MODE        = "adaptation"
ENABLE_FINETUNING = True
FINETUNING_METHOD = "lora"
SCALING_BUDGETS   = [100, 500, 1000, 0]
```

The script runs evaluation before and after fine-tuning (`EVAL_BEFORE_AFTER = True`).
Data scaling curves are produced when `SCALING_BUDGETS` is non-empty.

### Paper 3 — Audio strategies

```python
DEBUG_MODE        = False
PAPER_MODE        = "audio"
ENABLE_FINETUNING = False
```

Both SeamlessM4T (end-to-end) and Whisper+NLLB (cascade) scripts are required.
The report compares S2TT, normalised, trimmed, and chunk-based audio paths.

### Paper 4 — Cascade vs end-to-end

```python
DEBUG_MODE = False
PAPER_MODE = "cascade"
```

Run both model families. The framework automatically runs the oracle cascade analysis
(gold transcript → NLLB MT) and error propagation experiments when `PAPER_MODE = "cascade"`.
For detailed latency and break-even analysis, also run:

```bash
python papers/paper4_cascade/run_cascade_analysis.py
```

### Paper 5 — Cross-lingual transfer

```python
DEBUG_MODE        = False
PAPER_MODE        = "transfer"
ENABLE_FINETUNING = False
```

Use FLEURS scripts (Igbo, Yoruba, Hausa all available across experiments).
The framework computes typological similarity via lang2vec and runs cross-lingual transfer
experiments automatically when `PAPER_MODE = "transfer"`.

---

## Results report

After Step 7, `papers/<paper_id>/results_report.md` contains:

1. **Run metadata** — date, experiments, languages, SOTA file
2. **Experiment overview** — language support matrix and coverage notes
3. **Results tables** — text MT (BLEU, ChrF), ASR (WER, CER), audio strategies
4. **Comparisons** — SeamlessM4T vs Whisper+NLLB per language
5. **SOTA comparison** — gap analysis against published systems (if SOTA file provided)
6. **Key observations** — paper-mode-specific discussion points derived from the data
7. **Narrative placeholders** — `[NARRATIVE:...]` sections to fill when authoring
8. **Appendix** — full raw metric tables

> **This report gives direction for writing — it does not replace systematic analysis.**
> All numbers must be verified against raw outputs. Discussion points are heuristics based on
> metric patterns; the intellectual interpretation, citations, and conclusions are the authors'
> responsibility. A thorough review of the SOTA literature is required before submitting.

---

## Running locally

```bash
# activate environment
source venv/bin/activate

# run a single experiment
python FLEURS__SeamlessM4Tv2/notebooks/run_experiment.py

# generate the report (after consolidating metrics)
python papers/generate_report.py benchmark \
  --consolidated-dir outputs/consolidated_<timestamp>/ \
  --sota-file sota/paper1_benchmark/sota_results.csv

# local post-processing (optional — maps to paper template keys)
python papers/extract_results.py paper1_benchmark
python papers/fill_results.py    paper1_benchmark
```

---

## Output structure

Each experiment run creates a timestamped directory:

```
outputs/
└── 2025-06-01_14-30-00_seamlessm4t_fleurs/
    ├── config.json                  # full run configuration
    ├── metrics/
    │   ├── text_metrics.csv         # BLEU, ChrF per language and direction
    │   ├── audio_metrics.csv        # BLEU, ChrF per strategy, language, direction
    │   └── asr_metrics.csv          # WER, CER per language
    ├── tables/                      # CSV and Markdown summary tables
    ├── plots/                       # PNG score and EDA charts
    ├── interpretations/             # Auto-generated text interpretation fragments
    ├── summaries/                   # Per-experiment summary Markdown
    ├── predictions/qualitative/     # Side-by-side source / reference / hypothesis
    └── monitoring/                  # Step-by-step timing log
```

After consolidation (Step 6):

```
outputs/consolidated_<timestamp>/
    ├── text_metrics_all.csv         # merged across all experiments
    ├── asr_metrics_all.csv
    ├── audio_metrics_all.csv
    └── metrics_summary.md           # Markdown overview table
```

After report generation (Step 7):

```
papers/<paper_id>/
    └── results_report.md            # complete analysis report for paper writing
```

---

## Quick checklist before a full paper run

- [ ] Runtime set to GPU (Colab) or `torch.cuda.is_available()` returns `True` (local)
- [ ] `DEBUG_MODE = False`
- [ ] `PAPER_MODE` set to the correct value
- [ ] `MANUAL_LANGUAGES` matches the supported list (see matrix above)
- [ ] `SOTA_FILE` set if you have a baseline CSV to compare against
- [ ] `ENABLE_FINETUNING` set as required by the paper
- [ ] `SCALING_BUDGETS` set for Paper 2 (or left `[]` for other papers)
