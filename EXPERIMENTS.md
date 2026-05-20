# Running LinguoMT Experiments

This guide explains how to configure and run the evaluation scripts for all five papers in the LinguoMT series, both locally and on Google Colab.

---

## Repository layout

```
AfricanCeltic__SeamlessM4Tv2/notebooks/run_experiment.py   # African-Celtic × SeamlessM4T-v2
AfricanCeltic__WhisperNLLB/notebooks/run_experiment.py     # African-Celtic × Whisper+NLLB
FLEURS__SeamlessM4Tv2/notebooks/run_experiment.py          # FLEURS × SeamlessM4T-v2
FLEURS__WhisperNLLB/notebooks/run_experiment.py            # FLEURS × Whisper+NLLB
```

Each folder also contains `run_on_colab.ipynb` for one-click Colab execution.

---

## Language support matrix

Not every language is available in every dataset, and not every model supports every language. The table below shows what is actually used in each experiment.

| Experiment | Igbo | Yoruba | Hausa | Reason for exclusions |
|---|:---:|:---:|:---:|---|
| AfricanCeltic × SeamlessM4T-v2 | ✓ | ✓ | — | Hausa audio absent from African-Celtic dataset |
| AfricanCeltic × Whisper+NLLB | — | ✓ | — | Igbo: Whisper has no Igbo language token; Hausa: absent from dataset |
| FLEURS × SeamlessM4T-v2 | ✓ | ✓ | ✓ | All three supported |
| FLEURS × Whisper+NLLB | — | ✓ | ✓ | Igbo: Whisper has no Igbo language token |

### Controlling languages via `MANUAL_LANGUAGES`

Every script exposes a `MANUAL_LANGUAGES` list near the top of the configuration block. Set it to control exactly which languages are evaluated:

```python
# Use a fixed list — recommended for reproducibility
MANUAL_LANGUAGES = ["igbo", "yoruba"]

# Set to None to auto-detect from model + dataset capabilities
MANUAL_LANGUAGES = None
```

**Important:** Always respect the exclusions above. Passing an unsupported language will either raise a runtime error or silently produce empty predictions.

```python
# African-Celtic × SeamlessM4T-v2 — correct
MANUAL_LANGUAGES = ["igbo", "yoruba"]       # Hausa must be omitted

# African-Celtic × Whisper+NLLB — correct
MANUAL_LANGUAGES = ["yoruba"]               # Igbo and Hausa must be omitted

# FLEURS × SeamlessM4T-v2 — correct
MANUAL_LANGUAGES = ["igbo", "yoruba", "hausa"]

# FLEURS × Whisper+NLLB — correct
MANUAL_LANGUAGES = ["yoruba", "hausa"]      # Igbo must be omitted
```

---

## Key configuration variables

All of these are in the `# %% --- configuration ---` cell at the top of each script.

### Run mode

| Variable | Values | Effect |
|---|---|---|
| `DEBUG_MODE` | `True` / `False` | `True` scans fewer rows and runs fewer samples — fast smoke-test (~10 min on GPU). `False` runs the full evaluation. |
| `FAST_MODE` | `True` / `False` | Forces `DEBUG_MODE = True`. Use when iterating quickly. |
| `RUN_FULL_GRID` | `True` / `False` | `False` runs only Experiment_1 even in full mode. |
| `FORCE_RERUN` | `True` / `False` | Re-downloads and re-builds the dataset cache even if it already exists. |

### Paper mode

`PAPER_MODE` selects the output emphasis so each run produces the tables and plots relevant to one paper. Set it before running.

| Value | Paper | What it produces |
|---|---|---|
| `"benchmark"` | Paper 1 — LinguoMT | Zero-shot baseline metrics + SOTA comparison tables |
| `"adaptation"` | Paper 2 — LinguoMT-Adapt | Before/after fine-tuning comparison |
| `"audio"` | Paper 3 — LinguoMT-Audio | Audio strategy (S2TT, ASR+MT, ASR-only) analysis |
| `"cascade"` | Paper 4 — LinguoMT-Cascade | Cascade vs end-to-end comparison |
| `"transfer"` | Paper 5 — LinguoMT-Transfer | Cross-lingual transfer (Niger-Congo vs Afro-Asiatic) |

```python
PAPER_MODE = "benchmark"   # change to the paper you are running
```

### SOTA comparison

```python
SOTA_FILE = ""                                       # skip SOTA tables
SOTA_FILE = "sota/paper1_benchmark/sota_results.csv" # include SOTA tables
```

Leave `SOTA_FILE` empty until you have a populated baseline file.

### Fine-tuning

Fine-tuning is disabled by default. Enable it only for Papers 2 and 3.

```python
ENABLE_FINETUNING = False   # True → run fine-tuning before evaluation

FINETUNING_METHOD = "lora"  # lora | adapter | full
FINETUNE_TEXT_TRANSLATION      = True
FINETUNE_REVERSE_TRANSLATION   = True
FINETUNE_ASR                   = True
FINETUNE_DIRECT_SPEECH_TRANSLATION = False  # SeamlessM4T only

TEXT_FINETUNE_SAMPLES = 1000
ASR_FINETUNE_SAMPLES  = 500
ST_FINETUNE_SAMPLES   = 200
```

---

## Per-paper setup

### Paper 1 — Benchmark (zero-shot baselines)

Goal: evaluate pretrained models with no fine-tuning.

```python
DEBUG_MODE        = False
PAPER_MODE        = "benchmark"
ENABLE_FINETUNING = False
SOTA_FILE         = "sota/paper1_benchmark/sota_results.csv"
```

Run all four experiment scripts. Collect the output tables from each `outputs/` folder.

---

### Paper 2 — Adaptation (fine-tuning)

Goal: measure how much domain-specific fine-tuning improves results.

```python
DEBUG_MODE        = False
PAPER_MODE        = "adaptation"
ENABLE_FINETUNING = True
FINETUNING_METHOD = "lora"
```

The script runs evaluation before and after fine-tuning (`EVAL_BEFORE_AFTER = True`).

---

### Paper 3 — Audio strategies

Goal: compare speech-translation (S2TT), cascade (ASR→MT), and ASR-only paths.

```python
DEBUG_MODE        = False
PAPER_MODE        = "audio"
ENABLE_FINETUNING = False   # or True to also fine-tune audio paths
```

Both SeamlessM4T (end-to-end) and Whisper+NLLB (cascade) scripts are needed for a complete comparison.

---

### Paper 4 — Cascade vs end-to-end

Goal: systematically compare the cascade pipeline against the end-to-end model.

```python
DEBUG_MODE = False
PAPER_MODE = "cascade"
```

Run the SeamlessM4T scripts (end-to-end) and the WhisperNLLB scripts (cascade) on the same dataset for a fair comparison.

---

### Paper 5 — Cross-lingual transfer

Goal: analyse how performance differs between typological language families (Niger-Congo: Yoruba, Igbo vs Afro-Asiatic: Hausa).

```python
DEBUG_MODE        = False
PAPER_MODE        = "transfer"
ENABLE_FINETUNING = False
```

Use FLEURS scripts (all three languages available). The AfricanCeltic scripts can supplement Yoruba/Igbo results.

---

## Running locally

```bash
# activate your environment first
source venv/bin/activate

# run any script directly
python AfricanCeltic__SeamlessM4Tv2/notebooks/run_experiment.py
python FLEURS__WhisperNLLB/notebooks/run_experiment.py
```

Outputs are written to `<experiment>/outputs/<timestamp>_<model>_<dataset>_<debug|full>/`.

---

## Running on Google Colab

1. Open the corresponding `run_on_colab.ipynb` in the experiment folder.
2. Set the runtime to **GPU** (Runtime → Change runtime type → T4 or A100).
3. In **Step 3 — Select Mode**, set `DEBUG_MODE` to `True` for a test run or `False` for the full paper run.
4. Run All (Runtime → Run all).

The notebook clones the repo, patches the script with your chosen settings, and runs it. Outputs are saved to Google Drive and offered as a download.

---

## Output structure

Each run creates a timestamped folder:

```
outputs/
└── 2025-06-01_14-30-00_seamlessm4t_african_celtic_debug/
    ├── config.json          # full run configuration
    ├── metrics/             # BLEU, chrF, WER scores per language and experiment
    ├── predictions/         # hypothesis and reference text files
    ├── tables/              # CSV and markdown summary tables
    ├── plots/               # score plots and EDA charts
    ├── interpretations/     # generated interpretation text
    ├── summaries/           # per-experiment summaries
    ├── monitoring/          # step-by-step timing log
    └── predictions/qualitative/  # side-by-side example outputs
```

---

## Quick checklist before a full paper run

- [ ] Set `DEBUG_MODE = False`
- [ ] Set `PAPER_MODE` to the correct paper value
- [ ] Set `MANUAL_LANGUAGES` to the supported list for this experiment (see matrix above)
- [ ] Set `SOTA_FILE` if you have a baseline file to compare against
- [ ] Set `ENABLE_FINETUNING` as required by the paper
- [ ] Confirm the runtime has a GPU (Colab) or `torch.cuda.is_available()` returns `True` (local)
