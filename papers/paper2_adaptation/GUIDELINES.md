# Paper 2 — LinguoMT-Adapt: Experiment Guidelines

PEFT fine-tuning on FLEURS. Zero-shot baselines come from Paper 1 — do not re-run them.

**Deliverables:** Table 1 (before/after), Table 2 (parameter budget), Table 3 (data scaling), Figures 1–2, paper_outline.filled.md.

---

## Key files

| File | Purpose |
|------|---------|
| `papers/experiment_setup.yaml` | Model IDs, language codes — shared reference |
| `papers/paper2_adaptation/config.yaml` | Run settings, data budgets, result key mapping |
| `papers/paper2_adaptation/baselines.csv` | Pre-adaptation baselines (imported from Paper 1) |
| `papers/paper2_adaptation/paper_outline.md` | Paper skeleton with `[RESULT:key]` placeholders |
| `papers/fill_results.py` | Fills placeholders from experiment output CSVs |

---

## Prerequisites

- [ ] Paper 1 zero-shot results available
- [ ] GPU with ≥ 24 GB VRAM (LoRA on SeamlessM4T-v2-large); 16 GB for WhisperNLLB
- [ ] `peft` installed: `pip install peft`
- [ ] `PAPER_MODE = "adaptation"` and `ENABLE_FINETUNING = True` in run scripts

---

## Step 1 — Import Paper 1 baselines

Copy zero-shot scores from Paper 1 into `baselines.csv`:

```bash
# Extract Paper 1 zero-shot scores and append to Paper 2 baselines
python -c "
import csv, shutil
src = 'papers/paper1_benchmark/baselines.csv'
dst = 'papers/paper2_adaptation/baselines.csv'
# open src, filter rows with score, append to dst with ft_method=none column
print('Copy Paper 1 scores into paper2 baselines.csv')
"
```

Or manually copy the SeamlessM4T-v2 BLEU and Whisper WER rows (after they are populated in Step 1 of Paper 1) into `baselines.csv` with a `ft_method: none` note.

---

## Step 2 — Configure run scripts

In each `run_experiment.py`, set:

```python
DEBUG_MODE             = False
PAPER_MODE             = "adaptation"
ENABLE_FINETUNING      = True
FINETUNING_METHOD      = "lora"     # change to "adapter" for second run
TEXT_FINETUNE_SAMPLES  = 1000
ASR_FINETUNE_SAMPLES   = 500
EVAL_BEFORE_AFTER      = True
SOTA_FILE              = "papers/paper2_adaptation/baselines.csv"
```

Language settings:
- `FLEURS__SeamlessM4Tv2`: `MANUAL_LANGUAGES = ["igbo", "yoruba", "swahili"]`
- `FLEURS__WhisperNLLB`:   `MANUAL_LANGUAGES = ["yoruba", "hausa", "swahili"]`

---

## Step 3 — Run LoRA fine-tuning (main experiment)

```bash
python FLEURS__SeamlessM4Tv2/notebooks/run_experiment.py   # SeamlessM4T + LoRA
python FLEURS__WhisperNLLB/notebooks/run_experiment.py     # Whisper+NLLB + LoRA
```

Record per language after fine-tuning: BLEU, WER, trainable parameter count.

---

## Step 4 — Run adapter fine-tuning (comparison)

Change `FINETUNING_METHOD = "adapter"` and re-run the same scripts with the same data budget (1000 samples, seed 42).

---

## Step 5 — Data scaling experiment

Run LoRA at all four data budgets (100, 500, 1000, all-train). One run per budget:

```python
# In run_experiment.py, change TEXT_FINETUNE_SAMPLES for each run:
TEXT_FINETUNE_SAMPLES = 100    # then 500, then 1000, then None (all)
ASR_FINETUNE_SAMPLES  = 50     # proportionally
```

---

## Step 6 — Fill in paper_outline.md

```bash
python papers/fill_results.py paper2_adaptation
```

Output: `papers/paper2_adaptation/paper_outline.filled.md`

---

## Step 7 — Validate

```bash
python -c "
from framework.sota import load_and_validate_sota
load_and_validate_sota('papers/paper2_adaptation/baselines.csv', 'papers/paper2_adaptation/schema.json')
print('Schema OK')
"
```

---

## Reporting checklist

- [ ] Table 1: Before/after BLEU and WER for LoRA and adapter, per language
- [ ] Table 2: Parameter budget — trainable %, BLEU gain, GPU hours
- [ ] Table 3: Data scaling — BLEU/WER at 100/500/1000/full per language
- [ ] Figure 1: Learning curves (samples vs WER/BLEU)
- [ ] Figure 2: Efficiency frontier — BLEU gain vs trainable parameters
- [ ] Minimum data threshold identified (paired t-test, p < 0.05)
- [ ] `paper_outline.filled.md` generated

---

## Scope reminder

- Do NOT run audio preprocessing experiments here → Paper 3
- Do NOT compare cascade vs E2E here → Paper 4
- Do NOT vary source language for cross-lingual transfer here → Paper 5
