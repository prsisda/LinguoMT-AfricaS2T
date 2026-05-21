# Paper 3 — LinguoMT-Audio: Experiment Guidelines

Audio strategy analysis: S2TT (direct), cascade (ASR+MT), and ASR-only.
Clean-audio baselines come from Paper 1. No PEFT here.

**Deliverables:** Table 1 (strategy comparison), Table 2 (ASR quality effect), paper_outline.filled.md.

---

## Key files

| File | Purpose |
|------|---------|
| `papers/experiment_setup.yaml` | Model IDs, language codes — shared reference |
| `papers/paper3_audio/config.yaml` | Audio strategies defined, result key mapping |
| `papers/paper3_audio/baselines.csv` | Clean-audio baselines (from Paper 1) |
| `papers/paper3_audio/paper_outline.md` | Paper skeleton with `[RESULT:key]` placeholders |
| `papers/fill_results.py` | Fills placeholders from experiment output CSVs |

---

## Prerequisites

- [ ] Paper 1 clean-audio results available
- [ ] FLEURS validation split accessible (16 kHz audio)
- [ ] `PAPER_MODE = "audio"` in run scripts

---

## Step 1 — Import clean-audio baselines from Paper 1

The clean-audio baselines for Paper 3 are the same as Paper 1 results.
Copy the SeamlessM4T-v2 BLEU and Whisper WER rows from Paper 1 into `baselines.csv`.

---

## Step 2 — Configure run scripts

In each `run_experiment.py`, set:

```python
DEBUG_MODE        = False
PAPER_MODE        = "audio"
ENABLE_FINETUNING = False
SOTA_FILE         = "papers/paper3_audio/baselines.csv"
```

Language settings:
- `FLEURS__SeamlessM4Tv2`: `MANUAL_LANGUAGES = ["igbo", "yoruba", "swahili"]`
- `FLEURS__WhisperNLLB`:   `MANUAL_LANGUAGES = ["yoruba", "hausa", "swahili"]`

---

## Step 3 — Run all audio strategies

```bash
python FLEURS__SeamlessM4Tv2/notebooks/run_experiment.py   # S2TT + ASR
python FLEURS__WhisperNLLB/notebooks/run_experiment.py     # ASR + cascade + text-MT ceiling
```

The `audio` paper_mode activates all three strategy paths in the ExperimentRunner.

### Text-MT ceiling (oracle)

To establish the text-MT ceiling, run NLLB-200 on the **gold reference transcripts** from FLEURS (not on audio):

```python
# Collect FLEURS validation set reference text per language
# Run NLLB-200 on gold text → English
# Record BLEU as the upper bound for any cascade with this MT component
```

This is the maximum BLEU any ASR+MT cascade can achieve given NLLB-200 as its MT component.

---

## Step 4 — Compute ASR quality effect

Using results from Step 3, build Table 2:

```python
gap = text_ceiling_bleu - cascade_bleu
print(f"Yoruba gap attributable to ASR errors: {gap:.1f} BLEU points")
```

---

## Step 5 — Fill in paper_outline.md

```bash
python papers/fill_results.py paper3_audio
```

Output: `papers/paper3_audio/paper_outline.filled.md`

---

## Step 6 — Validate

```bash
python -c "
from framework.sota import load_and_validate_sota
load_and_validate_sota('papers/paper3_audio/baselines.csv', 'papers/paper3_audio/schema.json')
print('Schema OK')
"
```

---

## Reporting checklist

- [ ] Table 1: BLEU and WER for all three audio strategies × all supported languages
- [ ] Table 2: ASR intermediate WER, cascade BLEU, text ceiling BLEU, gap — per language
- [ ] The gap (text ceiling − cascade) is correctly attributed to ASR error propagation
- [ ] `paper_outline.filled.md` generated

---

## Scope reminder

- Do NOT run PEFT experiments here → Paper 2
- Do NOT compare cascade vs E2E architecture systematically here → Paper 4
- Do NOT vary languages beyond FLEURS supported set
