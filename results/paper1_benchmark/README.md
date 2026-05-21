# Paper 1 — Experiment Results

## Step 1 — Copy Colab output folders here

After each Colab run finishes, download the experiment output folder
(or unzip the archive) and copy it into the matching subdirectory below.

```
results/paper1_benchmark/from_colab/
├── FLEURS__SeamlessM4Tv2/          ← paste output folder contents here
│   └── outputs/
│       └── 2025-XX-XX_benchmark_full/
│           └── metrics/
│               ├── text_metrics.csv
│               ├── asr_metrics.csv
│               └── aggregate_metrics.csv
├── FLEURS__WhisperNLLB/
│   └── outputs/...
├── AfricanCeltic__SeamlessM4Tv2/
│   └── outputs/...
└── AfricanCeltic__WhisperNLLB/
    └── outputs/...
```

You need all four folders for a complete paper. You can run `extract_results.py`
after each individual run to check progress — missing experiments leave their
keys unfilled and are listed in the report.

## Step 2 — Extract results into results.csv

```bash
python papers/extract_results.py paper1_benchmark
```

This scans `from_colab/` for the latest `benchmark_full` output folder per experiment,
reads `text_metrics.csv` and `asr_metrics.csv`, maps rows to result keys, and writes
`results/paper1_benchmark/results.csv`.

## Step 3 — Fill the paper

```bash
python papers/fill_results.py paper1_benchmark
```

Produces `papers/paper1_benchmark/paper_draft_filled.md`.
`papers/paper1_benchmark/paper_draft.md` is never modified.

## Manual override

You can also edit `results.csv` directly to insert or correct individual values
without re-running the extract script. The format is:

```csv
key,value,experiment,language,metric,notes
fleurs_seamless.yoruba.bleu,5.42,FLEURS__SeamlessM4Tv2,Yoruba,BLEU,
fleurs_seamless.yoruba.wer,67.3,FLEURS__SeamlessM4Tv2,Yoruba,WER,
```

## Result keys for Paper 1

All keys that must be populated for a complete paper:

### FLEURS × SeamlessM4T-v2 Large

| Key | Description |
|-----|-------------|
| `fleurs_seamless.yoruba.bleu` | Yoruba S2TT BLEU (Source→English) |
| `fleurs_seamless.igbo.bleu` | Igbo S2TT BLEU |
| `fleurs_seamless.swahili.bleu` | Swahili S2TT BLEU |
| `fleurs_seamless.yoruba.spbleu` | Yoruba spBLEU |
| `fleurs_seamless.igbo.spbleu` | Igbo spBLEU |
| `fleurs_seamless.swahili.spbleu` | Swahili spBLEU |
| `fleurs_seamless.yoruba.chrf` | Yoruba chrF |
| `fleurs_seamless.igbo.chrf` | Igbo chrF |
| `fleurs_seamless.swahili.chrf` | Swahili chrF |
| `fleurs_seamless.yoruba.wer` | Yoruba ASR WER (%) |
| `fleurs_seamless.igbo.wer` | Igbo ASR WER (%) |
| `fleurs_seamless.swahili.wer` | Swahili ASR WER (%) |
| `fleurs_seamless.yoruba.textmt.bleu` | Yoruba text MT BLEU (English→Yoruba) |
| `fleurs_seamless.igbo.textmt.bleu` | Igbo text MT BLEU |
| `fleurs_seamless.swahili.textmt.bleu` | Swahili text MT BLEU |

### FLEURS × Whisper+NLLB-200

| Key | Description |
|-----|-------------|
| `fleurs_whisper.yoruba.wer` | Yoruba ASR WER (Whisper) |
| `fleurs_whisper.hausa.wer` | Hausa ASR WER |
| `fleurs_whisper.swahili.wer` | Swahili ASR WER |
| `fleurs_whisper.yoruba.bleu` | Yoruba cascade BLEU |
| `fleurs_whisper.hausa.bleu` | Hausa cascade BLEU |
| `fleurs_whisper.swahili.bleu` | Swahili cascade BLEU |
| `fleurs_whisper.yoruba.spbleu` | Yoruba cascade spBLEU |
| `fleurs_whisper.hausa.spbleu` | Hausa cascade spBLEU |
| `fleurs_whisper.yoruba.chrf` | Yoruba cascade chrF |
| `fleurs_whisper.hausa.chrf` | Hausa cascade chrF |

### African-Celtic × SeamlessM4T-v2 Large

| Key | Description |
|-----|-------------|
| `ac_seamless.yoruba.bleu` | Yoruba S2TT BLEU |
| `ac_seamless.igbo.bleu` | Igbo S2TT BLEU |
| `ac_seamless.yoruba.wer` | Yoruba ASR WER (%) |
| `ac_seamless.igbo.wer` | Igbo ASR WER (%) |
| `ac_seamless.yoruba.textmt.bleu` | Yoruba text MT BLEU |

### African-Celtic × Whisper+NLLB-200

| Key | Description |
|-----|-------------|
| `ac_whisper.yoruba.wer` | Yoruba ASR WER |
| `ac_whisper.hausa.wer` | Hausa ASR WER |
| `ac_whisper.yoruba.bleu` | Yoruba cascade BLEU |
| `ac_whisper.hausa.bleu` | Hausa cascade BLEU |

### Gap analysis (computed by extract script)

| Key | Description |
|-----|-------------|
| `gap.french.wer` | French reference WER |
| `gap.german.wer` | German reference WER |
| `gap.spanish.wer` | Spanish reference WER |
| `gap.yoruba.bleu_gap` | Absolute BLEU gap vs French |
| `gap.hausa.bleu_gap` | Hausa BLEU gap vs French |
| `metric_tau.bleu_spbleu` | Kendall τ BLEU vs spBLEU |
| `metric_tau.bleu_chrf` | Kendall τ BLEU vs chrF |
