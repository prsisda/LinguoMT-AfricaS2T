# Experiment Results

This directory receives output files from experiment runs (local or Colab).
One subdirectory per paper. Results here are inputs to `papers/fill_results.py`,
which fills placeholders in `paper_draft.md` and produces `paper_draft_filled.md`.

## Workflow

```
Google Colab run
    ↓ download output zip / folder
results/paper1_benchmark/from_colab/<experiment>/  ← drop here
    ↓ python papers/extract_results.py paper1_benchmark
results/paper1_benchmark/results.csv               ← flat key→value file
    ↓ python papers/fill_results.py paper1_benchmark
papers/paper1_benchmark/paper_draft_filled.md      ← complete paper with numbers
papers/paper1_benchmark/paper_draft.md             ← original with placeholders (unchanged)
```

## Papers

| Directory | Paper | Status |
|-----------|-------|--------|
| `paper1_benchmark/` | LinguoMT Benchmark | — |
| `paper2_adaptation/` | LinguoMT-Adapt | — |
| `paper3_audio/` | LinguoMT-Audio | — |
| `paper4_cascade/` | LinguoMT-Cascade | — |
| `paper5_transfer/` | LinguoMT-Transfer | — |
