# Results — Paper 4: LinguoMT-Cascade

Paper: *LinguoMT-Cascade: When Does Cascade Beat End-to-End for African Speech Translation?*

## Workflow

```
Google Colab (run experiments)
  └─ results/paper4_cascade/from_colab/<experiment>/outputs/<cascade_full_*>/metrics/

python papers/extract_results.py paper4_cascade   # → results.csv (fills values)
python papers/fill_results.py paper4_cascade      # → papers/paper4_cascade/paper_draft_filled.md
```

## Directory layout

```
results/paper4_cascade/
├── results.csv          ← key→value template; fill values here
├── from_colab/
│   ├── FLEURS__SeamlessM4Tv2/   ← E2E baseline (SeamlessM4T)
│   └── FLEURS__WhisperNLLB/     ← Cascade baseline (Whisper+NLLB)
└── README.md            ← this file
```

## Result keys

| Key | Experiment | Language | Metric | Notes |
|-----|-----------|---------|--------|-------|
| `e2e.yoruba.bleu` | paper1_import | Yoruba | BLEU | SeamlessM4T E2E BLEU from Paper 1 |
| `e2e.igbo.bleu` | paper1_import | Igbo | BLEU | SeamlessM4T E2E BLEU from Paper 1 |
| `e2e.swahili.bleu` | paper1_import | Swahili | BLEU | SeamlessM4T E2E BLEU from Paper 1 |
| `cascade.yoruba.bleu` | paper1_import | Yoruba | BLEU | Whisper+NLLB cascade BLEU from Paper 1 |
| `cascade.hausa.bleu` | paper1_import | Hausa | BLEU | Whisper+NLLB cascade BLEU from Paper 1 |
| `cascade.swahili.bleu` | paper1_import | Swahili | BLEU | Whisper+NLLB cascade BLEU from Paper 1 |
| `cascade.yoruba.wer` | paper1_import | Yoruba | WER | Whisper intermediate WER from Paper 1 |
| `cascade.hausa.wer` | paper1_import | Hausa | WER | Whisper intermediate WER from Paper 1 |
| `cascade.swahili.wer` | paper1_import | Swahili | WER | Whisper intermediate WER from Paper 1 |
| `oracle.yoruba.bleu` | FLEURS__WhisperNLLB | Yoruba | BLEU | NLLB on gold FLEURS transcripts |
| `oracle.hausa.bleu` | FLEURS__WhisperNLLB | Hausa | BLEU | NLLB on gold FLEURS transcripts |
| `oracle.igbo.bleu` | FLEURS__WhisperNLLB | Igbo | BLEU | NLLB on gold FLEURS transcripts (text only) |
| `oracle.swahili.bleu` | FLEURS__WhisperNLLB | Swahili | BLEU | NLLB on gold FLEURS transcripts |
| `error.yoruba.asr` | computed | Yoruba | BLEU | oracle.bleu − cascade.bleu (ASR error contribution) |
| `error.hausa.asr` | computed | Hausa | BLEU | oracle.bleu − cascade.bleu |
| `error.swahili.asr` | computed | Swahili | BLEU | oracle.bleu − cascade.bleu |
| `error.yoruba.arch` | computed | Yoruba | BLEU | e2e.bleu − cascade.bleu (architecture gap) |
| `error.swahili.arch` | computed | Swahili | BLEU | e2e.bleu − cascade.bleu |
| `breakeven.yoruba.wer` | computed | Yoruba | WER | WER at which cascade BLEU = E2E BLEU |
| `breakeven.hausa.wer` | computed | Hausa | WER | WER at which cascade BLEU = E2E BLEU |
| `breakeven.swahili.wer` | computed | Swahili | WER | WER at which cascade BLEU = E2E BLEU |
| `latency.e2e.median_ms` | measured | all | ms | SeamlessM4T median latency per sample |
| `latency.e2e.p95_ms` | measured | all | ms | SeamlessM4T P95 latency per sample |
| `latency.e2e.vram_mb` | measured | all | MB | SeamlessM4T peak VRAM |
| `latency.cascade.median_ms` | measured | all | ms | Whisper+NLLB cascade median latency |
| `latency.cascade.p95_ms` | measured | all | ms | Whisper+NLLB cascade P95 latency |
| `latency.cascade.vram_mb` | measured | all | MB | Whisper+NLLB cascade peak VRAM |

## Manual overrides

- Keys with `paper1_import` in the experiment column: copy from `results/paper1_benchmark/results.csv`
- `error.*` and `breakeven.*` keys: compute after filling E2E, cascade, and oracle values
- `latency.*` keys: record from Colab timing cells

## Colab output folder naming

The extract script looks for folders matching `*cascade*full*` inside:
```
results/paper4_cascade/from_colab/<experiment>/outputs/
```

The oracle cascade run (NLLB on gold transcripts) may output as a standalone CSV; place it in
`from_colab/FLEURS__WhisperNLLB/` so the extract script picks it up under the `fleurs_whisper` prefix.
