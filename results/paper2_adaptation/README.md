# Results — Paper 2: LinguoMT-Adapt

Paper: *LinguoMT-Adapt: Parameter-Efficient Adaptation for African Speech Translation*

## Workflow

```
Google Colab (run experiments)
  └─ results/paper2_adaptation/from_colab/<experiment>/outputs/<adaptation_full_*>/metrics/

python papers/extract_results.py paper2_adaptation   # → results.csv (fills values)
python papers/fill_results.py paper2_adaptation      # → papers/paper2_adaptation/paper_draft_filled.md
```

## Directory layout

```
results/paper2_adaptation/
├── results.csv          ← key→value template; fill values here
├── from_colab/
│   ├── FLEURS__SeamlessM4Tv2/   ← LoRA and Adapter runs
│   └── FLEURS__WhisperNLLB/     ← Whisper LoRA runs
└── README.md            ← this file
```

## Result keys

| Key | Experiment | Language | Metric | Notes |
|-----|-----------|---------|--------|-------|
| `seamless_lora.yoruba.bleu.before` | paper1_import | Yoruba | BLEU | Zero-shot baseline from Paper 1 |
| `seamless_lora.igbo.bleu.before` | paper1_import | Igbo | BLEU | Zero-shot baseline from Paper 1 |
| `seamless_lora.swahili.bleu.before` | paper1_import | Swahili | BLEU | Zero-shot baseline from Paper 1 |
| `seamless_lora.yoruba.wer.before` | paper1_import | Yoruba | WER | Zero-shot baseline from Paper 1 |
| `seamless_lora.igbo.wer.before` | paper1_import | Igbo | WER | Zero-shot baseline from Paper 1 |
| `whisper_lora.yoruba.wer.before` | paper1_import | Yoruba | WER | Whisper zero-shot from Paper 1 |
| `whisper_lora.hausa.wer.before` | paper1_import | Hausa | WER | Whisper zero-shot from Paper 1 |
| `seamless_lora.yoruba.bleu.after` | FLEURS__SeamlessM4Tv2 | Yoruba | BLEU | After LoRA (1000 samples) |
| `seamless_lora.igbo.bleu.after` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | After LoRA |
| `seamless_lora.swahili.bleu.after` | FLEURS__SeamlessM4Tv2 | Swahili | BLEU | After LoRA |
| `seamless_lora.yoruba.wer.after` | FLEURS__SeamlessM4Tv2 | Yoruba | WER | After LoRA |
| `seamless_lora.yoruba.bleu.gain` | computed | Yoruba | BLEU_gain | after.bleu − before.bleu |
| `seamless_lora.yoruba.wer.gain` | computed | Yoruba | WER_gain | before.wer − after.wer |
| `seamless_adapter.yoruba.bleu.after` | FLEURS__SeamlessM4Tv2 | Yoruba | BLEU | After Adapter FT |
| `seamless_adapter.igbo.bleu.after` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | After Adapter FT |
| `whisper_lora.yoruba.wer.after` | FLEURS__WhisperNLLB | Yoruba | WER | After LoRA |
| `whisper_lora.hausa.wer.after` | FLEURS__WhisperNLLB | Hausa | WER | After LoRA |
| `lora.trainable_params_m` | computed | all | params | LoRA trainable params (M) |
| `lora.trainable_pct` | computed | all | pct | LoRA params as % of total |
| `lora.gpu_hours` | measured | all | hours | Wall-clock GPU hours |
| `adapter.trainable_params_m` | computed | all | params | Adapter trainable params (M) |
| `adapter.trainable_pct` | computed | all | pct | Adapter params as % of total |
| `adapter.gpu_hours` | measured | all | hours | Wall-clock GPU hours |
| `scaling.yoruba.bleu.100` | FLEURS__SeamlessM4Tv2 | Yoruba | BLEU | LoRA with 100 samples |
| `scaling.yoruba.bleu.500` | FLEURS__SeamlessM4Tv2 | Yoruba | BLEU | LoRA with 500 samples |
| `scaling.yoruba.bleu.1000` | FLEURS__SeamlessM4Tv2 | Yoruba | BLEU | LoRA with 1000 samples |
| `scaling.yoruba.bleu.full` | FLEURS__SeamlessM4Tv2 | Yoruba | BLEU | LoRA with full train data |
| `scaling.igbo.bleu.100` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | LoRA with 100 samples |
| `scaling.igbo.bleu.500` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | LoRA with 500 samples |
| `scaling.igbo.bleu.1000` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | LoRA with 1000 samples |
| `scaling.igbo.bleu.full` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | LoRA with full train data |
| `scaling.yoruba.wer.100` | FLEURS__WhisperNLLB | Yoruba | WER | LoRA with 100 samples |
| `scaling.yoruba.wer.500` | FLEURS__WhisperNLLB | Yoruba | WER | LoRA with 500 samples |
| `scaling.yoruba.wer.1000` | FLEURS__WhisperNLLB | Yoruba | WER | LoRA with 1000 samples |
| `scaling.yoruba.wer.full` | FLEURS__WhisperNLLB | Yoruba | WER | LoRA with full train data |
| `scaling.hausa.wer.100` | FLEURS__WhisperNLLB | Hausa | WER | LoRA with 100 samples |
| `scaling.hausa.wer.500` | FLEURS__WhisperNLLB | Hausa | WER | LoRA with 500 samples |
| `scaling.hausa.wer.1000` | FLEURS__WhisperNLLB | Hausa | WER | LoRA with 1000 samples |
| `scaling.hausa.wer.full` | FLEURS__WhisperNLLB | Hausa | WER | LoRA with full train data |
| `scaling.min_threshold_samples` | computed | all | samples | Min samples for p<0.05 improvement |

## Manual overrides

Edit `results.csv` directly for:
- Values imported from Paper 1 (`.before` keys) — copy from `results/paper1_benchmark/results.csv`
- Computed deltas (`.gain` keys) — calculate after filling before/after
- GPU hours — note from Colab runtime panel

## Colab output folder naming

The extract script looks for folders matching `*adaptation*full*` inside:
```
results/paper2_adaptation/from_colab/<experiment>/outputs/
```
