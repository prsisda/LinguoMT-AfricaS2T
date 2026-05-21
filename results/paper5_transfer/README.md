# Results — Paper 5: LinguoMT-Transfer

Paper: *LinguoMT-Transfer: Linguistic Family Drives Cross-Lingual Transfer for African Speech*

## Workflow

```
Google Colab (run experiments)
  └─ results/paper5_transfer/from_colab/<experiment>/outputs/<transfer_full_*>/metrics/

python papers/extract_results.py paper5_transfer   # → results.csv (fills values)
python papers/fill_results.py paper5_transfer      # → papers/paper5_transfer/paper_draft_filled.md
```

## Directory layout

```
results/paper5_transfer/
├── results.csv          ← key→value template; fill values here
├── from_colab/
│   ├── FLEURS__SeamlessM4Tv2/   ← cross-lingual S2TT transfer experiments
│   └── FLEURS__WhisperNLLB/     ← cross-lingual ASR transfer experiments
└── README.md            ← this file
```

## Result keys

| Key | Experiment | Language | Metric | Notes |
|-----|-----------|---------|--------|-------|
| `topo.yoruba_igbo` | computed | Yoruba+Igbo | cosine | URIEL syntax_knn similarity |
| `topo.yoruba_hausa` | computed | Yoruba+Hausa | cosine | URIEL syntax_knn similarity |
| `topo.igbo_hausa` | computed | Igbo+Hausa | cosine | URIEL syntax_knn similarity |
| `topo.yoruba_english` | computed | Yoruba+English | cosine | URIEL reference pair |
| `zero.yoruba.bleu` | paper1_import | Yoruba | BLEU | SeamlessM4T zero-shot from Paper 1 |
| `zero.igbo.bleu` | paper1_import | Igbo | BLEU | SeamlessM4T zero-shot from Paper 1 |
| `zero.swahili.bleu` | paper1_import | Swahili | BLEU | SeamlessM4T zero-shot from Paper 1 |
| `zero.yoruba.wer` | paper1_import | Yoruba | WER | Whisper zero-shot from Paper 1 |
| `zero.hausa.wer` | paper1_import | Hausa | WER | Whisper zero-shot from Paper 1 |
| `xfer.yor_to_ibo.bleu` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | FT Yoruba → eval Igbo (same family) |
| `xfer.ibo_to_yor.bleu` | FLEURS__SeamlessM4Tv2 | Yoruba | BLEU | FT Igbo → eval Yoruba (same family) |
| `xfer.yor_to_hau.wer` | FLEURS__WhisperNLLB | Hausa | WER | FT Yoruba → eval Hausa (cross-family) |
| `xfer.hau_to_yor.wer` | FLEURS__WhisperNLLB | Yoruba | WER | FT Hausa → eval Yoruba (cross-family) |
| `xfer.eng_to_yor.bleu` | FLEURS__SeamlessM4Tv2 | Yoruba | BLEU | FT English → eval Yoruba (reference) |
| `xfer.eng_to_hau.wer` | FLEURS__WhisperNLLB | Hausa | WER | FT English → eval Hausa (reference) |
| `mono.yoruba.bleu` | FLEURS__SeamlessM4Tv2 | Yoruba | BLEU | Monolingual FT upper bound |
| `mono.igbo.bleu` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | Monolingual FT upper bound |
| `mono.swahili.bleu` | FLEURS__SeamlessM4Tv2 | Swahili | BLEU | Monolingual FT upper bound |
| `mono.yoruba.wer` | FLEURS__WhisperNLLB | Yoruba | WER | Monolingual FT upper bound |
| `mono.hausa.wer` | FLEURS__WhisperNLLB | Hausa | WER | Monolingual FT upper bound |
| `few.yoruba.wer.25` | FLEURS__WhisperNLLB | Yoruba | WER | Few-shot at 25 samples |
| `few.yoruba.wer.50` | FLEURS__WhisperNLLB | Yoruba | WER | Few-shot at 50 samples |
| `few.yoruba.wer.100` | FLEURS__WhisperNLLB | Yoruba | WER | Few-shot at 100 samples |
| `few.yoruba.wer.200` | FLEURS__WhisperNLLB | Yoruba | WER | Few-shot at 200 samples |
| `few.igbo.bleu.25` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | Few-shot at 25 samples |
| `few.igbo.bleu.50` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | Few-shot at 50 samples |
| `few.igbo.bleu.100` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | Few-shot at 100 samples |
| `few.igbo.bleu.200` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | Few-shot at 200 samples |
| `few.hausa.wer.25` | FLEURS__WhisperNLLB | Hausa | WER | Few-shot at 25 samples |
| `few.hausa.wer.50` | FLEURS__WhisperNLLB | Hausa | WER | Few-shot at 50 samples |
| `few.hausa.wer.100` | FLEURS__WhisperNLLB | Hausa | WER | Few-shot at 100 samples |
| `few.hausa.wer.200` | FLEURS__WhisperNLLB | Hausa | WER | Few-shot at 200 samples |
| `interaction.coeff` | computed | all | regression_coeff | Interaction: log(samples) × lang_family |
| `interaction.pvalue` | computed | all | p_value | p-value for interaction term |

## Manual overrides

- `topo.*` keys: run `lang2vec` locally (see `papers/paper5_transfer/config.yaml`)
- `zero.*` and `paper1_import` keys: copy from `results/paper1_benchmark/results.csv`
- `interaction.*` keys: compute from regression after filling all `few.*` values

## Transfer experiment design

| Pair | Train language | Eval language | Expected | Hypothesis |
|------|---------------|--------------|---------|-----------|
| `yor_to_ibo` | Yoruba | Igbo | Good transfer | Same Niger-Congo family |
| `ibo_to_yor` | Igbo | Yoruba | Good transfer | Same Niger-Congo family |
| `yor_to_hau` | Yoruba | Hausa | Poor transfer | Cross-family (Niger-Congo → Afro-Asiatic) |
| `hau_to_yor` | Hausa | Yoruba | Poor transfer | Cross-family |
| `eng_to_yor` | English | Yoruba | Reference | Baseline for cross-lingual transfer |

## Colab output folder naming

The extract script looks for folders matching `*transfer*full*` inside:
```
results/paper5_transfer/from_colab/<experiment>/outputs/
```
