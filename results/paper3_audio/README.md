# Results — Paper 3: LinguoMT-Audio

Paper: *LinguoMT-Audio: Audio Strategy Analysis for African Speech Translation*

## Workflow

```
Google Colab (run experiments)
  └─ results/paper3_audio/from_colab/<experiment>/outputs/<audio_full_*>/metrics/

python papers/extract_results.py paper3_audio   # → results.csv (fills values)
python papers/fill_results.py paper3_audio      # → papers/paper3_audio/paper_draft_filled.md
```

## Directory layout

```
results/paper3_audio/
├── results.csv          ← key→value template; fill values here
├── from_colab/
│   ├── FLEURS__SeamlessM4Tv2/   ← S2TT direct + ASR + text MT ceiling
│   └── FLEURS__WhisperNLLB/     ← ASR + cascade S2TT + NLLB text MT ceiling
└── README.md            ← this file
```

## Result keys

| Key | Experiment | Language | Metric | Notes |
|-----|-----------|---------|--------|-------|
| `seamless.yoruba.s2tt.bleu` | FLEURS__SeamlessM4Tv2 | Yoruba | BLEU | Direct S2TT (audio→English) |
| `seamless.igbo.s2tt.bleu` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | Direct S2TT |
| `seamless.swahili.s2tt.bleu` | FLEURS__SeamlessM4Tv2 | Swahili | BLEU | Direct S2TT |
| `seamless.yoruba.asr.wer` | FLEURS__SeamlessM4Tv2 | Yoruba | WER | ASR transcription WER |
| `seamless.igbo.asr.wer` | FLEURS__SeamlessM4Tv2 | Igbo | WER | ASR transcription WER |
| `seamless.swahili.asr.wer` | FLEURS__SeamlessM4Tv2 | Swahili | WER | ASR transcription WER |
| `seamless.yoruba.textmt.bleu` | FLEURS__SeamlessM4Tv2 | Yoruba | BLEU | Text MT on gold transcripts (ceiling) |
| `seamless.igbo.textmt.bleu` | FLEURS__SeamlessM4Tv2 | Igbo | BLEU | Text MT on gold transcripts (ceiling) |
| `seamless.swahili.textmt.bleu` | FLEURS__SeamlessM4Tv2 | Swahili | BLEU | Text MT on gold transcripts (ceiling) |
| `cascade.yoruba.asr.wer` | FLEURS__WhisperNLLB | Yoruba | WER | Whisper ASR intermediate step |
| `cascade.hausa.asr.wer` | FLEURS__WhisperNLLB | Hausa | WER | Whisper ASR intermediate step |
| `cascade.swahili.asr.wer` | FLEURS__WhisperNLLB | Swahili | WER | Whisper ASR intermediate step |
| `cascade.yoruba.s2tt.bleu` | FLEURS__WhisperNLLB | Yoruba | BLEU | Cascade S2TT (Whisper+NLLB) |
| `cascade.hausa.s2tt.bleu` | FLEURS__WhisperNLLB | Hausa | BLEU | Cascade S2TT |
| `cascade.swahili.s2tt.bleu` | FLEURS__WhisperNLLB | Swahili | BLEU | Cascade S2TT |
| `cascade.yoruba.textmt.bleu` | FLEURS__WhisperNLLB | Yoruba | BLEU | NLLB on gold transcripts (cascade ceiling) |
| `cascade.hausa.textmt.bleu` | FLEURS__WhisperNLLB | Hausa | BLEU | NLLB on gold transcripts |
| `cascade.swahili.textmt.bleu` | FLEURS__WhisperNLLB | Swahili | BLEU | NLLB on gold transcripts |
| `cascade.igbo.textmt.bleu` | FLEURS__WhisperNLLB | Igbo | BLEU | NLLB on gold transcripts (text only) |
| `cascade.yoruba.gap` | computed | Yoruba | BLEU | textmt.bleu − cascade.s2tt.bleu |
| `cascade.hausa.gap` | computed | Hausa | BLEU | textmt.bleu − cascade.s2tt.bleu |
| `cascade.swahili.gap` | computed | Swahili | BLEU | textmt.bleu − cascade.s2tt.bleu |

## Audio strategies overview

| Strategy | Model | Input | Output |
|---------|-------|-------|--------|
| S2TT direct | SeamlessM4T-v2 | audio | English text |
| ASR-only | Whisper-large-v3 or SeamlessM4T | audio | source-language text |
| Cascade ASR+MT | Whisper + NLLB-200-600M | audio | English text |
| Text MT ceiling | NLLB-200-600M | gold transcripts | English text |

## Colab output folder naming

The extract script looks for folders matching `*audio*full*` inside:
```
results/paper3_audio/from_colab/<experiment>/outputs/
```
