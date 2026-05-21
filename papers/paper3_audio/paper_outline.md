# LinguoMT-Audio: Audio Strategy Analysis for African Speech Translation

> Placeholders: `[RESULT:key]` — filled by `python papers/fill_results.py paper3_audio`
> Narrative: `[NARRATIVE:key]` — fill manually.
> Clean-audio baselines reference Paper 1 results directly.

---

## Abstract

We compare three audio processing strategies for African speech translation:
direct end-to-end S2TT (SeamlessM4T-v2), a cascade of ASR + MT (Whisper +
NLLB-200), and ASR-only (transcription without translation). Evaluated on
FLEURS across Yoruba, Hausa, Igbo, and Swahili, we find that [NARRATIVE:abstract_finding].
The text-MT ceiling — NLLB-200 applied to gold transcripts — reveals that
intermediate ASR quality accounts for [NARRATIVE:asr_contribution]% of the
gap between cascade and oracle performance.

---

## 1. Introduction

[NARRATIVE:intro]

---

## 2. Audio Strategies

Three strategies are evaluated:

1. **Direct S2TT** (SeamlessM4T-v2 only): audio → English in a single forward pass.
2. **Cascade ASR+MT** (Whisper+NLLB-200): audio → source-language transcript (Whisper) → English (NLLB-200).
3. **ASR-only** (Whisper or SeamlessM4T-v2 ASR head): audio → source-language transcript (no translation).
4. **Text-MT ceiling**: gold FLEURS transcripts → NLLB-200 → English (establishes upper bound for any cascade).

Language coverage per strategy:

| Strategy | Yoruba | Hausa | Igbo | Swahili |
|----------|--------|-------|------|---------|
| S2TT direct (SeamlessM4T) | ✓ | — | ✓ | ✓ |
| ASR-only (Whisper) | ✓ | ✓ | — | ✓ |
| ASR-only (SeamlessM4T) | ✓ | — | ✓ | ✓ |
| Cascade ASR+MT | ✓ | ✓ | — | ✓ |
| Text-MT ceiling | ✓ | ✓ | ✓ | ✓ |

---

## 3. Results

### 3.1 Strategy Comparison (Table 1)

**Table 1: BLEU (translation) and WER (transcription) by audio strategy**

| Strategy | Yoruba | Hausa | Igbo | Swahili |
|----------|--------|-------|------|---------|
| **S2TT direct (SeamlessM4T)** BLEU | [RESULT:seamless.yoruba.s2tt.bleu] | — | [RESULT:seamless.igbo.s2tt.bleu] | [RESULT:seamless.swahili.s2tt.bleu] |
| **Cascade ASR+MT** BLEU | [RESULT:cascade.yoruba.s2tt.bleu] | [RESULT:cascade.hausa.s2tt.bleu] | — | [RESULT:cascade.swahili.s2tt.bleu] |
| **Text-MT ceiling** BLEU | [RESULT:cascade.yoruba.textmt.bleu] | [RESULT:cascade.hausa.textmt.bleu] | [RESULT:cascade.igbo.textmt.bleu] | [RESULT:cascade.swahili.textmt.bleu] |
| **ASR-only** (Whisper) WER | [RESULT:cascade.yoruba.asr.wer] | [RESULT:cascade.hausa.asr.wer] | — | [RESULT:cascade.swahili.asr.wer] |
| **ASR-only** (SeamlessM4T) WER | [RESULT:seamless.yoruba.asr.wer] | — | [RESULT:seamless.igbo.asr.wer] | [RESULT:seamless.swahili.asr.wer] |

[NARRATIVE:table1_discussion]

### 3.2 ASR Quality and Cascade Performance (Table 2)

**Table 2: Intermediate ASR quality and its effect on cascade BLEU**

| Language | ASR WER (%) | Cascade BLEU | Text ceiling BLEU | Gap (ceiling − cascade) |
|----------|------------|-------------|------------------|------------------------|
| Yoruba | [RESULT:cascade.yoruba.asr.wer] | [RESULT:cascade.yoruba.s2tt.bleu] | [RESULT:cascade.yoruba.textmt.bleu] | [RESULT:cascade.yoruba.gap] |
| Hausa | [RESULT:cascade.hausa.asr.wer] | [RESULT:cascade.hausa.s2tt.bleu] | [RESULT:cascade.hausa.textmt.bleu] | [RESULT:cascade.hausa.gap] |
| Swahili | [RESULT:cascade.swahili.asr.wer] | [RESULT:cascade.swahili.s2tt.bleu] | [RESULT:cascade.swahili.textmt.bleu] | [RESULT:cascade.swahili.gap] |

[NARRATIVE:asr_quality_discussion]

---

## 4. Discussion

[NARRATIVE:discussion]

### 4.1 When Does Each Strategy Win?

[NARRATIVE:strategy_recommendations]

### 4.2 Limitations

[NARRATIVE:limitations]

---

## 5. Conclusion

[NARRATIVE:conclusion]

---

## References

- Paper 1 (this series): LinguoMT Benchmark
- Barrault et al. (2023). Seamless. arXiv:2312.05187
- Radford et al. (2023). Whisper. arXiv:2212.04356
- Costa-jussà et al. (2022). NLLB. arXiv:2207.04672
