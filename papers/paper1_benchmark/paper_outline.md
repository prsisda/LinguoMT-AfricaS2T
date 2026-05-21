# LinguoMT: Zero-Shot Speech Translation Benchmarking for African Languages

> **How to use this file**
> Placeholders follow the pattern `[RESULT:key]`. After running experiments,
> run `python papers/fill_results.py paper1_benchmark` to replace each
> placeholder with the corresponding value from the experiment output CSVs.
> Fill narrative placeholders (`[NARRATIVE:key]`) manually.

---

## Abstract

We present LinguoMT, a systematic zero-shot evaluation of pretrained speech
translation and recognition models on four African languages — Yoruba, Hausa,
Igbo, and Swahili — across two datasets: FLEURS and African-Celtic. We
evaluate two systems: SeamlessM4T-v2-large, an end-to-end speech translation
model, and a cascade pipeline combining Whisper-large-v3 for ASR with
NLLB-200-distilled-600M for machine translation. [NARRATIVE:abstract_finding]
Our results establish baseline performance figures for under-resourced African
languages and quantify the gap relative to high-resource European languages.

---

## 1. Introduction

[NARRATIVE:introduction_motivation]

The main contributions of this paper are:
- Zero-shot ASR and S2TT evaluation on four African languages across two
  datasets with distinct acoustic properties
- A systematic comparison of end-to-end vs cascade architecture at zero shot
- A quantification of the African–European language performance gap under
  identical evaluation conditions

---

## 2. Models and Datasets

### 2.1 Models

**SeamlessM4T-v2 Large** (`facebook/seamless-m4t-v2-large`) is an end-to-end
multimodal translation model supporting speech and text across 100+ languages.
It performs ASR and S2TT within a single pass.

**Whisper large-v3 + NLLB-200-distilled-600M** is a cascade pipeline: Whisper
(`openai/whisper-large-v3`) transcribes speech to the source language, then
NLLB-200 (`facebook/nllb-200-distilled-600M`) translates the transcript.
Language codes: Yoruba `yo` / `yor_Latn`, Hausa `ha` / `hau_Latn`,
Swahili `sw` / `swh_Latn` (Igbo excluded: no Whisper language token).

### 2.2 Datasets

**FLEURS** (Few-shot Learning Evaluation of Universal Representations of
Speech) is a multilingual speech benchmark covering 102 languages. We use the
`validation` split for evaluation. Language configs: `yo_ng` (Yoruba),
`ha_ng` (Hausa), `ig_ng` (Igbo), `sw_ke` (Swahili).

**African-Celtic** (`McGill-NLP/african_celtic_dataset`, 48 kHz audio) provides
an independent evaluation set with different acoustic conditions. We use the
`dev` split. Hausa audio is absent from this dataset.

### 2.3 Language Support Matrix

| Language | Family | SeamlessM4T (FLEURS) | Whisper (FLEURS) | SeamlessM4T (AC) | Whisper (AC) |
|----------|--------|----------------------|------------------|------------------|--------------|
| Yoruba   | Niger-Congo | S2TT + ASR | ASR | S2TT + ASR | ASR |
| Hausa    | Afro-Asiatic | Text-MT only† | ASR | — | ASR |
| Igbo     | Niger-Congo | S2TT + ASR | — | S2TT + ASR | — |
| Swahili  | Niger-Congo | S2TT + ASR | ASR | — | — |

† Hausa excluded from SeamlessM4T S2TT: `hau` not in speech output vocabulary.

---

## 3. Results

### 3.1 ASR on FLEURS (Table 1)

**Table 1: Word Error Rate (%) on FLEURS validation split — lower is better**

| Model | Yoruba | Hausa | Igbo | Swahili |
|-------|--------|-------|------|---------|
| SeamlessM4T-v2 Large | [RESULT:fleurs_seamless.yoruba.wer] | — | [RESULT:fleurs_seamless.igbo.wer] | [RESULT:fleurs_seamless.swahili.wer] |
| Whisper large-v3 | [RESULT:fleurs_whisper.yoruba.wer] | [RESULT:fleurs_whisper.hausa.wer] | — | [RESULT:fleurs_whisper.swahili.wer] |
| *Whisper-large-v3 (published)* | *[BASELINE:radford2023whisper.yoruba.wer]* | *[BASELINE:radford2023whisper.hausa.wer]* | *—* | *—* |
| *MMS-300M (published)* | *[BASELINE:pratap2023mms.yoruba.wer]* | *[BASELINE:pratap2023mms.hausa.wer]* | *[BASELINE:pratap2023mms.igbo.wer]* | *—* |

— = language not supported by this model.
*Italic rows* = published baselines from prior work, not our runs.

[NARRATIVE:asr_discussion]

### 3.2 Speech-to-Text Translation on FLEURS (Table 2)

**Table 2: BLEU on FLEURS validation split (Source → English)**

| Model | Yoruba | Hausa | Igbo | Swahili |
|-------|--------|-------|------|---------|
| SeamlessM4T-v2 Large (E2E) | [RESULT:fleurs_seamless.yoruba.bleu] | — | [RESULT:fleurs_seamless.igbo.bleu] | [RESULT:fleurs_seamless.swahili.bleu] |
| Whisper+NLLB-200 (Cascade) | [RESULT:fleurs_whisper.yoruba.bleu] | [RESULT:fleurs_whisper.hausa.bleu] | — | [RESULT:fleurs_whisper.swahili.bleu] |
| *SeamlessM4T-v2 (published)* | *[BASELINE:barrault2023seamless.yoruba.bleu]* | *[BASELINE:barrault2023seamless.hausa.bleu]* | *[BASELINE:barrault2023seamless.igbo.bleu]* | *—* |
| *mSLAM-CTC (published)* | *[BASELINE:conneau2022fleurs.yoruba.bleu]* | *—* | *—* | *—* |

[NARRATIVE:s2tt_discussion]

### 3.3 Results on African-Celtic (Table 3)

**Table 3: ASR WER (%) and S2TT BLEU on African-Celtic dev split**

| Model | Task | Yoruba | Hausa | Igbo |
|-------|------|--------|-------|------|
| SeamlessM4T-v2 | WER | [RESULT:ac_seamless.yoruba.wer] | — | [RESULT:ac_seamless.igbo.wer] |
| SeamlessM4T-v2 | BLEU | [RESULT:ac_seamless.yoruba.bleu] | — | [RESULT:ac_seamless.igbo.bleu] |
| Whisper large-v3 | WER | [RESULT:ac_whisper.yoruba.wer] | [RESULT:ac_whisper.hausa.wer] | — |
| Whisper+NLLB (Cascade) | BLEU | [RESULT:ac_whisper.yoruba.bleu] | [RESULT:ac_whisper.hausa.bleu] | — |

[NARRATIVE:african_celtic_discussion]

### 3.4 High-Resource Language Gap (Table 4)

**Table 4: Performance gap — African languages vs European reference languages**

| Language | Family | Whisper WER (%) | BLEU gap vs French |
|----------|--------|-----------------|--------------------|
| Yoruba | Niger-Congo | [RESULT:fleurs_whisper.yoruba.wer] | [RESULT:gap.yoruba.bleu] |
| Hausa | Afro-Asiatic | [RESULT:fleurs_whisper.hausa.wer] | [RESULT:gap.hausa.bleu] |
| Igbo | Niger-Congo | — | [RESULT:gap.igbo.bleu] |
| Swahili | Niger-Congo | [RESULT:fleurs_whisper.swahili.wer] | [RESULT:gap.swahili.bleu] |
| French | Indo-European | [RESULT:gap.french.wer] | 0.0 (reference) |
| German | Indo-European | [RESULT:gap.german.wer] | [RESULT:gap.german.bleu] |
| Spanish | Indo-European | [RESULT:gap.spanish.wer] | [RESULT:gap.spanish.bleu] |

[NARRATIVE:gap_discussion]

### 3.5 Metric Consistency (Table 5)

**Table 5: Kendall τ between metric rankings across all systems**

| Pair | Kendall τ |
|------|-----------|
| BLEU vs spBLEU | [RESULT:metric_tau.bleu_spbleu] |
| BLEU vs chrF | [RESULT:metric_tau.bleu_chrf] |
| spBLEU vs chrF | [RESULT:metric_tau.spbleu_chrf] |

[NARRATIVE:metric_discussion]

---

## 4. Discussion

[NARRATIVE:discussion_main]

### 4.1 End-to-End vs Cascade

[NARRATIVE:e2e_vs_cascade]

### 4.2 Cross-Dataset Consistency

[NARRATIVE:cross_dataset]

### 4.3 Limitations

[NARRATIVE:limitations]

---

## 5. Conclusion

[NARRATIVE:conclusion]

---

## References

- Barrault et al. (2023). *Seamless: Multilingual Expressive and Streaming Speech Translation*. arXiv:2312.05187
- Radford et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML 2023. arXiv:2212.04356
- Costa-jussà et al. (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation*. arXiv:2207.04672
- Conneau et al. (2022). *FLEURS: Few-Shot Learning Evaluation of Universal Representations of Speech*. IEEE SLT 2022. arXiv:2205.12446
- Bapna et al. (2022). *mSLAM: Massively Multilingual Joint Pre-Training for Speech and Text*. arXiv:2202.01374
- Pratap et al. (2023). *Scaling Speech Technology to 1,000+ Languages*. arXiv:2305.13516
- Babu et al. (2022). *XLS-R: Self-Supervised Cross-Lingual Speech Representation Learning at Scale*. Interspeech 2022. arXiv:2111.09296
- Papineni et al. (2002). *BLEU: A Method for Automatic Evaluation of Machine Translation*. ACL 2002.
- Nunamaker et al. (1991). *Systems Development in Information Systems Research*. JMIS.
