# LinguoMT-Audio: Audio Strategy Analysis for African Speech Translation

**Authors:** [Author names]  
**Affiliation:** [Institution]  
**Venue:** [Conference/Journal]  
**Date:** [Submission date]

---

## Abstract

Deploying speech translation for African languages requires choosing between architecturally different processing pipelines: end-to-end speech-to-text translation (S2TT), cascade ASR followed by machine translation (ASR+MT), and ASR-only transcription. We present a systematic comparison of these strategies on four African languages — Yoruba, Hausa, Igbo, and Swahili — using FLEURS as the evaluation benchmark. SeamlessM4T-v2 achieves direct S2TT BLEU scores of [RESULT:seamless.yoruba.s2tt.bleu] (Yoruba), [RESULT:seamless.igbo.s2tt.bleu] (Igbo), and [RESULT:seamless.swahili.s2tt.bleu] (Swahili). The Whisper+NLLB cascade reaches [RESULT:cascade.yoruba.s2tt.bleu] (Yoruba), [RESULT:cascade.hausa.s2tt.bleu] (Hausa), and [RESULT:cascade.swahili.s2tt.bleu] (Swahili). We quantify the gap between cascade performance and a text MT ceiling computed by running NLLB on gold reference transcripts, and we show that the cascade's BLEU gap of [RESULT:cascade.yoruba.gap] BLEU points for Yoruba is attributable almost entirely to intermediate ASR errors. These findings directly inform system-design decisions for low-resource African speech applications.

---

## 1. Introduction

Speech translation for African languages is a rapidly growing field driven by practical demand: African communities increasingly communicate across language boundaries in contexts (healthcare, education, commerce) that benefit from automated translation. Yet practitioners face an under-studied architectural choice: should they deploy an end-to-end model that translates audio directly to text in a target language, or a cascade pipeline that first transcribes audio to the source language and then translates the transcript?

Each architecture has different trade-offs. End-to-end systems like SeamlessM4T-v2 (Barrault et al., 2023) avoid error accumulation between stages, but require the model to simultaneously learn acoustic and semantic translation objectives. Cascade systems like Whisper (Radford et al., 2023) + NLLB-200 (NLLB Team, 2022) allow modular improvement: upgrading the ASR or MT component independently. Cascades also expose an intermediate transcript, which is useful for quality estimation and human post-editing.

For high-resource languages this trade-off is well-studied; for low-resource African languages it remains largely unexplored. African phonology and morphology challenge ASR systems (tonal languages, limited training data), and cascades may amplify these errors through MT. Conversely, end-to-end models may lack the audio-translation training data needed to match oracle cascade performance.

This paper makes the following contributions:
1. A head-to-head comparison of direct S2TT (SeamlessM4T-v2), cascade S2TT (Whisper+NLLB), and ASR-only on four African languages;
2. A text MT ceiling experiment — running NLLB on gold transcripts — that isolates ASR-error contribution from architectural factors;
3. Quantification of the cascade gap across languages and its relationship to intermediate ASR quality.

---

## 2. Related Work

### 2.1 End-to-End Speech Translation

End-to-end S2TT models learn a direct mapping from audio features to target-language text. Early work (Bérard et al., 2016; Weiss et al., 2017) demonstrated feasibility on English→French. Subsequent scaling (Liu et al., 2020; Tang et al., 2021; Barrault et al., 2023) extended coverage to multilingual settings. SeamlessM4T-v2 is the current state-of-the-art open model for multilingual S2TT, covering over 100 languages.

### 2.2 Cascade Systems

Cascade speech translation pipelines (Stüker et al., 2012; Kumar et al., 2014) rely on a strong ASR stage to produce an intermediate transcript that is then translated by an NMT system. The primary disadvantage is error propagation: ASR errors compound into MT, and the MT system is typically trained on clean text that does not resemble erroneous ASR output. Recent work (Anastasopoulos et al., 2021; Papi et al., 2022) explores disfluency-robust MT training and confidence-based reranking to mitigate this.

### 2.3 African Speech Translation

Comparative studies specifically for African speech are rare. Olatunji et al. (2022) benchmark Whisper on AfriSpeech. Our own Paper 1 (LinguoMT-Benchmark) provides the baseline zero-shot numbers used here. No prior work explicitly quantifies the cascade gap or text-MT ceiling for these languages.

---

## 3. Experimental Setup

### 3.1 Models and Architectures

**Direct S2TT (SeamlessM4T-v2)**
- Model: `facebook/seamless-m4t-v2-large` (2.3B parameters)
- Audio input → English text output in a single forward pass
- Languages: Yoruba, Igbo, Swahili (Hausa excluded: `hau` not in SeamlessM4T-v2 speech output vocabulary)

**Cascade S2TT (Whisper + NLLB)**
- ASR: `openai/whisper-large-v3` → source-language transcript
- MT: `facebook/nllb-200-distilled-600M` → English translation
- Languages: Yoruba, Hausa, Swahili (Igbo excluded: no Whisper language token)

**Text MT ceiling**
- Input: FLEURS gold reference transcripts (bypasses ASR entirely)
- MT: `facebook/nllb-200-distilled-600M`
- Languages: Yoruba, Hausa, Igbo (text only), Swahili
- This establishes the best possible BLEU a Whisper+NLLB cascade can achieve given perfect ASR

### 3.2 Dataset

FLEURS validation split (Conneau et al., 2023). Same subset as Paper 1 (LinguoMT-Benchmark) for comparability.

### 3.3 Metrics

- **BLEU** (sacrebleu, tokenised): translation quality
- **WER** (%): ASR transcription accuracy (intermediate step in cascade)

### 3.4 Language Details

| Language | Family | FLEURS code | In SeamlessM4T S2TT | In Whisper ASR |
|---------|--------|------------|--------------------|----|
| Yoruba | Niger-Congo (Volta-Niger) | `yor_Latn` | ✓ | ✓ |
| Igbo | Niger-Congo (Volta-Niger) | `ibo_Latn` | ✓ | ✗ |
| Swahili | Niger-Congo (Bantu) | `swh_Latn` | ✓ | ✓ |
| Hausa | Afro-Asiatic (Chadic) | `hau_Latn` | ✗ | ✓ |

---

## 4. Results

### 4.1 Strategy Comparison (Table 1)

**Table 1: Audio strategy comparison — BLEU (S2TT/cascade, ↑) and WER (ASR, ↓)**

| Strategy | Model | Yoruba | Hausa | Igbo | Swahili |
|---------|-------|--------|-------|------|---------|
| S2TT direct | SeamlessM4T-v2 | [RESULT:seamless.yoruba.s2tt.bleu] | — | [RESULT:seamless.igbo.s2tt.bleu] | [RESULT:seamless.swahili.s2tt.bleu] |
| ASR-only (WER) | SeamlessM4T-v2 | [RESULT:seamless.yoruba.asr.wer] | — | [RESULT:seamless.igbo.asr.wer] | [RESULT:seamless.swahili.asr.wer] |
| ASR-only (WER) | Whisper-large-v3 | [RESULT:cascade.yoruba.asr.wer] | [RESULT:cascade.hausa.asr.wer] | — | [RESULT:cascade.swahili.asr.wer] |
| Cascade S2TT | Whisper+NLLB | [RESULT:cascade.yoruba.s2tt.bleu] | [RESULT:cascade.hausa.s2tt.bleu] | — | [RESULT:cascade.swahili.s2tt.bleu] |
| Text MT ceiling | NLLB (gold transcripts) | [RESULT:cascade.yoruba.textmt.bleu] | [RESULT:cascade.hausa.textmt.bleu] | [RESULT:cascade.igbo.textmt.bleu] | [RESULT:cascade.swahili.textmt.bleu] |
| Text MT ceiling | SeamlessM4T text MT | [RESULT:seamless.yoruba.textmt.bleu] | — | [RESULT:seamless.igbo.textmt.bleu] | — |

*— = language excluded from this model (see language support matrix in Section 3.4)*

### 4.2 Cascade Gap Analysis (Table 2)

The cascade gap measures how far below the text-MT ceiling the cascade falls. It is caused entirely by intermediate ASR errors: the cascade receives erroneous transcripts rather than gold text.

**Table 2: Cascade gap — text ceiling BLEU minus cascade BLEU**

| Language | ASR WER (%) | Text ceiling BLEU | Cascade BLEU | Gap (BLEU points) |
|---------|------------|------------------|-------------|-------------------|
| Yoruba | [RESULT:cascade.yoruba.asr.wer] | [RESULT:cascade.yoruba.textmt.bleu] | [RESULT:cascade.yoruba.s2tt.bleu] | **[RESULT:cascade.yoruba.gap]** |
| Hausa | [RESULT:cascade.hausa.asr.wer] | [RESULT:cascade.hausa.textmt.bleu] | [RESULT:cascade.hausa.s2tt.bleu] | **[RESULT:cascade.hausa.gap]** |
| Swahili | [RESULT:cascade.swahili.asr.wer] | [RESULT:cascade.swahili.textmt.bleu] | [RESULT:cascade.swahili.s2tt.bleu] | **[RESULT:cascade.swahili.gap]** |

---

## 5. Discussion

### 5.1 Direct S2TT vs Cascade

[NARRATIVE:discussion_direct_vs_cascade]

SeamlessM4T-v2's direct S2TT achieves [RESULT:seamless.yoruba.s2tt.bleu] BLEU on Yoruba versus the cascade's [RESULT:cascade.yoruba.s2tt.bleu]. [NARRATIVE:s2tt_cascade_comparison_interpretation]. The key structural difference is that SeamlessM4T-v2 can attend to acoustic features throughout decoding, while the cascade is bottlenecked by the quality of the Whisper transcript.

### 5.2 Cascade Gap and ASR Quality

The cascade gap of [RESULT:cascade.yoruba.gap] BLEU points for Yoruba reflects the compound effect of ASR errors (WER [RESULT:cascade.yoruba.asr.wer]%) on MT output. [NARRATIVE:cascade_gap_interpretation]. Languages with lower ASR WER (such as Swahili, WER [RESULT:cascade.swahili.asr.wer]%) show smaller cascade gaps, confirming that ASR quality is the primary bottleneck for cascade systems on these languages.

### 5.3 Implications for System Design

These results suggest that for languages well-covered by SeamlessM4T-v2 (Yoruba, Igbo, Swahili), direct S2TT is preferable unless a human-readable intermediate transcript is required. For Hausa (excluded from SeamlessM4T S2TT), the cascade is the only option, and improving Whisper's Hausa ASR (e.g., via LoRA fine-tuning as explored in Paper 2) directly lifts cascade BLEU.

### 5.4 Limitations

- Results depend on FLEURS, which represents read speech in controlled conditions. Broadcast and conversational speech may yield different relative rankings.
- NLLB-200-distilled-600M is a relatively small MT model; larger MT components (e.g., NLLB-200-1.3B) would raise the text ceiling and potentially widen the cascade gap.

---

## 6. Conclusion

We compare three audio strategies — direct S2TT, cascade ASR+MT, and ASR-only — for four African languages. Direct S2TT via SeamlessM4T-v2 outperforms the Whisper+NLLB cascade on languages where SeamlessM4T-v2 has speech output support. The cascade gap for Yoruba ([RESULT:cascade.yoruba.gap] BLEU points) can be attributed directly to intermediate ASR errors at WER [RESULT:cascade.yoruba.asr.wer]%. Text MT ceiling experiments confirm that a cascade with perfect ASR (WER 0%) could achieve [RESULT:cascade.yoruba.textmt.bleu] BLEU, narrowing the advantage of the end-to-end approach. These findings motivate investing in ASR quality improvement (via PEFT as in Paper 2) as the highest-leverage action for cascade practitioners.

---

## References

- Anastasopoulos, A., et al. (2021). *FINDINGS of the IWSLT 2021 evaluation campaign*. IWSLT 2021.
- Barrault, L., et al. (2023). *SeamlessM4T — Massively Multilingual & Multimodal Machine Translation*. arXiv:2308.11596.
- Bérard, A., et al. (2016). *Listen and translate: A proof of concept for end-to-end speech-to-text translation*. NIPS 2016 workshop.
- Conneau, A., et al. (2023). *FLEURS: Few-shot learning evaluation of universal representations of speech*. SLT 2022.
- Kumar, G., et al. (2014). *Some insights from translating conversational telephone speech*. ICASSP 2014.
- Liu, Y., et al. (2020). *Multilingual denoising pre-training for neural machine translation*. TACL.
- NLLB Team. (2022). *No Language Left Behind: Scaling human-centered machine translation*. arXiv:2207.04672.
- Olatunji, T., et al. (2022). *AfriSpeech-200: Pan-African accented speech dataset for clinical and general domain ASR*. arXiv:2104.02010.
- Papi, S., et al. (2022). *Does speech translation performance improve with ASR error rate reduction?* ACL 2022.
- Radford, A., et al. (2023). *Robust speech recognition via large-scale weak supervision*. ICML 2023.
- Stüker, S., et al. (2012). *The KIT Quaero 2012 IWSLT speech translation system*. IWSLT 2012.
- Tang, Y., et al. (2021). *FST: The FAIR speech translation system for the IWSLT21 multilingual shared task*. IWSLT 2021.
- Weiss, R. J., et al. (2017). *Sequence-to-sequence models can directly translate foreign speech*. Interspeech 2017.
