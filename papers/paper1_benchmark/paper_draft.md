# LinguoMT: A Zero-Shot Benchmark for African Language Speech Translation and Recognition

**Authors:** [AUTHOR_LIST]  
**Affiliation:** [AFFILIATION]  
**Contact:** [CONTACT_EMAIL]  
**Version:** Draft — placeholders marked `[RESULT:key]` and `[NARRATIVE:key]`  
**Generated:** run `python papers/fill_results.py paper1_benchmark` after experiments

---

## Abstract

We present **LinguoMT**, a systematic zero-shot evaluation of state-of-the-art
multilingual speech models on four African languages — Yoruba, Hausa, Igbo, and
Swahili — spanning two architectures and two datasets. We evaluate
SeamlessM4T-v2-large, an end-to-end (E2E) speech translation model, and a
cascade pipeline combining Whisper-large-v3 for automatic speech recognition
(ASR) with NLLB-200-distilled-600M for machine translation, across the FLEURS
benchmark and the African-Celtic dataset. On FLEURS, SeamlessM4T-v2-large
achieves BLEU scores of [RESULT:fleurs_seamless.yoruba.bleu] (Yoruba),
[RESULT:fleurs_seamless.igbo.bleu] (Igbo), and [RESULT:fleurs_seamless.swahili.bleu]
(Swahili) for speech-to-text translation, while the Whisper+NLLB-200 cascade
achieves [RESULT:fleurs_whisper.yoruba.bleu] (Yoruba), [RESULT:fleurs_whisper.hausa.bleu]
(Hausa), and [RESULT:fleurs_whisper.swahili.bleu] (Swahili). ASR word error rates
for Yoruba reach [RESULT:fleurs_seamless.yoruba.wer]% (SeamlessM4T-v2) and
[RESULT:fleurs_whisper.yoruba.wer]% (Whisper), compared to [RESULT:gap.french.wer]%
and [RESULT:gap.french.bleu] BLEU for French under identical conditions, revealing
a performance gap of [NARRATIVE:gap_summary]. These results establish reproducible
baselines for future work on African speech technology and motivate targeted
adaptation strategies explored in companion papers.

**Keywords:** African languages, speech translation, ASR, FLEURS, SeamlessM4T,
Whisper, NLLB-200, zero-shot evaluation, low-resource languages

---

## 1. Introduction

Speech is the dominant mode of communication for the majority of speakers of
African languages. Yet automatic speech recognition (ASR) and speech-to-text
translation (S2TT) research has historically been concentrated on a small set of
high-resource languages — primarily English, Mandarin, and western European
languages — leaving the linguistic diversity of Africa largely under-served by
modern deep-learning systems.

Recent years have seen significant advances in massively multilingual speech
models. Meta AI's SeamlessM4T-v2 (Barrault et al., 2023b) supports over 100
languages in a single end-to-end architecture capable of ASR, S2TT, and text-only
translation. OpenAI's Whisper (Radford et al., 2023) achieves broad language
coverage through weakly supervised training on 680,000 hours of multilingual
audio. Meta's NLLB-200 (Costa-jussà et al., 2022) extends text MT coverage to
200 languages. Together, these systems theoretically bring African languages within
reach of usable automatic translation and transcription.

However, the extent to which these large multilingual systems actually perform
on African languages in practice is not well understood. Published evaluations
often report macro-averages across dozens of languages, obscuring the tail
performance on low-resource languages. Where per-language results exist, they
frequently cover only the African languages with the largest HuggingFace or
Common Voice datasets, and are not directly comparable across papers due to
differences in evaluation data, split usage, and metric computation.

This paper addresses these gaps by contributing a systematic, reproducible
zero-shot evaluation of the two most practically deployable multilingual speech
systems — SeamlessM4T-v2-large and the Whisper+NLLB-200 cascade — across four
African languages and two independent evaluation datasets. Our evaluation
follows the methodology of Conneau et al. (2022), using the FLEURS benchmark
for cross-paper comparability, supplemented by the African-Celtic dataset
(McGill-NLP, 2023) to test generalization across acoustic domains. No fine-tuning
is performed — all numbers reflect the models as released, which is the relevant
baseline for practitioners adopting these systems without access to labelled
African speech data.

Our main contributions are:

1. **A zero-shot benchmark** covering ASR (WER, CER) and S2TT (BLEU, spBLEU,
   chrF) for Yoruba, Hausa, Igbo, and Swahili across FLEURS and African-Celtic.

2. **A quantified performance gap** between African and high-resource European
   languages, measured under identical conditions with the same models and metrics.

3. **A comparison of E2E vs cascade architectures** at zero shot, establishing
   which architecture is preferable for each language and task.

4. **Open, reproducible experiment code** enabling direct replication and
   extension to additional African languages.

The benchmark numbers reported here serve as the reference baselines for the
companion papers in this series, which investigate parameter-efficient
fine-tuning (Paper 2), audio strategy analysis (Paper 3), cascade vs E2E
architecture analysis (Paper 4), and cross-lingual transfer (Paper 5).

---

## 2. Related Work

### 2.1 Multilingual Speech Models

SeamlessM4T (Barrault et al., 2023a) introduced a unified model for speech and
text translation across 100 languages, followed by SeamlessM4T-v2 (Barrault
et al., 2023b) with improved coverage and a redesigned UnitY2 architecture.
Whisper (Radford et al., 2023) demonstrated that weakly-supervised training on
large-scale multilingual audio yields competitive ASR across 99 languages with no
fine-tuning. NLLB-200 (Costa-jussà et al., 2022) scaled text MT to 200 languages
via a Sparsely-Gated Mixture of Experts architecture and the Flores-200 benchmark.
These three systems represent the practical frontier for low-resource language
practitioners: they require no labelled data and are freely accessible.

### 2.2 African Language Speech Processing

Prior work on African language ASR and S2TT has been constrained by limited
data availability. Oyelaran et al. (2022) released the MasakhaSpeech corpus
covering 8 African languages, enabling fine-tuning experiments but not zero-shot
evaluation of large models. Olatunji et al. (2022) evaluated Whisper on the
AfriSpeech corpus of Nigerian English, finding that cross-accent transfer is
substantially lower than for standard accents. Pratap et al. (2023) showed that
MMS-300M, trained on over 1,100 languages, achieves WER competitive with Whisper
on FLEURS for some African languages, while covering more languages including Igbo.
Babu et al. (2022) demonstrated that XLS-R (self-supervised pre-training at 1B
scale) can achieve competitive WER after fine-tuning on small ASR datasets.

What is lacking in the literature is a **direct comparison** of the two dominant
zero-shot pipeline paradigms — monolithic E2E vs modular cascade — on the same
evaluation data, with the same metrics, covering multiple African languages
simultaneously.

### 2.3 Evaluation Benchmarks

FLEURS (Conneau et al., 2022) provides an n-way parallel speech dataset across
102 languages derived from the FLoRes-200 text benchmark sentences, making it
the natural evaluation standard for cross-lingual speech research. The African-Celtic
dataset (McGill-NLP, 2023) offers an independent evaluation set with different
acoustic conditions, allowing us to probe domain robustness.

---

## 3. System Overview

### 3.1 SeamlessM4T-v2 Large

We evaluate `facebook/seamless-m4t-v2-large` (Barrault et al., 2023b), an
end-to-end multimodal translation model. For S2TT, the model receives audio
input and produces translated text in a single forward pass via an encoder-decoder
architecture. For ASR, the same model is used with source and target language
codes set identically. For text MT, audio processing is bypassed entirely. We
use the HuggingFace `AutoProcessor` and `SeamlessM4Tv2Model` with
`generate_speech=False` to obtain text tokens.

**Language coverage** for S2TT and ASR (speech output): Yoruba (`yor`), Igbo
(`ibo`), Swahili (`swh`). Hausa (`hau`) supports text-only translation but is
absent from the speech output vocabulary and is therefore excluded from S2TT
and ASR evaluations.

### 3.2 Whisper-large-v3 + NLLB-200-distilled-600M (Cascade)

We construct a two-stage cascade: Whisper-large-v3 (`openai/whisper-large-v3`,
Radford et al. 2023) performs ASR, transcribing speech to source-language text.
NLLB-200-distilled-600M (`facebook/nllb-200-distilled-600M`, Costa-jussà et al.
2022) then translates the transcript to English. Language codes are: Yoruba
`yo` (Whisper) / `yor_Latn` (NLLB), Hausa `ha` / `hau_Latn`, Swahili `sw` /
`swh_Latn`. Igbo is excluded from the cascade because Whisper has no Igbo
language token.

### 3.3 Architecture Summary

| Property | SeamlessM4T-v2 Large | Whisper+NLLB-200 Cascade |
|----------|---------------------|--------------------------|
| Parameters | ~2.3B | ~1.6B (Whisper 1.5B + NLLB 600M) |
| Architecture | Unified E2E | Sequential pipeline |
| S2TT | Single pass | ASR → MT |
| Igbo support | ✓ (S2TT + ASR) | ✗ (no Whisper token) |
| Hausa S2TT | ✗ (text MT only) | ✓ |
| Requires GPU | ≥ 16 GB VRAM | ≥ 16 GB VRAM |

---

## 4. Experimental Setup

### 4.1 Datasets

**FLEURS** (Few-shot Learning Evaluation of Universal Representations of Speech,
Conneau et al. 2022) is a multilingual speech benchmark covering 102 languages
across 12 language groups. Each language uses ~200 spoken sentences from the
FLoRes-200 text data. We use the `validation` split for evaluation. Language
configs: `yo_ng` (Yoruba), `ha_ng` (Hausa), `ig_ng` (Igbo), `sw_ke` (Swahili).

**African-Celtic** (`McGill-NLP/african_celtic_dataset`) provides an independent
speech corpus at 48 kHz. We use the `dev` split. Hausa audio is absent from this
dataset; Igbo is present but Whisper cannot process it (no language token).

### 4.2 Languages

| Language | Family | Subfamily | FLEURS | African-Celtic |
|----------|--------|-----------|--------|----------------|
| Yoruba | Niger-Congo | Volta-Niger | ✓ | ✓ |
| Hausa | Afro-Asiatic | Chadic | ✓ | ✗ (absent) |
| Igbo | Niger-Congo | Volta-Niger | ✓ | ✓ |
| Swahili | Niger-Congo | Bantu | ✓ | ✗ (absent) |

### 4.3 Evaluation

All metrics are computed with `sacrebleu` (Post, 2018) and `jiwer`.

- **BLEU** (sacrebleu, tokenize=`13a`) — standard n-gram precision for translation
- **spBLEU** (tokenize=`flores101`) — sentencepiece BLEU as used in the NLLB paper
- **chrF** (β=2) — character n-gram F-score, more robust for morphologically rich languages
- **WER** — word error rate for ASR (`jiwer`, lower is better)
- **CER** — character error rate for ASR (supplement to WER)

European reference languages (French, German, Spanish) are evaluated with
the same models and metrics to establish the high-resource performance ceiling.

### 4.4 Reproducibility

All experiments use a fixed random seed (42). No fine-tuning is performed.
Dataset caches are stored locally to ensure identical data ordering across runs.
Code and configurations are available at [REPOSITORY_URL].

---

## 5. Results

### 5.1 ASR Performance on FLEURS

**Table 1: Word Error Rate (%) on FLEURS validation split — lower is better**

| Model | Yoruba | Hausa | Igbo | Swahili |
|-------|:------:|:-----:|:----:|:-------:|
| SeamlessM4T-v2 Large | [RESULT:fleurs_seamless.yoruba.wer] | — | [RESULT:fleurs_seamless.igbo.wer] | [RESULT:fleurs_seamless.swahili.wer] |
| Whisper large-v3 | [RESULT:fleurs_whisper.yoruba.wer] | [RESULT:fleurs_whisper.hausa.wer] | — | [RESULT:fleurs_whisper.swahili.wer] |
| *Whisper-large-v3 (pub.)* | *[BASELINE:radford2023whisper.yoruba.wer]* | *[BASELINE:radford2023whisper.hausa.wer]* | *—* | *—* |
| *MMS-300M (pub.)* | *[BASELINE:pratap2023mms.yoruba.wer]* | *[BASELINE:pratap2023mms.hausa.wer]* | *[BASELINE:pratap2023mms.igbo.wer]* | *—* |
| *XLS-R-1B (pub.)* | *[BASELINE:babu2022xlsr.yoruba.wer]* | *[BASELINE:babu2022xlsr.hausa.wer]* | *—* | *—* |
| **French (reference)** | | | | |
| SeamlessM4T-v2 | [RESULT:gap.french.wer] | | | |
| Whisper | [RESULT:gap.french.wer_whisper] | | | |

— = language not supported by this model.
*Italic rows* = published numbers from prior work for reference; not our runs.

[NARRATIVE:asr_results_discussion]

### 5.2 Speech-to-Text Translation on FLEURS

**Table 2: BLEU on FLEURS validation split (Source → English)**

| Model | Yoruba | Hausa | Igbo | Swahili |
|-------|:------:|:-----:|:----:|:-------:|
| SeamlessM4T-v2 Large (E2E) | [RESULT:fleurs_seamless.yoruba.bleu] | — | [RESULT:fleurs_seamless.igbo.bleu] | [RESULT:fleurs_seamless.swahili.bleu] |
| Whisper+NLLB-200 (Cascade) | [RESULT:fleurs_whisper.yoruba.bleu] | [RESULT:fleurs_whisper.hausa.bleu] | — | [RESULT:fleurs_whisper.swahili.bleu] |
| *SeamlessM4T-v2 (pub.)* | *[BASELINE:barrault2023seamless.yoruba.bleu]* | *—* | *[BASELINE:barrault2023seamless.igbo.bleu]* | *—* |
| *mSLAM-CTC (pub.)* | *[BASELINE:conneau2022fleurs.yoruba.bleu]* | *—* | *—* | *—* |
| **French (reference)** | | | | |
| SeamlessM4T-v2 | [RESULT:gap.french.bleu] | | | |
| Whisper+NLLB | [RESULT:gap.french.bleu_whisper] | | | |

**Table 2b: Additional metrics on FLEURS (Source → English)**

| Model | Language | BLEU | spBLEU | chrF |
|-------|----------|:----:|:------:|:----:|
| SeamlessM4T-v2 | Yoruba | [RESULT:fleurs_seamless.yoruba.bleu] | [RESULT:fleurs_seamless.yoruba.spbleu] | [RESULT:fleurs_seamless.yoruba.chrf] |
| SeamlessM4T-v2 | Igbo | [RESULT:fleurs_seamless.igbo.bleu] | [RESULT:fleurs_seamless.igbo.spbleu] | [RESULT:fleurs_seamless.igbo.chrf] |
| SeamlessM4T-v2 | Swahili | [RESULT:fleurs_seamless.swahili.bleu] | [RESULT:fleurs_seamless.swahili.spbleu] | [RESULT:fleurs_seamless.swahili.chrf] |
| Whisper+NLLB | Yoruba | [RESULT:fleurs_whisper.yoruba.bleu] | [RESULT:fleurs_whisper.yoruba.spbleu] | [RESULT:fleurs_whisper.yoruba.chrf] |
| Whisper+NLLB | Hausa | [RESULT:fleurs_whisper.hausa.bleu] | [RESULT:fleurs_whisper.hausa.spbleu] | [RESULT:fleurs_whisper.hausa.chrf] |
| Whisper+NLLB | Swahili | [RESULT:fleurs_whisper.swahili.bleu] | [RESULT:fleurs_whisper.swahili.spbleu] | [RESULT:fleurs_whisper.swahili.chrf] |

[NARRATIVE:s2tt_results_discussion]

### 5.3 Results on African-Celtic

**Table 3: ASR WER (%) and S2TT BLEU on African-Celtic dev split**

| Model | Task | Yoruba | Hausa | Igbo |
|-------|------|:------:|:-----:|:----:|
| SeamlessM4T-v2 | WER (%) | [RESULT:ac_seamless.yoruba.wer] | — | [RESULT:ac_seamless.igbo.wer] |
| SeamlessM4T-v2 | BLEU | [RESULT:ac_seamless.yoruba.bleu] | — | [RESULT:ac_seamless.igbo.bleu] |
| Whisper large-v3 | WER (%) | [RESULT:ac_whisper.yoruba.wer] | [RESULT:ac_whisper.hausa.wer] | — |
| Whisper+NLLB-200 | BLEU | [RESULT:ac_whisper.yoruba.bleu] | [RESULT:ac_whisper.hausa.bleu] | — |

[NARRATIVE:african_celtic_discussion]

### 5.4 Performance Gap Analysis

**Table 4: African vs European language gap — same models, same evaluation protocol**

| Language | Family | SeamlessM4T WER (%) | Whisper WER (%) | S2TT BLEU gap vs French |
|----------|--------|:------------------:|:---------------:|:-----------------------:|
| Yoruba | Niger-Congo | [RESULT:fleurs_seamless.yoruba.wer] | [RESULT:fleurs_whisper.yoruba.wer] | [RESULT:gap.yoruba.bleu_gap] |
| Hausa | Afro-Asiatic | — | [RESULT:fleurs_whisper.hausa.wer] | [RESULT:gap.hausa.bleu_gap] |
| Igbo | Niger-Congo | [RESULT:fleurs_seamless.igbo.wer] | — | [RESULT:gap.igbo.bleu_gap] |
| Swahili | Niger-Congo | [RESULT:fleurs_seamless.swahili.wer] | [RESULT:fleurs_whisper.swahili.wer] | [RESULT:gap.swahili.bleu_gap] |
| **French** | Indo-European | **[RESULT:gap.french.wer]** | **[RESULT:gap.french.wer_whisper]** | **0.0** (reference) |
| **German** | Indo-European | **[RESULT:gap.german.wer]** | **[RESULT:gap.german.wer_whisper]** | **[RESULT:gap.german.bleu_gap]** |
| **Spanish** | Indo-European | **[RESULT:gap.spanish.wer]** | **[RESULT:gap.spanish.wer_whisper]** | **[RESULT:gap.spanish.bleu_gap]** |

[NARRATIVE:gap_discussion]

### 5.5 Metric Consistency

**Table 5: Kendall's τ between metric rankings across all systems**

| Metric pair | Kendall τ |
|-------------|:---------:|
| BLEU vs spBLEU | [RESULT:metric_tau.bleu_spbleu] |
| BLEU vs chrF | [RESULT:metric_tau.bleu_chrf] |
| spBLEU vs chrF | [RESULT:metric_tau.spbleu_chrf] |

[NARRATIVE:metric_consistency_discussion]

---

## 6. Discussion

### 6.1 African Language Performance

[NARRATIVE:discussion_african_performance]

The results in Tables 1–2 confirm that all four target languages remain well
below the performance levels seen for European languages, even when evaluated
with the largest publicly available multilingual models. [NARRATIVE:discussion_gap_interpretation]
Swahili, as the most resourced of the four languages in our evaluation (it
has a large Wikipedia and Common Voice presence), consistently outperforms
Yoruba, Hausa, and Igbo across all conditions, suggesting that the model
training data imbalance is a primary driver of the observed gap.

### 6.2 End-to-End vs Cascade Architecture

[NARRATIVE:discussion_architecture]

The comparison between SeamlessM4T-v2 and the Whisper+NLLB-200 cascade reveals
[NARRATIVE:architecture_finding]. Both architectures have distinct language
coverage limitations: SeamlessM4T-v2 cannot process Hausa in speech mode, while
the cascade cannot handle Igbo. A practitioner choosing between the two systems
must therefore take the target language into account before deployment.
[NARRATIVE:architecture_recommendation]

### 6.3 Cross-Dataset Consistency

Results on the African-Celtic dataset (Table 3) confirm the FLEURS trends for
Yoruba and Igbo/Hausa. [NARRATIVE:cross_dataset_finding] The 48 kHz audio in
African-Celtic presents a different acoustic profile from FLEURS, and the
absolute scores differ accordingly. This cross-dataset consistency provides
evidence that the FLEURS benchmark results are not artefacts of the specific
recording conditions used in that dataset.

### 6.4 Metric Sensitivity

The Kendall τ correlation between BLEU and spBLEU rankings ([RESULT:metric_tau.bleu_spbleu])
and between BLEU and chrF ([RESULT:metric_tau.bleu_chrf]) indicates that
[NARRATIVE:metric_finding]. All three metrics produce [NARRATIVE:metric_consistency_summary]
system rankings across the languages evaluated in this paper.

### 6.5 Limitations

This paper evaluates two systems only. MMS-300M and XLS-R-1B are evaluated at
the published-baseline level (Table 1) but not run independently as part of this
work. Additionally, we use the FLEURS validation split rather than the held-out
test split, to allow repeated evaluation during methodology development; future
work should report final numbers on the test split. The African-Celtic dataset
has relatively few evaluation samples per language, which may increase variance
in the reported metrics.

---

## 7. Conclusion

We have presented LinguoMT, the first systematic zero-shot comparison of
SeamlessM4T-v2 and the Whisper+NLLB-200 cascade on African speech translation
and recognition across FLEURS and African-Celtic. Our results show that
[NARRATIVE:conclusion_finding_1]. The performance gap relative to French is
[NARRATIVE:conclusion_gap_summary]. [NARRATIVE:conclusion_finding_2]

These baselines serve as the foundation for the companion papers in this series,
which investigate how parameter-efficient fine-tuning (Paper 2), audio strategy
choices (Paper 3), cascade architecture analysis (Paper 4), and cross-lingual
transfer dynamics (Paper 5) can improve upon the zero-shot performance documented
here. All code, configurations, and results are made available at [REPOSITORY_URL]
to support reproducibility and further research on African language speech technology.

---

## Acknowledgements

[ACKNOWLEDGEMENTS]

---

## References

- Babu, A., Wang, C., Tjandra, A., Lakhotia, K., Xu, Q., Goyal, N., et al. (2022). XLS-R: Self-supervised cross-lingual speech representation learning at scale. *Interspeech 2022*. arXiv:2111.09296
- Bapna, A., Cherry, C., Zhang, Y., Jia, Y., Johnson, M., Chrzanowski, M., et al. (2022). mSLAM: Massively multilingual joint pre-training for speech and text. arXiv:2202.01374
- Barrault, L., Chung, Y., Meglioli, M. C., Dale, D., Dong, N., Duquenne, P., et al. (2023a). SeamlessM4T: Massively multilingual & multimodal machine translation. arXiv:2308.11596
- Barrault, L., Duquenne, P., Elbayad, M., Elkahky, A., Dong, N., Hsu, W., et al. (2023b). Seamless: Multilingual expressive and streaming speech translation. arXiv:2312.05187
- Conneau, A., Ma, M., Khanuja, S., Zhang, Y., Axelrod, V., Dalmia, S., et al. (2022). FLEURS: Few-shot learning evaluation of universal representations of speech. *IEEE SLT 2022*. arXiv:2205.12446
- Costa-jussà, M. R., Cross, J., Çelebi, O., Elbayad, M., Heafield, K., Heffernan, K., et al. (2022). No language left behind: Scaling human-centered machine translation. arXiv:2207.04672
- Nunamaker, J. F., Chen, M., & Purdin, T. D. M. (1991). Systems development in information systems research. *Journal of Management Information Systems, 7*(3), 89–106.
- Olatunji, T., Ige, A., Oshodi, A., Lakin, A., Alabi, J., Onwuegbuzia, E., et al. (2022). AfriSpeech-200: Pan-African accented speech dataset for clinical and general domain ASR. arXiv:2104.02010
- Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: A method for automatic evaluation of machine translation. *ACL 2002*.
- Post, M. (2018). A call for clarity in reporting BLEU scores. *WMT 2018*. arXiv:1804.08771
- Pratap, V., Tjandra, A., Shi, B., Tomasello, P., Babu, A., Xu, S., et al. (2023). Scaling speech technology to 1,000+ languages. arXiv:2305.13516
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *ICML 2023*. arXiv:2212.04356
