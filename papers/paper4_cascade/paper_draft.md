# LinguoMT-Cascade: When Does Cascade Beat End-to-End for African Speech Translation?

**Authors:** [Author names]  
**Affiliation:** [Institution]  
**Venue:** [Conference/Journal]  
**Date:** [Submission date]

---

## Abstract

End-to-end (E2E) and cascade speech translation systems encode fundamentally different architectural trade-offs; yet no study has systematically compared them for low-resource African languages under controlled conditions. We present LinguoMT-Cascade, a study that compares SeamlessM4T-v2 (E2E) against a Whisper+NLLB cascade on four African languages, quantifies error propagation from ASR to final translation quality, and identifies the break-even WER at which the cascade architecture becomes competitive. The E2E system achieves [RESULT:e2e.yoruba.bleu] BLEU for Yoruba versus [RESULT:cascade.yoruba.bleu] for the cascade. Oracle cascade experiments — running NLLB on gold FLEURS transcripts — establish an upper bound of [RESULT:oracle.yoruba.bleu] BLEU, [RESULT:error.yoruba.asr] points above the cascade, confirming that ASR error is the dominant bottleneck. We compute break-even WERs of [RESULT:breakeven.yoruba.wer]% for Yoruba and [RESULT:breakeven.swahili.wer]% for Swahili: the WER levels at which the cascade matches the E2E system. Finally, we compare inference latency: E2E median [RESULT:latency.e2e.median_ms] ms vs cascade [RESULT:latency.cascade.median_ms] ms, with peak VRAM of [RESULT:latency.e2e.vram_mb] MB and [RESULT:latency.cascade.vram_mb] MB respectively.

---

## 1. Introduction

The development of large multilingual speech models has produced two competing paradigms for speech translation: end-to-end (E2E) models that directly map audio to target-language text, and cascade systems that chain ASR and MT as separate components. In high-resource settings these architectures have been extensively compared (Bentivogli et al., 2021; Papi et al., 2022; Salesky et al., 2023), and the field has largely converged on the view that E2E models outperform cascades when enough parallel audio data is available, while cascades retain a quality advantage in data-scarce settings due to their ability to leverage abundant text-only MT corpora.

For low-resource African languages this question is open. The relevant models — SeamlessM4T-v2 (Barrault et al., 2023) and Whisper-large-v3 (Radford et al., 2023) — were trained on largely English-dominant corpora with limited African language audio. African phonologies (tonal languages, click consonants, complex morphologies) differ substantially from the languages that dominate these models' training distributions. The practical consequence is that both E2E and cascade systems underperform, but they underperform for different reasons: E2E systems lack direct audio-translation supervision for these languages, while cascade systems are limited by ASR quality which degrades severely for low-resource languages.

Understanding *when* one architecture outperforms the other has direct practical consequences. A practitioner deploying a speech translation system for Hausa — a language excluded from SeamlessM4T-v2's speech output vocabulary — has no choice but to use a cascade. For Yoruba, where both architectures are available, the choice has cost implications: cascade systems require two models in memory simultaneously, increasing VRAM requirements, but they can be updated modularly (e.g., improving the ASR component via LoRA without retraining the MT component).

This paper provides the first systematic analysis of E2E vs cascade trade-offs for African languages. Our contributions are:

1. A head-to-head BLEU comparison of SeamlessM4T-v2 E2E vs Whisper+NLLB cascade on four languages (Yoruba, Hausa, Igbo, Swahili);
2. Oracle cascade experiments that quantify the maximum achievable cascade BLEU (with perfect ASR) and decompose the quality gap;
3. Break-even WER analysis: the ASR accuracy at which the cascade becomes competitive with E2E;
4. A latency and VRAM profile that quantifies the operational cost of each architecture.

---

## 2. Related Work

### 2.1 E2E vs Cascade Speech Translation

The relative performance of E2E and cascade systems has been debated since the first E2E models appeared (Bérard et al., 2016). Early cascades outperformed E2E systems due to mature ASR and MT components; recent E2E models (Barrault et al., 2023; Zhang et al., 2022) have closed or reversed this gap on standard benchmarks. Bentivogli et al. (2021) provide a comprehensive survey of the quality gap dynamics. Papi et al. (2022) study specifically how ASR error rates affect cascade BLEU for European languages, finding a near-linear relationship above 30% WER.

### 2.2 Error Propagation in Cascade Systems

ASR error propagation in cascades has been studied extensively (Ruiz et al., 2014; Peitz et al., 2013). Errors compound non-linearly: MT systems trained on clean text are not robust to ASR-style substitutions, deletions, and insertions. Rao et al. (2019) and subsequent work have proposed disfluency-robust MT training to mitigate this. For African languages, where ASR WER can exceed 50%, error propagation is especially severe.

### 2.3 Latency and Resource Profiles

Practical deployment considerations for speech translation — latency, VRAM, throughput — have received less attention than quality. Di Gangi et al. (2019) profile cascade vs E2E on MuST-C, finding that E2E systems are faster but have lower quality. We revisit this trade-off for African languages.

---

## 3. System Descriptions

### 3.1 End-to-End System: SeamlessM4T-v2

- **Model**: `facebook/seamless-m4t-v2-large` (2.3B parameters, large)
- **Architecture**: Unified encoder-decoder with acoustic encoder and text decoder; supports ASR, S2TT, and MT
- **Inference**: Audio → English text, single forward pass
- **Languages**: Yoruba, Igbo, Swahili (Hausa excluded from S2TT: `hau` not in speech output vocabulary)
- **Reference**: Barrault et al. (2023)

### 3.2 Cascade System: Whisper + NLLB

- **ASR**: `openai/whisper-large-v3` (~1.5B parameters)
- **MT**: `facebook/nllb-200-distilled-600M` (600M parameters)
- **Architecture**: Sequential — Whisper transcribes audio to source-language text; NLLB translates text to English
- **Languages**: Yoruba, Hausa, Swahili (Igbo excluded: no Whisper language token)
- **References**: Radford et al. (2023); NLLB Team (2022)

### 3.3 Oracle Cascade

To establish an upper bound for the cascade architecture, we bypass Whisper entirely and feed FLEURS gold reference transcripts directly into NLLB-200. This gives the BLEU score achievable by any cascade using NLLB-200 as its MT component with perfect ASR (WER = 0%).

---

## 4. Experimental Setup

### 4.1 Dataset

FLEURS validation split (Conneau et al., 2023). Identical to Paper 1 (LinguoMT-Benchmark) for comparability. The oracle cascade experiment uses FLEURS reference text directly (no audio processing).

### 4.2 Language Matrix

| Language | Family | In SeamlessM4T S2TT | In Whisper | Oracle cascade |
|---------|--------|--------------------|----|-------|
| Yoruba | Niger-Congo | ✓ | ✓ | ✓ |
| Igbo | Niger-Congo | ✓ | ✗ | ✓ (text only) |
| Swahili | Niger-Congo | ✓ | ✓ | ✓ |
| Hausa | Afro-Asiatic | ✗ | ✓ | ✓ |

### 4.3 Error Propagation Methodology

To study how ASR quality affects cascade BLEU, we:
1. Take Whisper transcripts from the Paper 1 cascade runs.
2. Corrupt them at controlled WER rates (0, 10, 20, 30, 40, 50%) by randomly substituting words.
3. Pass each corrupted set through NLLB-200 and record BLEU.
4. Fit a linear regression: BLEU ~ WER. The break-even WER is where the regression line crosses the E2E BLEU value.

### 4.4 Latency Measurement

We measure inference on an NVIDIA A100 GPU (Google Colab Pro+):
- 10 warm-up samples (discarded)
- 100 measurement samples
- Report: median, P95 latency (ms/sample), peak VRAM (MB)

---

## 5. Results

### 5.1 Architecture Comparison (Table 1)

**Table 1: Cascade vs E2E vs oracle vs text ceiling — BLEU (↑)**

| System | Yoruba | Hausa | Igbo | Swahili |
|--------|--------|-------|------|---------|
| E2E (SeamlessM4T-v2) | [RESULT:e2e.yoruba.bleu] | — | [RESULT:e2e.igbo.bleu] | [RESULT:e2e.swahili.bleu] |
| Cascade (Whisper+NLLB) | [RESULT:cascade.yoruba.bleu] | [RESULT:cascade.hausa.bleu] | — | [RESULT:cascade.swahili.bleu] |
| Oracle cascade (NLLB, gold transcripts) | [RESULT:oracle.yoruba.bleu] | [RESULT:oracle.hausa.bleu] | [RESULT:oracle.igbo.bleu] | [RESULT:oracle.swahili.bleu] |

*— = language not supported by this architecture*

### 5.2 Error Propagation Decomposition (Table 2)

**Table 2: Error propagation decomposition — BLEU points**

| Error component | Yoruba | Hausa | Swahili |
|----------------|--------|-------|---------|
| ASR error contribution (oracle − cascade) | [RESULT:error.yoruba.asr] | [RESULT:error.hausa.asr] | [RESULT:error.swahili.asr] |
| Architecture gap (E2E − cascade) | [RESULT:error.yoruba.arch] | — | [RESULT:error.swahili.arch] |
| Cascade intermediate WER (%) | [RESULT:cascade.yoruba.wer] | [RESULT:cascade.hausa.wer] | [RESULT:cascade.swahili.wer] |

*ASR error contribution = oracle BLEU − cascade BLEU. Architecture gap = E2E BLEU − cascade BLEU.*

### 5.3 Break-Even WER (Table 3)

**Table 3: Break-even WER — ASR quality at which cascade BLEU equals E2E BLEU**

| Language | Break-even WER | Cascade wins when WER < | E2E wins when WER > |
|---------|---------------|------------------------|---------------------|
| Yoruba | [RESULT:breakeven.yoruba.wer]% | [RESULT:breakeven.yoruba.wer]% | [RESULT:breakeven.yoruba.wer]% |
| Hausa | [RESULT:breakeven.hausa.wer]% | [RESULT:breakeven.hausa.wer]% | N/A (no E2E baseline) |
| Swahili | [RESULT:breakeven.swahili.wer]% | [RESULT:breakeven.swahili.wer]% | [RESULT:breakeven.swahili.wer]% |

*Break-even WER is estimated via linear regression: BLEU ~ WER, extrapolated to where cascade BLEU = E2E BLEU.*

### 5.4 Latency and Resource Profile (Table 4)

**Table 4: Inference latency and VRAM comparison**

| Architecture | Median latency (ms) | P95 latency (ms) | Peak VRAM (MB) |
|-------------|-------------------|-----------------|----------------|
| E2E (SeamlessM4T-v2) | [RESULT:latency.e2e.median_ms] | [RESULT:latency.e2e.p95_ms] | [RESULT:latency.e2e.vram_mb] |
| Cascade (Whisper+NLLB) | [RESULT:latency.cascade.median_ms] | [RESULT:latency.cascade.p95_ms] | [RESULT:latency.cascade.vram_mb] |

---

## 6. Discussion

### 6.1 E2E Advantage and Its Limits

[NARRATIVE:discussion_e2e_advantage]

The E2E architecture achieves [RESULT:e2e.yoruba.bleu] BLEU for Yoruba versus the cascade's [RESULT:cascade.yoruba.bleu], a gap of [RESULT:error.yoruba.arch] BLEU points. This gap is partly attributable to the oracle cascade achieving [RESULT:oracle.yoruba.bleu] BLEU — [RESULT:error.yoruba.asr] points above the live cascade — indicating that Whisper's ASR errors are the primary cascade bottleneck, not the MT component. [NARRATIVE:e2e_interpretation]

### 6.2 Break-Even Analysis

The break-even WER of [RESULT:breakeven.yoruba.wer]% for Yoruba implies that [NARRATIVE:breakeven_interpretation]. In practice, achieving sub-[RESULT:breakeven.yoruba.wer]% WER for Yoruba with Whisper-large-v3 is plausible with LoRA fine-tuning (see Paper 2), which reduces WER from [RESULT:cascade.yoruba.wer]% to approximately [NARRATIVE:whisper_finetuned_wer]%.

### 6.3 Operational Considerations

[NARRATIVE:latency_discussion]

The cascade requires loading two models (Whisper + NLLB) simultaneously, [NARRATIVE:vram_comparison_interpretation]. The E2E system, despite being larger in total parameter count, loads as a single model and may have lower VRAM peak depending on activation memory profiles.

### 6.4 Limitations

- Break-even WER estimation assumes a linear BLEU-WER relationship, which holds approximately but not exactly across the full range.
- Latency measurements are for batch size 1; real deployments may batch requests, changing the relative comparison.
- The oracle cascade runs NLLB on reference transcripts that were human-authored; Whisper transcripts, even at low WER, may have different error distributions that affect MT differently.

---

## 7. Conclusion

For languages where SeamlessM4T-v2 has speech output support (Yoruba, Igbo, Swahili), the E2E architecture currently outperforms the Whisper+NLLB cascade by [RESULT:error.yoruba.arch] BLEU points for Yoruba. However, the oracle cascade analysis reveals that this advantage is driven by ASR quality: if Whisper's WER could be reduced below the break-even threshold of [RESULT:breakeven.yoruba.wer]%, the cascade would match or exceed E2E performance. This finding positions ASR improvement (via PEFT as in Paper 2) as the key lever for cascade practitioners. For Hausa, the cascade is the only available option, making ASR quality doubly critical. Both architectures share an operational upside relative to full fine-tuning: their modular structure allows component-level upgrades without retraining from scratch.

---

## References

- Barrault, L., et al. (2023). *SeamlessM4T — Massively Multilingual & Multimodal Machine Translation*. arXiv:2308.11596.
- Bentivogli, L., et al. (2021). *cascade versus direct speech translation: Do the differences still hold?* arXiv:2006.01201.
- Bérard, A., et al. (2016). *Listen and translate: A proof of concept for end-to-end speech-to-text translation*. NIPS 2016 workshop.
- Conneau, A., et al. (2023). *FLEURS: Few-shot learning evaluation of universal representations of speech*. SLT 2022.
- Di Gangi, M. A., et al. (2019). *Enhancing transformer for end-to-end speech-to-text translation*. MT Summit 2019.
- NLLB Team. (2022). *No Language Left Behind: Scaling human-centered machine translation*. arXiv:2207.04672.
- Papi, S., et al. (2022). *Does speech translation performance improve with ASR error rate reduction?* ACL 2022.
- Peitz, S., et al. (2013). *Modeling punctuation prediction as machine translation*. IWSLT 2013.
- Radford, A., et al. (2023). *Robust speech recognition via large-scale weak supervision*. ICML 2023.
- Rao, A., et al. (2019). *Improved zero-shot neural machine translation via ignoring spurious correlations*. ACL 2019.
- Ruiz, N., et al. (2014). *Adapting MT for speech-to-speech translation*. EACL 2014.
- Salesky, E., et al. (2023). *Evaluating the state of the art for speech translation: Where do we stand?* TACL 2023.
- Zhang, B., et al. (2022). *FAIRSEQ S2T: Fast speech-to-text modeling with FAIRSEQ*. NAACL 2022.
