# LinguoMT-Cascade: When Does Cascade Beat End-to-End for African Speech Translation?

> Placeholders: `[RESULT:key]` — filled by `python papers/fill_results.py paper4_cascade`
> Narrative: `[NARRATIVE:key]` — fill manually.
> E2E and cascade baselines (`[RESULT:e2e.*]`, `[RESULT:cascade.*]`) are imported from Paper 1.

---

## Abstract

We systematically compare cascade (Whisper + NLLB-200) and end-to-end
(SeamlessM4T-v2) architectures for African speech translation on FLEURS.
Using oracle and error-propagation experiments, we identify the break-even
ASR word error rate at which each architecture is preferred. For Yoruba,
the cascade matches or exceeds E2E performance when Whisper WER is below
[RESULT:breakeven.yoruba.wer]%. The E2E model has [NARRATIVE:latency_comparison]
latency advantage. [NARRATIVE:abstract_finding]

---

## 1. Introduction

[NARRATIVE:intro]

---

## 2. Setup

**Paper 1 baselines** (zero-shot, imported directly):

| Language | E2E BLEU (SeamlessM4T) | Cascade BLEU (Whisper+NLLB) | Cascade ASR WER |
|----------|------------------------|-----------------------------|--------------------|
| Yoruba | [RESULT:e2e.yoruba.bleu] | [RESULT:cascade.yoruba.bleu] | [RESULT:cascade.yoruba.wer] |
| Hausa | — | [RESULT:cascade.hausa.bleu] | [RESULT:cascade.hausa.wer] |
| Igbo | [RESULT:e2e.igbo.bleu] | — | — |
| Swahili | [RESULT:e2e.swahili.bleu] | [RESULT:cascade.swahili.bleu] | [RESULT:cascade.swahili.wer] |

---

## 3. Results

### 3.1 Architecture Comparison (Table 1)

**Table 1: BLEU across all architectures — Source → English**

| System | Yoruba | Hausa | Igbo | Swahili |
|--------|--------|-------|------|---------|
| E2E (SeamlessM4T-v2) | [RESULT:e2e.yoruba.bleu] | — | [RESULT:e2e.igbo.bleu] | [RESULT:e2e.swahili.bleu] |
| Cascade (Whisper+NLLB-600M) | [RESULT:cascade.yoruba.bleu] | [RESULT:cascade.hausa.bleu] | — | [RESULT:cascade.swahili.bleu] |
| Oracle cascade (gold ASR) | [RESULT:oracle.yoruba.bleu] | [RESULT:oracle.hausa.bleu] | [RESULT:oracle.igbo.bleu] | [RESULT:oracle.swahili.bleu] |

[NARRATIVE:table1_discussion]

### 3.2 Error Propagation (Table 2 / Figure 2)

**Table 2: BLEU gap decomposition**

| Component | Yoruba | Hausa | Swahili |
|-----------|--------|-------|---------|
| ASR error contribution (oracle − cascade) | [RESULT:error.yoruba.asr] | [RESULT:error.hausa.asr] | [RESULT:error.swahili.asr] |
| Architecture gap (E2E − cascade) | [RESULT:error.yoruba.arch] | — | [RESULT:error.swahili.arch] |

[NARRATIVE:error_propagation_discussion]

### 3.3 Break-even WER Analysis (Table 4)

**Table 4: WER threshold for architecture preference**

| Language | Break-even WER (%) | Cascade preferred when WER < | E2E preferred when WER > |
|----------|-------------------|------------------------------|--------------------------|
| Yoruba | [RESULT:breakeven.yoruba.wer] | [RESULT:breakeven.yoruba.wer]% | [RESULT:breakeven.yoruba.wer]% |
| Hausa | [RESULT:breakeven.hausa.wer] | [RESULT:breakeven.hausa.wer]% | [RESULT:breakeven.hausa.wer]% |
| Swahili | [RESULT:breakeven.swahili.wer] | [RESULT:breakeven.swahili.wer]% | [RESULT:breakeven.swahili.wer]% |

[NARRATIVE:breakeven_discussion]

### 3.4 Latency and Resources (Table 5)

**Table 5: Inference latency and VRAM on a single GPU**

| Architecture | Median latency (ms/sample) | P95 latency (ms/sample) | Peak VRAM (MB) |
|-------------|--------------------------|------------------------|----------------|
| E2E (SeamlessM4T-v2) | [RESULT:latency.e2e.median_ms] | [RESULT:latency.e2e.p95_ms] | [RESULT:latency.e2e.vram_mb] |
| Cascade (Whisper+NLLB) | [RESULT:latency.cascade.median_ms] | [RESULT:latency.cascade.p95_ms] | [RESULT:latency.cascade.vram_mb] |

[NARRATIVE:latency_discussion]

---

## 4. Discussion

[NARRATIVE:discussion]

### 4.1 Practical Recommendations

[NARRATIVE:recommendations]

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
