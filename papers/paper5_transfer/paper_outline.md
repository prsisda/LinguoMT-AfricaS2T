# LinguoMT-Transfer: Linguistic Family Drives Cross-Lingual Transfer for African Speech

> Placeholders: `[RESULT:key]` — filled by `python papers/fill_results.py paper5_transfer`
> Narrative: `[NARRATIVE:key]` — fill manually.
> Zero-shot baselines (`[RESULT:zero.*]`) are imported from Paper 1.

---

## Abstract

We analyse how linguistic family membership predicts cross-lingual transfer
efficiency in speech translation and recognition for African languages. Comparing
Niger-Congo (Yoruba, Igbo, Swahili) against Afro-Asiatic (Hausa), we find
that typological similarity — as measured by URIEL cosine distance ([RESULT:topo.yoruba_igbo]
for Yoruba↔Igbo vs [RESULT:topo.yoruba_hausa] for Yoruba↔Hausa) —
[NARRATIVE:abstract_finding]. Crucially, cross-lingual fine-tuning within
the Niger-Congo family reaches comparable performance to monolingual fine-tuning
with [NARRATIVE:efficiency_finding]% fewer training samples.

---

## 1. Introduction

[NARRATIVE:intro]

---

## 2. Languages and Typological Distance

### 2.1 Language Families

| Language | Family | Subfamily | ISO 639-1 |
|----------|--------|-----------|-----------|
| Yoruba | Niger-Congo | Volta-Niger | yo |
| Igbo | Niger-Congo | Volta-Niger | ig |
| Swahili | Niger-Congo | Bantu | sw |
| Hausa | Afro-Asiatic | Chadic | ha |

### 2.2 Typological Similarity (Table 1)

**Table 1: URIEL cosine similarity (syntax_knn features)**

| | Yoruba | Igbo | Hausa | English |
|-|--------|------|-------|---------|
| **Yoruba** | 1.00 | [RESULT:topo.yoruba_igbo] | [RESULT:topo.yoruba_hausa] | [RESULT:topo.yoruba_english] |
| **Igbo** | [RESULT:topo.yoruba_igbo] | 1.00 | [RESULT:topo.igbo_hausa] | — |
| **Hausa** | [RESULT:topo.yoruba_hausa] | [RESULT:topo.igbo_hausa] | 1.00 | — |

[NARRATIVE:typology_discussion]

---

## 3. Results

### 3.1 Zero-Shot Baselines (from Paper 1)

| Language | Family | SeamlessM4T BLEU | Whisper WER |
|----------|--------|------------------|------------|
| Yoruba | Niger-Congo | [RESULT:zero.yoruba.bleu] | [RESULT:zero.yoruba.wer] |
| Igbo | Niger-Congo | [RESULT:zero.igbo.bleu] | — |
| Hausa | Afro-Asiatic | — | [RESULT:zero.hausa.wer] |

### 3.2 Transfer Strategy Comparison (Table 2)

**Table 2: WER (ASR) and BLEU (S2TT) by transfer strategy**

| Strategy | Yoruba WER | Hausa WER | Yoruba BLEU | Igbo BLEU |
|----------|-----------|----------|------------|----------|
| Zero-shot | [RESULT:zero.yoruba.wer] | [RESULT:zero.hausa.wer] | [RESULT:zero.yoruba.bleu] | [RESULT:zero.igbo.bleu] |
| Few-shot (100 samples) | [RESULT:few.yoruba.wer.100] | [RESULT:few.hausa.wer.100] | — | — |
| Cross-lingual FT (Yoruba→Igbo) | — | — | — | [RESULT:xfer.yor_to_ibo.bleu] |
| Cross-lingual FT (Yoruba→Hausa) | — | [RESULT:xfer.yor_to_hau.wer] | — | — |

[NARRATIVE:table2_discussion]

### 3.3 Few-Shot Learning Curves (Table 3 / Figure 1)

**Table 3: WER/BLEU vs training samples — by language family**

| Samples | Yoruba WER (NC) | Igbo BLEU (NC) | Hausa WER (AA) |
|---------|----------------|---------------|---------------|
| 0 (zero) | [RESULT:zero.yoruba.wer] | [RESULT:zero.igbo.bleu] | [RESULT:zero.hausa.wer] |
| 25 | [RESULT:few.yoruba.wer.25] | [RESULT:few.igbo.bleu.25] | [RESULT:few.hausa.wer.25] |
| 50 | [RESULT:few.yoruba.wer.50] | [RESULT:few.igbo.bleu.50] | [RESULT:few.hausa.wer.50] |
| 100 | [RESULT:few.yoruba.wer.100] | [RESULT:few.igbo.bleu.100] | [RESULT:few.hausa.wer.100] |
| 200 | [RESULT:few.yoruba.wer.200] | [RESULT:few.igbo.bleu.200] | [RESULT:few.hausa.wer.200] |

Transfer efficiency per sample: Niger-Congo [RESULT:efficiency.nc] vs Afro-Asiatic [RESULT:efficiency.aa].

[NARRATIVE:scaling_discussion]

### 3.4 Interaction Analysis

Regression: WER ~ log(samples) × language_family

| Term | Coefficient | p-value |
|------|------------|---------|
| log(samples) | [RESULT:interaction.log_coeff] | [RESULT:interaction.log_pvalue] |
| lang_family (AA) | [RESULT:interaction.family_coeff] | [RESULT:interaction.family_pvalue] |
| **Interaction** | [RESULT:interaction.coeff] | [RESULT:interaction.pvalue] |

[NARRATIVE:interaction_discussion]

---

## 4. Discussion

[NARRATIVE:discussion]

### 4.1 Linguistic Relatedness as a Predictor

[NARRATIVE:linguistic_relatedness]

### 4.2 Practical Implications

[NARRATIVE:practical_implications]

### 4.3 Limitations

[NARRATIVE:limitations]

---

## 5. Conclusion

[NARRATIVE:conclusion]

---

## References

- Paper 1 (this series): LinguoMT Benchmark
- Paper 2 (this series): LinguoMT-Adapt
- Littell et al. (2017). URIEL and lang2vec. ACL 2017.
- Pratap et al. (2023). MMS. arXiv:2305.13516
- Babu et al. (2022). XLS-R. arXiv:2111.09296
- Conneau et al. (2022). FLEURS. arXiv:2205.12446
