# LinguoMT-Adapt: Parameter-Efficient Adaptation for African Speech Translation

> Placeholders: `[RESULT:key]` — filled by `python papers/fill_results.py paper2_adaptation`
> Narrative: `[NARRATIVE:key]` — fill manually after reviewing results.
> Zero-shot baselines (`[RESULT:*.before]`) are imported from Paper 1.

---

## Abstract

We investigate parameter-efficient fine-tuning (PEFT) for African speech
translation and recognition. Starting from the zero-shot baselines established
in LinguoMT (Paper 1), we apply LoRA and adapter modules to SeamlessM4T-v2-large
and Whisper-large-v3 on FLEURS. With only [RESULT:lora.trainable_pct]% of
parameters updated, LoRA achieves [NARRATIVE:abstract_gain] improvement over
zero-shot on Yoruba S2TT. [NARRATIVE:abstract_finding]

---

## 1. Introduction

[NARRATIVE:intro]

---

## 2. Setup

**Zero-shot baselines** (from Paper 1, Table 2):

| Language | SeamlessM4T BLEU (0-shot) | Whisper WER (0-shot) |
|----------|---------------------------|----------------------|
| Yoruba   | [RESULT:seamless_lora.yoruba.bleu.before] | [RESULT:whisper_lora.yoruba.wer.before] |
| Hausa    | —  | [RESULT:whisper_lora.hausa.wer.before] |
| Igbo     | [RESULT:seamless_lora.igbo.bleu.before] | — |
| Swahili  | [RESULT:seamless_lora.swahili.bleu.before] | — |

**Fine-tuning data:** FLEURS train split, [RESULT:lora.trainable_params_m]M trainable
parameters (LoRA), seed 42, main budget = 1 000 samples per language.

---

## 3. Results

### 3.1 Before / After Adaptation (Table 1)

**Table 1: BLEU (S2TT) and WER (ASR) — zero-shot vs after fine-tuning**

| Method | Trainable % | Yoruba BLEU | Igbo BLEU | Hausa WER | Yoruba WER |
|--------|------------|-------------|-----------|-----------|-----------|
| Zero-shot | 0 | [RESULT:seamless_lora.yoruba.bleu.before] | [RESULT:seamless_lora.igbo.bleu.before] | [RESULT:whisper_lora.hausa.wer.before] | [RESULT:whisper_lora.yoruba.wer.before] |
| LoRA | [RESULT:lora.trainable_pct] | [RESULT:seamless_lora.yoruba.bleu.after] | [RESULT:seamless_lora.igbo.bleu.after] | [RESULT:whisper_lora.hausa.wer.after] | [RESULT:whisper_lora.yoruba.wer.after] |
| Adapter | [RESULT:adapter.trainable_pct] | [RESULT:seamless_adapter.yoruba.bleu.after] | [RESULT:seamless_adapter.igbo.bleu.after] | — | — |
| *Full FT (published)* | *100* | *[BASELINE:lugo2022masakhane.yoruba.wer]* | *—* | *[BASELINE:lugo2022masakhane.hausa.wer]* | *—* |

[NARRATIVE:table1_discussion]

### 3.2 Parameter Efficiency (Table 2)

**Table 2: Parameter budget comparison**

| Method | Trainable (M) | % of total | BLEU gain (Yoruba) | WER gain (Yoruba) | GPU hours |
|--------|--------------|-----------|-------------------|------------------|-----------|
| LoRA | [RESULT:lora.trainable_params_m] | [RESULT:lora.trainable_pct] | [RESULT:seamless_lora.yoruba.bleu.gain] | [RESULT:seamless_lora.yoruba.wer.gain] | [RESULT:lora.gpu_hours] |
| Adapter | [RESULT:adapter.trainable_params_m] | [RESULT:adapter.trainable_pct] | [RESULT:seamless_adapter.yoruba.bleu.gain] | — | [RESULT:adapter.gpu_hours] |

[NARRATIVE:efficiency_discussion]

### 3.3 Data Scaling (Table 3 / Figure 1)

**Table 3: BLEU/WER at different fine-tuning data budgets — LoRA, Yoruba**

| Budget (samples) | Yoruba BLEU | Yoruba WER | Igbo BLEU | Hausa WER |
|-----------------|------------|-----------|----------|----------|
| 0 (zero-shot) | [RESULT:seamless_lora.yoruba.bleu.before] | [RESULT:whisper_lora.yoruba.wer.before] | [RESULT:seamless_lora.igbo.bleu.before] | [RESULT:whisper_lora.hausa.wer.before] |
| 100 | [RESULT:scaling.yoruba.bleu.100] | [RESULT:scaling.yoruba.wer.100] | [RESULT:scaling.igbo.bleu.100] | [RESULT:scaling.hausa.wer.100] |
| 500 | [RESULT:scaling.yoruba.bleu.500] | [RESULT:scaling.yoruba.wer.500] | [RESULT:scaling.igbo.bleu.500] | [RESULT:scaling.hausa.wer.500] |
| 1 000 | [RESULT:scaling.yoruba.bleu.1000] | [RESULT:scaling.yoruba.wer.1000] | [RESULT:scaling.igbo.bleu.1000] | [RESULT:scaling.hausa.wer.1000] |
| Full train | [RESULT:scaling.yoruba.bleu.full] | [RESULT:scaling.yoruba.wer.full] | [RESULT:scaling.igbo.bleu.full] | [RESULT:scaling.hausa.wer.full] |

Minimum data threshold (significant improvement, p<0.05): [RESULT:scaling.min_threshold_samples] samples.

[NARRATIVE:scaling_discussion]

---

## 4. Discussion

[NARRATIVE:discussion]

### 4.1 LoRA vs Adapter

[NARRATIVE:lora_vs_adapter]

### 4.2 Limitations

[NARRATIVE:limitations]

---

## 5. Conclusion

[NARRATIVE:conclusion]

---

## References

- Paper 1 (this series): LinguoMT Benchmark — zero-shot baselines
- Lugo et al. (2022). MasakhaSpeech. arXiv:2206.00253
- Olatunji et al. (2022). AfriSpeech. arXiv:2104.02010
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685
- Bapna & Firat (2019). Simple, Scalable Adaptation for Neural Machine Translation. EMNLP.
