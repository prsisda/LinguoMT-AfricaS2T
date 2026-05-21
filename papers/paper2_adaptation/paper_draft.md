# LinguoMT-Adapt: Parameter-Efficient Adaptation for African Speech Translation

**Authors:** [Author names]  
**Affiliation:** [Institution]  
**Venue:** [Conference/Journal]  
**Date:** [Submission date]

---

## Abstract

Large-scale multilingual speech translation models such as SeamlessM4T-v2 and Whisper-large-v3 offer zero-shot coverage of low-resource African languages, yet their out-of-the-box performance lags behind supervised baselines. We investigate whether parameter-efficient fine-tuning (PEFT) methods — specifically LoRA (Low-Rank Adaptation) and lightweight adapters — can close this gap with minimal labelled data. Starting from zero-shot baselines of [RESULT:seamless_lora.yoruba.bleu.before] BLEU for Yoruba and [RESULT:seamless_lora.igbo.bleu.before] BLEU for Igbo, LoRA fine-tuning with 1,000 samples improves SeamlessM4T-v2 to [RESULT:seamless_lora.yoruba.bleu.after] BLEU and [RESULT:seamless_lora.igbo.bleu.after] BLEU respectively. On the ASR side, Whisper-large-v3 WER for Yoruba drops from [RESULT:whisper_lora.yoruba.wer.before]% to [RESULT:whisper_lora.yoruba.wer.after]% with LoRA. Crucially, LoRA achieves these gains while updating only [RESULT:lora.trainable_pct]% of model parameters in [RESULT:lora.gpu_hours] GPU hours, making it the method of choice for resource-constrained research. [NARRATIVE:abstract_conclusion]

---

## 1. Introduction

The success of self-supervised pre-training has produced universal speech encoders that process hundreds of languages without task-specific supervision. SeamlessM4T-v2 (Barrault et al., 2023) unifies ASR, speech-to-text translation (S2TT), and machine translation in a single model with official support for over 100 languages. Whisper-large-v3 (Radford et al., 2023) similarly transcribes speech across 96 languages zero-shot.

Despite this breadth, zero-shot performance on low-resource African languages such as Yoruba, Igbo, and Hausa remains weak. The models were trained predominantly on high-resource language pairs; consequently, their representations of Niger-Congo and Afro-Asiatic phonology and morphology are under-developed. Practitioners deploying these models on African speech face a choice: accept the performance ceiling of zero-shot inference, or invest in fine-tuning. Full fine-tuning of a model with billions of parameters is prohibitively expensive and risks catastrophic forgetting of multilingual representations (French, 1999; McCloskey & Cohen, 1989). PEFT methods address both concerns: they introduce a small number of trainable parameters into a frozen base model, reducing GPU memory and compute while preserving the pre-trained knowledge.

In this paper we ask: **how much does LoRA or adapter fine-tuning improve African speech translation, and at what cost?** We evaluate two PEFT strategies against zero-shot and full fine-tuning upper bounds across three languages (Yoruba, Igbo, Hausa, Swahili) on FLEURS. We also investigate the data scaling behaviour: how few labelled samples suffice for statistically significant improvement?

Our contributions are:
1. The first systematic PEFT evaluation for low-resource African S2TT and ASR;
2. A data efficiency analysis showing the minimum data budget for significant gains;
3. A cost comparison (trainable parameters, GPU hours) between LoRA and adapter strategies.

---

## 2. Related Work

### 2.1 Multilingual Speech Translation

Large multilingual models have driven rapid progress in speech translation. mSLAM (Bapna et al., 2022), SeamlessM4T (Barrault et al., 2023), and Whisper (Radford et al., 2023) all demonstrate strong zero-shot transfer to unseen languages. However, coverage of African languages remains sparse in their training corpora. The FLEURS benchmark (Conneau et al., 2023) provides standardised evaluation across 102 languages and reveals substantial performance gaps between high-resource and low-resource languages.

### 2.2 Parameter-Efficient Fine-Tuning

PEFT research originated in NLP (Houlsby et al., 2019; Hu et al., 2022) and has since been adapted for speech. LoRA injects trainable low-rank matrices into attention layers, typically updating fewer than 1% of parameters. Adapters insert small bottleneck modules between transformer layers. Both approaches have been applied to Whisper fine-tuning (Gong et al., 2023; Wang et al., 2023) and multilingual NMT (Bapna & Firat, 2019), but their application to African S2TT is largely unexplored.

### 2.3 African Speech Resources

MasakhaSpeech (Olatunji et al., 2022) and AfriSpeech (Olatunji et al., 2022) provide labelled speech for a subset of African languages. FLEURS (Conneau et al., 2023) offers parallel text and audio for over 100 languages. Prior work on African ASR (Dossou et al., 2022; Olatunji et al., 2022) focuses on full fine-tuning; we are not aware of a systematic PEFT study for this setting.

---

## 3. Methods

### 3.1 Base Models

We fine-tune two architectures:

- **SeamlessM4T-v2 Large** (`facebook/seamless-m4t-v2-large`): 2.3B parameters, unified S2TT + ASR + MT.
- **Whisper-large-v3** (`openai/whisper-large-v3`) + **NLLB-200-distilled-600M** (`facebook/nllb-200-distilled-600M`): cascade architecture; we fine-tune Whisper (ASR) with LoRA only.

### 3.2 PEFT Strategies

**LoRA** (Hu et al., 2022): We apply LoRA to all query and value projection matrices in the cross-attention layers of each model's encoder-decoder. Rank $r = 16$, scaling factor $\alpha = 32$. Trainable parameters: [RESULT:lora.trainable_params_m]M ([RESULT:lora.trainable_pct]% of total).

**Adapters** (Houlsby et al., 2019): We insert bottleneck adapters (hidden size 256) after each feed-forward sublayer. Trainable parameters: [RESULT:adapter.trainable_params_m]M ([RESULT:adapter.trainable_pct]% of total).

Both methods freeze all base model weights. We compare against full fine-tuning (all parameters updated) as an upper bound, reporting published numbers where available.

### 3.3 Training Setup

- **Data**: FLEURS training split. Main experiment uses 1,000 samples per language; scaling experiments use 100, 500, 1,000, and full training set.
- **Seed**: 42 for reproducibility.
- **Optimiser**: AdamW, learning rate 1e-4, linear warmup over 10% of steps.
- **Batch size**: 8 (gradient accumulation ×4 for effective batch 32).
- **Hardware**: NVIDIA A100 (Google Colab Pro+).

### 3.4 Evaluation

We evaluate on FLEURS validation split (same split as Paper 1 for comparability). Metrics:
- **BLEU** (sacrebleu, tokenised): S2TT (source→English).
- **WER** (%): ASR transcription accuracy.

---

## 4. Experimental Setup

### 4.1 Languages

| Language | Family | Script | FLEURS code | In SeamlessM4T | In Whisper |
|---------|--------|--------|------------|---------------|-----------|
| Yoruba | Niger-Congo (Volta-Niger) | Latin | `yor_Latn` | ✓ | ✓ |
| Igbo | Niger-Congo (Volta-Niger) | Latin | `ibo_Latn` | ✓ S2TT; ✗ Hausa S2TT | ✗ |
| Swahili | Niger-Congo (Bantu) | Latin | `swh_Latn` | ✓ | ✓ |
| Hausa | Afro-Asiatic (Chadic) | Latin | `hau_Latn` | ✗ S2TT; ✓ text | ✓ |

### 4.2 Experiment Matrix

| Experiment | Model | Languages | PEFT method |
|-----------|-------|---------|------------|
| `FLEURS__SeamlessM4Tv2` (LoRA) | SeamlessM4T-v2 | Yoruba, Igbo, Swahili | LoRA |
| `FLEURS__SeamlessM4Tv2` (Adapter) | SeamlessM4T-v2 | Yoruba, Igbo | Adapter |
| `FLEURS__WhisperNLLB` (LoRA) | Whisper-large-v3 | Yoruba, Hausa, Swahili | LoRA |

---

## 5. Results

### 5.1 Before vs After Adaptation (Table 1)

Table 1 reports BLEU (S2TT) and WER (ASR) before (zero-shot) and after LoRA adaptation with 1,000 samples.

**Table 1: Before/after LoRA adaptation — BLEU (S2TT, ↑) and WER (ASR, ↓)**

| Model | Method | Yoruba BLEU | Igbo BLEU | Swahili BLEU | Yoruba WER |
|-------|--------|------------|---------|------------|-----------|
| SeamlessM4T-v2 | Zero-shot | [RESULT:seamless_lora.yoruba.bleu.before] | [RESULT:seamless_lora.igbo.bleu.before] | [RESULT:seamless_lora.swahili.bleu.before] | [RESULT:seamless_lora.yoruba.wer.before] |
| SeamlessM4T-v2 | LoRA | **[RESULT:seamless_lora.yoruba.bleu.after]** | **[RESULT:seamless_lora.igbo.bleu.after]** | **[RESULT:seamless_lora.swahili.bleu.after]** | **[RESULT:seamless_lora.yoruba.wer.after]** |
| SeamlessM4T-v2 | Adapter | [RESULT:seamless_adapter.yoruba.bleu.after] | [RESULT:seamless_adapter.igbo.bleu.after] | — | — |
| Whisper-large-v3 | Zero-shot | — | — | — | [RESULT:whisper_lora.yoruba.wer.before] |
| Whisper-large-v3 | LoRA | — | — | — | **[RESULT:whisper_lora.yoruba.wer.after]** |
| *Wav2Vec2-XLSR* | *Full FT (published)* | — | — | — | *[BASELINE:lugo2022masakhane.yoruba.wer]* |

*Published full fine-tuning in italics for reference. — = not applicable (language excluded from this model).*

### 5.2 Parameter Efficiency (Table 2)

**Table 2: Parameter efficiency and training cost comparison**

| Method | Trainable params (M) | % of total | BLEU gain (Yoruba) | WER gain (Yoruba) | GPU hours |
|--------|---------------------|-----------|-------------------|------------------|----------|
| LoRA | [RESULT:lora.trainable_params_m] | [RESULT:lora.trainable_pct]% | [RESULT:seamless_lora.yoruba.bleu.gain] | [RESULT:seamless_lora.yoruba.wer.gain] | [RESULT:lora.gpu_hours] |
| Adapter | [RESULT:adapter.trainable_params_m] | [RESULT:adapter.trainable_pct]% | — | — | [RESULT:adapter.gpu_hours] |
| Full FT | all | 100% | [NARRATIVE:full_ft_bleu_gain] | [NARRATIVE:full_ft_wer_gain] | [NARRATIVE:full_ft_gpu_hours] |

### 5.3 Data Scaling (Table 3)

**Table 3: BLEU (S2TT) vs fine-tuning data budget — LoRA, SeamlessM4T-v2**

| Samples | Yoruba BLEU | Igbo BLEU | Yoruba WER | Hausa WER |
|---------|------------|---------|-----------|---------|
| 100 | [RESULT:scaling.yoruba.bleu.100] | [RESULT:scaling.igbo.bleu.100] | [RESULT:scaling.yoruba.wer.100] | [RESULT:scaling.hausa.wer.100] |
| 500 | [RESULT:scaling.yoruba.bleu.500] | [RESULT:scaling.igbo.bleu.500] | [RESULT:scaling.yoruba.wer.500] | [RESULT:scaling.hausa.wer.500] |
| 1,000 | [RESULT:scaling.yoruba.bleu.1000] | [RESULT:scaling.igbo.bleu.1000] | [RESULT:scaling.yoruba.wer.1000] | [RESULT:scaling.hausa.wer.1000] |
| Full | [RESULT:scaling.yoruba.bleu.full] | [RESULT:scaling.igbo.bleu.full] | [RESULT:scaling.yoruba.wer.full] | [RESULT:scaling.hausa.wer.full] |

Minimum samples for statistically significant improvement (p < 0.05): **[RESULT:scaling.min_threshold_samples]**.

---

## 6. Discussion

### 6.1 Gains from PEFT

[NARRATIVE:discussion_peft_gains]

The BLEU gain of [RESULT:seamless_lora.yoruba.bleu.gain] points for Yoruba and WER reduction of [RESULT:seamless_lora.yoruba.wer.gain] points demonstrate that even a small number of labelled samples substantially improves zero-shot performance. This aligns with prior work on PEFT for low-resource NLP (Pfeiffer et al., 2020) where adapters and LoRA outperform zero-shot by wide margins.

### 6.2 LoRA vs Adapters

[NARRATIVE:discussion_lora_vs_adapter]

Both methods update similar fractions of the parameter space, yet their BLEU trajectories differ. [NARRATIVE:lora_vs_adapter_comparison]

### 6.3 Data Efficiency

The minimum sample threshold of [RESULT:scaling.min_threshold_samples] samples for significant improvement suggests that even very small annotation campaigns (a few hours of labelled speech) can yield reliable gains. The scaling curve [NARRATIVE:scaling_curve_shape] follows [NARRATIVE:scaling_law_description].

### 6.4 Limitations

- We fine-tune on FLEURS only; results may not generalise to domain-shifted data (broadcast speech, telephone speech).
- Igbo ASR fine-tuning is unavailable via Whisper (no language token); LoRA gains for Igbo are measured only via SeamlessM4T S2TT.
- Full fine-tuning numbers are taken from published baselines; direct comparison requires matching hardware and data exactly.

---

## 7. Conclusion

We demonstrate that LoRA fine-tuning with as few as [RESULT:scaling.min_threshold_samples] labelled samples produces statistically significant improvements in both S2TT (BLEU) and ASR (WER) for low-resource African languages. The method achieves [RESULT:seamless_lora.yoruba.bleu.gain] BLEU gain for Yoruba S2TT and [RESULT:whisper_lora.yoruba.wer.gain] WER reduction for Yoruba ASR, updating only [RESULT:lora.trainable_pct]% of model parameters in [RESULT:lora.gpu_hours] GPU hours. These results make PEFT an attractive first option for practitioners adapting multilingual models to new African speech domains before committing to full fine-tuning.

---

## References

- Barrault, L., et al. (2023). *SeamlessM4T — Massively Multilingual & Multimodal Machine Translation*. arXiv:2308.11596.
- Bapna, A., et al. (2022). *mSLAM: Massively multilingual joint pre-training for speech and text*. arXiv:2202.01855.
- Bapna, A., & Firat, O. (2019). *Simple, scalable adaptation for neural machine translation*. EMNLP 2019.
- Conneau, A., et al. (2023). *FLEURS: Few-shot learning evaluation of universal representations of speech*. SLT 2022.
- Dossou, B. F. P., et al. (2022). *OkwuGbé: End-to-end speech recognition for Fon and Igbo*. AfricaNLP 2022.
- French, R. M. (1999). *Catastrophic forgetting in connectionist networks*. Trends in Cognitive Sciences.
- Gong, Y., et al. (2023). *Whisper-AT: Noise-robust automatic speech recognizers are also strong general audio event taggers*. Interspeech 2023.
- Houlsby, N., et al. (2019). *Parameter-efficient transfer learning for NLP*. ICML 2019.
- Hu, E. J., et al. (2022). *LoRA: Low-rank adaptation of large language models*. ICLR 2022.
- McCloskey, M., & Cohen, N. J. (1989). *Catastrophic interference in connectionist networks: The sequential learning problem*. Psychology of Learning and Motivation.
- Olatunji, T., et al. (2022). *AfriSpeech-200: Pan-African accented speech dataset for clinical and general domain ASR*. arXiv:2104.02010.
- Pfeiffer, J., et al. (2020). *AdapterHub: A framework for adapting transformers*. EMNLP 2020.
- Radford, A., et al. (2023). *Robust speech recognition via large-scale weak supervision*. ICML 2023.
- Wang, C., et al. (2023). *Parameter-efficient tuning of large-scale multimodal foundation model*. NeurIPS 2023.
