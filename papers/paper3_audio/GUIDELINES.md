# Paper 3 — Experiment Guidelines: LinguoMT-Audio

Step-by-step workflow for running and reporting all audio preprocessing and robustness experiments.

<!-- TOC -->
- [Paper 3 — Experiment Guidelines: LinguoMT-Audio](#paper-3-experiment-guidelines-linguomt-audio)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Step 1 — Record clean-audio baselines](#step-1-record-clean-audio-baselines)
  - [Step 2 — Generate noisy audio variants](#step-2-generate-noisy-audio-variants)
  - [Step 3 — Measure degradation under noise (Table 1, Figure 1)](#step-3-measure-degradation-under-noise-table-1-figure-1)
  - [Step 4 — Apply VAD and evaluate (preprocessing ablation)](#step-4-apply-vad-and-evaluate-preprocessing-ablation)
  - [Step 5 — Apply loudness normalisation and evaluate](#step-5-apply-loudness-normalisation-and-evaluate)
  - [Step 6 — Combined preprocessing pipeline](#step-6-combined-preprocessing-pipeline)
  - [Step 7 — SpecAugment robustness training (Table 3)](#step-7-specaugment-robustness-training-table-3)
  - [Step 8 — Generate tables and figures](#step-8-generate-tables-and-figures)
  - [Reporting checklist](#reporting-checklist)
  - [Scope reminder](#scope-reminder)
<!-- /TOC -->

---

## Overview

This paper asks: how much does acoustic degradation hurt ASR and S2TT on African languages, and which preprocessing steps recover the most performance? The clean-audio baselines come from Paper 1 — do not re-run them, cite them. Your job is to introduce noise, measure degradation, and then measure how much each preprocessing step recovers.

**Deliverables:** Table 1 (SNR degradation), Table 2 (preprocessing ablation), Table 3 (augmentation robustness), Figure 1 (WER/BLEU vs SNR), Figure 2 (preprocessing ablation bar chart).

---

## Prerequisites

- [ ] Paper 1 clean-audio baselines available (Whisper WER, SeamlessM4T-v2 BLEU on FLEURS)
- [ ] `sox` or `librosa` available for audio manipulation
- [ ] `audiomentations` or equivalent for noise augmentation
- [ ] FLEURS test split audio files accessible (16 kHz WAV)
- [ ] `PAPER_MODE = "audio"` in your run script

---

## Step 1 — Record clean-audio baselines

Copy scores from Paper 1 into `references.yaml` as entries with `audio_condition: clean`. These are your degradation baseline.

| Entry | Score source |
|-------|-------------|
| Whisper-large-v3 WER, Yoruba (clean) | Paper 1, Table 1 |
| Whisper-large-v3 WER, Hausa (clean) | Paper 1, Table 1 |
| SeamlessM4T-v2 BLEU, Yoruba (clean) | Paper 1, Table 2 |
| SeamlessM4T-v2 BLEU, Hausa (clean) | Paper 1, Table 2 |

Do not include Igbo for Whisper (unsupported). Use SeamlessM4T-v2 for Igbo's clean baseline.

---

## Step 2 — Generate noisy audio variants

Add additive noise to the FLEURS test split at six SNR levels. Use babble noise as the primary noise type (most realistic for African-language field recordings), with white noise as a secondary condition.

**SNR levels (dB):** 20, 15, 10, 5, 0, −5  
**Noise types:** `babble` (primary), `white` (secondary)

```python
import soundfile as sf
import numpy as np

def add_noise(signal, noise, snr_db):
    sig_power  = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
    return signal + scale * noise
```

Save noisy files to a separate directory — do not overwrite originals:
```
data/noisy/babble/snr_20/<original_filename>.wav
data/noisy/babble/snr_10/<original_filename>.wav
...
```

Record in `sota_results.csv` or `references.yaml`: `snr_db`, `noise_type`, `audio_condition: noisy`.

---

## Step 3 — Measure degradation under noise (Table 1, Figure 1)

Run Whisper-large-v3 (ASR) and SeamlessM4T-v2 (S2TT) on all noisy variants. Two models × 6 SNR levels × 2 noise types × 2 languages = 48 evaluation runs.

```python
PAPER_MODE      = "audio"
AUDIO_CONDITION = "noisy"
SNR_DB          = 10       # repeat for each level
NOISE_TYPE      = "babble"
ENABLE_FINETUNING = False
```

Record for each condition:
- WER (Whisper) and BLEU (SeamlessM4T-v2) per language
- `snr_db` and `noise_type` fields in the result entry

Plot Figure 1: x-axis = SNR (dB), y-axis = WER or BLEU, one line per language.

---

## Step 4 — Apply VAD and evaluate (preprocessing ablation)

Apply voice activity detection to strip silence before inference. Use the Silero VAD model or webrtcvad.

```python
AUDIO_CONDITION = "vad_filtered"
VAD_APPLIED     = True
```

Run Whisper and SeamlessM4T-v2 on VAD-filtered versions of the **noisy** audio at SNR 5 dB (the hardest realistic condition). Compare WER/BLEU against the no-preprocessing noisy result from Step 3.

Record: `audio_condition: vad_filtered`, `vad_applied: true`, `snr_db: 5`.

---

## Step 5 — Apply loudness normalisation and evaluate

Normalise audio to −23 LUFS (EBU R128 standard) before inference.

```bash
ffmpeg -i input.wav -af loudnorm=I=-23:TP=-1.5:LRA=11 output_norm.wav
```

Run on the same SNR 5 dB noisy audio used in Step 4.

Record: `audio_condition: normalized`, `snr_db: 5`.

---

## Step 6 — Combined preprocessing pipeline

Run VAD + normalisation together as a single pipeline on SNR 5 dB audio. This is the "full preprocessing" condition.

Record: `audio_condition: normalized`, `vad_applied: true`, `snr_db: 5`.

Build Table 2 from Steps 4–6:

| Condition | WER Yoruba | WER Hausa | BLEU Yoruba | BLEU Hausa |
|-----------|-----------|-----------|-------------|------------|
| Noisy (no preprocessing) | | | | |
| VAD only | | | | |
| Normalisation only | | | | |
| VAD + Normalisation | | | | |

---

## Step 7 — SpecAugment robustness training (Table 3)

Fine-tune Whisper-large-v3 on FLEURS **train** split with SpecAugment augmentation, then evaluate on the noisy test set.

This is the only fine-tuning in this paper. Its purpose is to test whether augmentation-trained models are more robust to noise — not to improve clean-audio performance.

```python
ENABLE_FINETUNING  = True
FINETUNING_METHOD  = "full"     # full fine-tune with augmentation
SPECAUGMENT        = True
FREQ_MASK_PARAM    = 27
TIME_MASK_PARAM    = 100
NUM_MASKS          = 2
```

Evaluate the augmentation-trained model at SNR 5 and SNR 10 dB. Report:
- WER on noisy audio (augmented-trained model)
- WER on noisy audio (original model, from Step 3)
- WER on clean audio (check for regression against Paper 1 baseline)

**Important:** the clean-audio WER of the augmented model must not degrade more than 2 WER points vs the Paper 1 baseline. If it does, reduce augmentation strength.

---

## Step 8 — Generate tables and figures

```bash
python run_audio.py --output-only
```

Verify:
- Figure 1: SNR degradation curves look monotonically worse as SNR decreases
- Table 2: preprocessing ablation shows clear ranking across conditions
- Table 3: augmentation-trained model shows reduced degradation on noisy audio

---

## Reporting checklist

- [ ] Table 1: WER and BLEU at each SNR level, for both models and both noise types
- [ ] Table 2: Preprocessing ablation (VAD, norm, combined) at SNR 5 dB
- [ ] Table 3: Augmentation robustness — augmented vs original model on noisy audio
- [ ] Figure 1: SNR vs WER/BLEU degradation curves
- [ ] Figure 2: Preprocessing ablation bar chart
- [ ] All `audio_condition`, `snr_db`, `vad_applied` fields filled in `references.yaml`
- [ ] Clean-audio regression check passed (Step 7)
- [ ] `paper_references.csv` updated

---

## Scope reminder

Do **not** run PEFT fine-tuning experiments here — augmentation training (Step 7) is a robustness probe only. The PEFT efficiency question is owned by Paper 2.  
Do **not** compare cascade vs E2E architectures here — that is Paper 4.  
Do **not** vary languages beyond Yoruba and Hausa for noise experiments — Igbo is excluded because Whisper does not support it, and SeamlessM4T-v2 alone is insufficient for a full robustness study without ASR.
