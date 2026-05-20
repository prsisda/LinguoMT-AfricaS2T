# Paper 4 — Experiment Guidelines: LinguoMT-Cascade

Step-by-step workflow for running and reporting the cascade vs end-to-end architecture comparison.

<!-- TOC -->
- [Paper 4 — Experiment Guidelines: LinguoMT-Cascade](#paper-4-experiment-guidelines-linguomt-cascade)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Step 1 — Verify and import Paper 1 cascade numbers](#step-1-verify-and-import-paper-1-cascade-numbers)
  - [Step 2 — Run the oracle ASR cascade (Table 3)](#step-2-run-the-oracle-asr-cascade-table-3)
  - [Step 3 — Error propagation analysis (Table 2, Figure 2)](#step-3-error-propagation-analysis-table-2-figure-2)
  - [Step 4 — WER threshold analysis (Table 4)](#step-4-wer-threshold-analysis-table-4)
  - [Step 5 — Latency measurement (Table 5)](#step-5-latency-measurement-table-5)
  - [Step 6 — Subcomponent ablation (supporting analysis)](#step-6-subcomponent-ablation-supporting-analysis)
  - [Step 7 — Generate tables and figures](#step-7-generate-tables-and-figures)
  - [Reporting checklist](#reporting-checklist)
  - [Scope reminder](#scope-reminder)
<!-- /TOC -->

---

## Overview

This paper asks: does the Whisper + NLLB-200 cascade match end-to-end SeamlessM4T-v2, and when does each architecture win? The E2E baselines come from Paper 1. The cascade zero-shot BLEU from Paper 1 Step 4 is your starting number — this paper deepens the analysis with error propagation, oracle experiments, and latency measurement.

**Deliverables:** Table 1 (cascade vs E2E vs text ceiling), Table 2 (error propagation), Table 3 (oracle ASR cascade), Table 4 (WER threshold analysis), Table 5 (latency), Figure 1 (WER vs BLEU scatter), Figure 2 (error propagation curve).

---

## Prerequisites

- [ ] Paper 1 results available: SeamlessM4T-v2 E2E BLEU, Whisper WER, cascade BLEU (all three languages)
- [ ] FLEURS test split + gold transcripts (for oracle experiment)
- [ ] `PAPER_MODE = "cascade"` in your run script
- [ ] Timing infrastructure: use Python `time.perf_counter` or `torch.cuda.Event` for latency

---

## Step 1 — Verify and import Paper 1 cascade numbers

Do not re-run the zero-shot cascade from scratch. Import the numbers recorded in Paper 1, Step 4. Fill them into `references.yaml` with `architecture: cascade`.

Verify the numbers are consistent:
- Cascade BLEU should be ≤ text-MT ceiling BLEU (NLLB-200 on gold transcripts)
- If cascade BLEU > ceiling, something is wrong with the gold transcript evaluation — recheck

Fill in:
- SeamlessM4T-v2-large E2E BLEU (`architecture: end_to_end`) from Paper 1
- Whisper+NLLB cascade BLEU (`architecture: cascade`) from Paper 1
- NLLB-200 text-MT ceiling BLEU on Flores-200 (`architecture: cascade`, note it's text-only)

---

## Step 2 — Run the oracle ASR cascade (Table 3)

Feed the **gold FLEURS reference transcripts** directly into NLLB-200, bypassing Whisper entirely. This isolates translation quality from ASR errors and gives the theoretical ceiling for any cascade that uses NLLB-200 as its MT component.

```python
PAPER_MODE      = "cascade"
MODEL_ID        = "facebook/nllb-200-distilled-600M"
INPUT_TYPE      = "gold_transcript"   # use reference text, not audio
DATASET_ID      = "google/fleurs"
```

Record oracle BLEU per language. The gap between oracle BLEU and actual cascade BLEU is attributable purely to ASR error propagation.

---

## Step 3 — Error propagation analysis (Table 2, Figure 2)

Measure how much of the BLEU gap between the cascade and the oracle is caused by ASR errors. Use the following decomposition:

```
ASR error contribution = Oracle BLEU − Cascade BLEU
Translation noise      = Text-ceiling BLEU − Oracle BLEU  (should be ≈ 0 if same dataset)
Architecture gap       = E2E BLEU − Cascade BLEU
```

To build the error propagation curve (Figure 2):
1. Take the Whisper transcripts generated during Paper 1, Step 4
2. Introduce controlled WER by randomly substituting words in the transcripts at rates: 10 %, 20 %, 30 %, 40 %, 50 %
3. Pass each degraded transcript set through NLLB-200
4. Plot x-axis = simulated WER, y-axis = resulting cascade BLEU

```python
def corrupt_transcript(text, target_wer, seed=42):
    # randomly replace `target_wer` fraction of words with "<unk>"
    ...
```

Report the WER level at which cascade BLEU falls below E2E BLEU — this is the "break-even WER".

---

## Step 4 — WER threshold analysis (Table 4)

Using the error propagation curve from Step 3, compute:

| Language | Break-even WER | Cascade wins when ASR WER < | E2E wins when ASR WER > |
|----------|---------------|----------------------------|------------------------|
| Yoruba | | | |
| Hausa | | | |
| Igbo | | | |

This is the actionable finding of the paper: it tells practitioners when to invest in ASR quality vs switching to an E2E model.

---

## Step 5 — Latency measurement (Table 5)

Measure wall-clock inference time for a fixed batch of 100 FLEURS test samples.

**Cascade:** time Whisper + NLLB-200 separately and jointly  
**E2E:** time SeamlessM4T-v2

```python
import torch, time

torch.cuda.synchronize()
t0 = time.perf_counter()
# run inference
torch.cuda.synchronize()
latency_ms = (time.perf_counter() - t0) * 1000 / n_samples
```

Report:
- Median latency per sample (ms)
- 95th-percentile latency per sample (ms)
- GPU VRAM peak (MB) for each architecture

Run on the same GPU. Warm up with 10 samples before recording. Record `latency_ms` field in `sota_results.csv`.

---

## Step 6 — Subcomponent ablation (supporting analysis)

Swap NLLB-200 model sizes to show how the MT component quality affects cascade BLEU:

| MT component | BLEU (Yoruba) | BLEU (Hausa) | BLEU (Igbo) |
|-------------|--------------|-------------|------------|
| NLLB-200-distilled-600M | | | |
| NLLB-200-1.3B | | | |

Keep ASR (Whisper-large-v3) fixed. This shows whether investing in a larger MT model is more efficient than investing in ASR quality.

---

## Step 7 — Generate tables and figures

```bash
python run_cascade.py --output-only
```

Verify:
- Table 1 shows cascade ≤ oracle ≤ text-ceiling (if any order is violated, recheck scoring)
- Figure 2 error propagation curve is monotonically decreasing
- Break-even WER values are plausible (typically 30–60 % WER for African languages)

---

## Reporting checklist

- [ ] Table 1: Cascade vs E2E vs oracle vs text-MT ceiling × 3 languages
- [ ] Table 2: Error propagation decomposition (ASR contribution vs translation noise)
- [ ] Table 3: Oracle ASR cascade BLEU × 3 languages
- [ ] Table 4: WER threshold / break-even analysis × 3 languages
- [ ] Table 5: Latency comparison (cascade vs E2E) with VRAM usage
- [ ] Figure 1: WER vs BLEU scatter plot across languages and conditions
- [ ] Figure 2: Simulated WER vs cascade BLEU (error propagation curve)
- [ ] `architecture` field set for every entry in `references.yaml`
- [ ] `asr_wer` filled for cascade entries
- [ ] `latency_ms` filled in `sota_results.csv`
- [ ] `paper_references.csv` updated

---

## Scope reminder

Do **not** run PEFT fine-tuning here — that is Paper 2. The cascade and E2E models are used at their zero-shot or Paper-1-baseline quality.  
Do **not** vary audio conditions (noise, VAD) here — that is Paper 3.  
Do **not** analyse cross-lingual transfer or typological distance here — that is Paper 5.
