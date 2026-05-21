# Paper 4 — LinguoMT-Cascade: Experiment Guidelines

Cascade vs end-to-end architecture comparison. E2E and cascade zero-shot baselines come from Paper 1.

**Deliverables:** Table 1 (arch comparison), Table 2 (error propagation), Table 3 (oracle), Table 4 (break-even WER), Table 5 (latency), Figures 1–2, paper_outline.filled.md.

---

## Key files

| File | Purpose |
|------|---------|
| `papers/experiment_setup.yaml` | Model IDs, language codes — shared reference |
| `papers/paper4_cascade/config.yaml` | Error propagation, latency, ablation configuration |
| `papers/paper4_cascade/baselines.csv` | Cascade and E2E baselines (from Paper 1) |
| `papers/paper4_cascade/paper_outline.md` | Paper skeleton with `[RESULT:key]` placeholders |
| `papers/fill_results.py` | Fills placeholders from experiment output CSVs |

---

## Prerequisites

- [ ] Paper 1 results: SeamlessM4T-v2 E2E BLEU, Whisper WER, cascade BLEU (all languages)
- [ ] FLEURS validation gold transcripts (for oracle experiment)
- [ ] `PAPER_MODE = "cascade"` in run scripts

---

## Step 1 — Import Paper 1 baselines

Do not re-run the zero-shot experiments. Copy from Paper 1 outputs:
- SeamlessM4T-v2 E2E BLEU per language → `baselines.csv` with `architecture: end_to_end`
- Whisper+NLLB cascade BLEU per language → `baselines.csv` with `architecture: cascade`
- Whisper intermediate ASR WER per language

Sanity check: cascade BLEU must be ≤ oracle BLEU ≤ text-MT ceiling BLEU.

---

## Step 2 — Run oracle cascade

Feed FLEURS gold reference transcripts directly into NLLB-200 (bypass Whisper):

```python
# Set up in a standalone script or notebook:
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
nllb_tok   = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# For each (gold_transcript, language) in FLEURS validation:
#   nllb_tok.src_lang = lang_cfg["nllb_code"]
#   tgt_id = nllb_tok.convert_tokens_to_ids("eng_Latn")
#   translate gold_transcript → English
#   compute BLEU vs English reference
```

Language codes: Yoruba `yor_Latn`, Hausa `hau_Latn`, Igbo `ibo_Latn`, Swahili `swh_Latn`.

---

## Step 3 — Error propagation curve

Using the Whisper transcripts from Paper 1:

```python
import random

def corrupt_transcript(text, target_wer, seed=42):
    rng   = random.Random(seed)
    words = text.split()
    n_corrupt = int(len(words) * target_wer)
    idxs = rng.sample(range(len(words)), min(n_corrupt, len(words)))
    for i in idxs:
        words[i] = "<unk>"
    return " ".join(words)

# wer_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# For each rate: corrupt transcripts → NLLB → compute BLEU
# Record (wer_rate, bleu) pairs for Figure 2
```

Break-even WER = WER where cascade BLEU crosses E2E BLEU on the curve.

---

## Step 4 — Latency measurement

Measure on the same GPU. Warm up 10 samples before recording 100:

```python
import torch, time
torch.cuda.synchronize()
t0 = time.perf_counter()
for sample in test_samples[:100]:
    # run inference
    pass
torch.cuda.synchronize()
latency_ms = (time.perf_counter() - t0) * 1000 / 100
```

Report median and P95 latency, and peak VRAM (`torch.cuda.max_memory_allocated()`).

---

## Step 5 — NLLB model size ablation (optional)

Repeat the cascade with `facebook/nllb-200-1.3B` as the MT component (keep Whisper fixed).
Compare BLEU gain vs VRAM cost to show whether MT model size is worth investing in.

---

## Step 6 — Fill in paper_outline.md

```bash
python papers/fill_results.py paper4_cascade
```

Output: `papers/paper4_cascade/paper_outline.filled.md`

---

## Step 7 — Validate

```bash
python -c "
from framework.sota import load_and_validate_sota
load_and_validate_sota('papers/paper4_cascade/baselines.csv', 'papers/paper4_cascade/schema.json')
print('Schema OK')
"
```

---

## Reporting checklist

- [ ] Table 1: Cascade vs E2E vs oracle vs text ceiling × all languages
- [ ] Table 2: Error propagation decomposition (ASR contribution vs architecture gap)
- [ ] Table 3: Oracle cascade BLEU × all languages
- [ ] Table 4: Break-even WER per language
- [ ] Table 5: Latency (ms/sample) and VRAM (MB) per architecture
- [ ] Figure 1: WER vs BLEU scatter plot
- [ ] Figure 2: Error propagation curve (simulated WER vs cascade BLEU)
- [ ] `paper_outline.filled.md` generated

---

## Scope reminder

- Do NOT run PEFT here → Paper 2
- Do NOT vary audio conditions → Paper 3
- Do NOT analyse cross-lingual transfer → Paper 5
