# Paper 1 — LinguoMT Benchmark: Experiment Guidelines

Zero-shot evaluation of pretrained models. No fine-tuning. No audio augmentation.

**Deliverables:** Table 1 (ASR WER), Table 2 (S2TT BLEU), Table 3 (African-Celtic), Table 4 (gap analysis), Table 5 (metric correlation), paper_outline.filled.md.

---

## Key files

| File | Purpose |
|------|---------|
| `papers/experiment_setup.yaml` | All model IDs, language codes, dataset splits — single source of truth |
| `papers/paper1_benchmark/config.yaml` | Paper-specific run settings, expected tables, result key mapping |
| `papers/paper1_benchmark/baselines.csv` | Published SOTA scores (read by scripts via `SOTA_FILE`) |
| `papers/paper1_benchmark/paper_outline.md` | Paper skeleton with `[RESULT:key]` placeholders |
| `papers/fill_results.py` | Fills placeholders from experiment output CSVs |

---

## Prerequisites

- [ ] GPU with ≥ 16 GB VRAM
- [ ] Python environment: `transformers`, `datasets`, `sacrebleu`, `jiwer`
- [ ] `PAPER_MODE = "benchmark"` and `ENABLE_FINETUNING = False` in run scripts
- [ ] `SOTA_FILE = "papers/paper1_benchmark/baselines.csv"` in run scripts

---

## Step 1 — Fill in published SOTA baselines

Open `papers/paper1_benchmark/baselines.csv`. Rows with an empty `score` column need to be populated manually from the papers listed in the `notes` column. Rows already marked **VERIFIED** have confirmed values.

**VERIFIED scores already in baselines.csv:**
- NLLB-200-distilled-600M chrF++ on Flores-200:
  - Yoruba → English: **39.9** (source: `dl.fbaipublicfiles.com/…/metrics.csv`)
  - Hausa → English: **52.6** (same source)

**Scores still needed — fetch from these exact locations:**

| Row | Where to find |
|-----|---------------|
| SeamlessM4T-v2 BLEU, Yoruba/Igbo/Swahili | Table B.1 of [arXiv:2312.05187](https://arxiv.org/abs/2312.05187) |
| Whisper-large-v3 WER, Yoruba/Hausa/Swahili | [HuggingFace model card](https://huggingface.co/openai/whisper-large-v3) — FLEURS ASR eval table |
| NLLB spBLEU, Yoruba/Hausa/Igbo | Table 2 of [arXiv:2207.04672](https://arxiv.org/abs/2207.04672) |
| mSLAM-CTC BLEU, Yoruba | Table 3 of [arXiv:2205.12446](https://arxiv.org/abs/2205.12446) |
| mSLAM BLEU, Yoruba/Hausa | Supplementary S2TT table of [arXiv:2202.01374](https://arxiv.org/abs/2202.01374) |
| MMS-300M WER, Yoruba/Hausa/Igbo | Appendix of [arXiv:2305.13516](https://arxiv.org/abs/2305.13516) |
| XLS-R-1B WER, Yoruba/Hausa | Table 4 of [arXiv:2111.09296](https://arxiv.org/abs/2111.09296) |

Validate after filling:
```bash
python -c "
import csv
with open('papers/paper1_benchmark/baselines.csv') as f:
    rows = list(csv.DictReader(f))
filled = [r for r in rows if r.get('score')]
print(f'{len(filled)}/{len(rows)} baselines have scores')
"
```

---

## Step 2 — Configure the run scripts

In each `run_experiment.py`, set:

```python
DEBUG_MODE        = False
PAPER_MODE        = "benchmark"
ENABLE_FINETUNING = False
SOTA_FILE         = "papers/paper1_benchmark/baselines.csv"
```

Language settings are already correct in each script (from `experiment_setup.yaml`):
- `FLEURS__SeamlessM4Tv2`:      `MANUAL_LANGUAGES = ["igbo", "yoruba", "swahili"]`
- `FLEURS__WhisperNLLB`:        `MANUAL_LANGUAGES = ["yoruba", "hausa", "swahili"]`
- `AfricanCeltic__SeamlessM4Tv2`: `MANUAL_LANGUAGES = ["igbo", "yoruba"]`
- `AfricanCeltic__WhisperNLLB`:   `MANUAL_LANGUAGES = ["yoruba", "hausa"]`

---

## Step 3 — Run all four experiments

Run each script. On Colab, use the central `run_on_colab.ipynb` with `PAPER_MODE = "benchmark"`.
Locally:

```bash
python FLEURS__SeamlessM4Tv2/notebooks/run_experiment.py
python FLEURS__WhisperNLLB/notebooks/run_experiment.py
python AfricanCeltic__SeamlessM4Tv2/notebooks/run_experiment.py
python AfricanCeltic__WhisperNLLB/notebooks/run_experiment.py
```

Each run writes results to `<experiment>/outputs/<timestamp>_benchmark_full/`.

---

## Step 4 — Run high-resource gap analysis

Repeat Step 3 for three European reference languages to build Table 4. Override `MANUAL_LANGUAGES` temporarily:

```python
# French, German, Spanish — same datasets, same scripts
MANUAL_LANGUAGES = ["french", "german", "spanish"]   # check language key names in experiment_setup.yaml
```

Compute:
- Relative WER gap = (WER_african − WER_reference) / WER_reference × 100
- Absolute BLEU gap = BLEU_reference − BLEU_african

---

## Step 5 — Run metric sensitivity check

Reuse the SeamlessM4T-v2 output from Step 3 — no new inference needed. Re-score with all three metrics:

```bash
python -c "
import sacrebleu, scipy.stats
# load hypotheses and references from outputs/*/predictions/
# bleu_scores, spbleu_scores, chrf_scores = [list of per-language scores]
tau, _ = scipy.stats.kendalltau(bleu_scores, spbleu_scores)
print(f'BLEU vs spBLEU Kendall tau: {tau:.3f}')
"
```

---

## Step 6 — Fill in paper_outline.md

```bash
python papers/fill_results.py paper1_benchmark
```

This reads the latest output CSVs from each experiment folder and replaces all `[RESULT:key]` placeholders. Output: `papers/paper1_benchmark/paper_outline.filled.md`.

Review the filled document. Fill `[NARRATIVE:key]` sections manually.

---

## Step 7 — Validate

```bash
python -c "
from framework.sota import load_and_validate_sota
load_and_validate_sota('papers/paper1_benchmark/baselines.csv', 'papers/paper1_benchmark/schema.json')
print('Schema OK')
"
```

---

## Reporting checklist

- [ ] `baselines.csv` — all non-null scores filled and verified
- [ ] Table 1: ASR WER for all model × language combinations (note unsupported pairs)
- [ ] Table 2: S2TT BLEU, both datasets, both models
- [ ] Table 3: African-Celtic results
- [ ] Table 4: High-resource gap (compare African vs French/German/Spanish)
- [ ] Table 5: Kendall τ between BLEU, spBLEU, chrF
- [ ] `paper_outline.filled.md` generated with no remaining `[RESULT:*]` placeholders
- [ ] All `[NARRATIVE:*]` sections filled manually

---

## What feeds into other papers

| Output | Used by |
|--------|---------|
| SeamlessM4T-v2 zero-shot BLEU (all languages) | Papers 2, 4, 5 |
| Whisper-large-v3 zero-shot WER (Yoruba, Hausa) | Papers 3, 4, 5 |
| Cascade BLEU + intermediate WER | Papers 3, 4 |
| High-resource BLEU/WER | All papers (framing) |
