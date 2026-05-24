# Paper 2 — LinguoMT-Adapt: Parameter-Efficient Fine-Tuning for African Speech Translation

---

## Title

**LinguoMT-Adapt: Parameter-Efficient Fine-Tuning for African Speech Translation**

---

## Abstract

Large pre-trained multilingual speech models achieve strong zero-shot performance across many languages, yet their results on low-resource African languages remain substantially below those on high-resource counterparts. Full fine-tuning to close this gap is computationally expensive and risks catastrophic forgetting of multilingual representations. This paper investigates parameter-efficient fine-tuning (PEFT) as a practical alternative, applying LoRA [hu2022lora] and lightweight adapter modules [bapna2019simple_adapters] to adapt pre-trained speech models to Yoruba, Hausa, and Igbo with limited target-language data. We compare PEFT strategies against zero-shot baselines from SeamlessM4T-v2 [seamlesscommunication2023seamless_yoruba] and full fine-tuning baselines from MasakhaSpeech [dossou2022masakhaspeech_hausa] and AfriSpeech-200 [olatunji2023afrispeech_hausa]. Results show that LoRA [hu2022lora] achieves competitive adaptation gains while training fewer than 1% of model parameters, offering a resource-efficient path for African-language speech technology deployment in low-compute settings.

---

## Chapter 1 — Introduction

### 1.1 Motivation

Community-driven efforts such as MasakhaSpeech [dossou2022masakhaspeech_hausa] and AfriSpeech-200 [olatunji2023afrispeech_hausa] have demonstrated that fine-tuning large pre-trained models on even modest amounts of African-language audio yields substantial improvements in ASR quality. However, these projects rely on full fine-tuning — updating all model parameters — which is computationally intensive, risks overwriting multilingual representations, and requires a separate model checkpoint per language. At the same time, zero-shot inference from large models such as SeamlessM4T-v2 [seamlesscommunication2023seamless_yoruba] still underperforms adapted systems by a wide margin on African languages. Parameter-efficient fine-tuning methods — LoRA [hu2022lora] and adapter modules [bapna2019simple_adapters] — offer a middle path: they freeze the bulk of the pre-trained model and inject small trainable components, drastically reducing compute and storage overhead while preserving the multilingual backbone. Whether these methods transfer effectively to African-language speech remains an open question.

### 1.2 Problem Statement

There is no systematic comparison of parameter-efficient fine-tuning strategies for African-language speech translation that controls for the number of trainable parameters, the amount of target-language data, and the evaluation protocol. Existing work either applies full fine-tuning [dossou2022masakhaspeech_hausa][olatunji2023afrispeech_hausa] or evaluates zero-shot models [seamlesscommunication2023seamless_yoruba], leaving the PEFT regime unexplored for Yoruba, Hausa, and Igbo. Without such a comparison, practitioners cannot make principled choices between LoRA [hu2022lora] and adapter [bapna2019simple_adapters] strategies, nor can they estimate the expected performance gain per additional hour of target-language audio.

### 1.3 Research Questions

**RQ1.** Does LoRA fine-tuning [hu2022lora] applied to SeamlessM4T-v2 [seamlesscommunication2023seamless_yoruba] yield significant improvements over zero-shot inference on ASR WER and S2TT BLEU for Yoruba, Hausa, and Igbo, when training with fewer than 10 hours of target-language audio?

**RQ2.** How does the adaptation gain from LoRA [hu2022lora] compare to the full fine-tuning gains reported in MasakhaSpeech [dossou2022masakhaspeech_hausa] and AfriSpeech-200 [olatunji2023afrispeech_hausa], and what fraction of the full fine-tuning gain is recovered at what fraction of the parameter budget?

**RQ3.** How does adaptation performance scale with the number of target-language training samples, and is there a minimum data threshold below which PEFT [hu2022lora][bapna2019simple_adapters] provides no measurable benefit over the zero-shot baseline [seamlesscommunication2023seamless_yoruba]?

---

**Paper mode:** `adaptation`  
**Schema file:** `schema.json` — read by the framework to validate data before generating comparison tables.  
**Experiment guidelines:** [GUIDELINES.md](GUIDELINES.md) — step-by-step workflow for running and reporting all experiments.

---

## Data sources in this folder

| File | Format | Purpose |
|------|--------|---------|
| `references.yaml` | YAML list | Bibliography + comparison metadata (preferred) |
| `sota_results.csv` | CSV | Tabular baselines — one row per result |

---

## Working with `references.yaml` (recommended)

Each entry is a bibliographic reference enriched with comparison fields.
Use one entry per paper × language combination.

**To activate a baseline in comparison tables: fill in `score`.**

### Entry format

```yaml
- citation_key: dossou2022masakhaspeech_hausa
  type: inproceedings
  author: "Dossou, Bonaventure F. P. and Emezue, Chris Chinenye and others"
  title: "MasakhaSpeech: End-to-End Speech Recognition for 5 African Languages"
  year: 2022
  booktitle: "Proceedings of Interspeech 2022"
  url: "https://arxiv.org/abs/2206.00253"

  # --- comparison fields ---
  model: "Wav2Vec2-XLSR (fine-tuned)"
  datasets: [MasakhaSpeech]                      # list — all datasets evaluated on
  language: Hausa
  directions: [ASR]                              # list — all task directions reported
  metrics: [WER]                                 # list — score corresponds to metrics[0]
  score: null                                    # fill in WER from Table 2
  ft_method: full_finetune                       # paper-specific required field
  summary: >
    Fine-tunes Wav2Vec2-XLSR on 5 African languages (~10h each). Direct adaptation baseline
    for Hausa ASR fine-tuning. Shows what full fine-tuning achieves with limited data.
  notes: "Fill in Hausa WER from Table 2 of arXiv:2206.00253"
```

### Reference fields

#### Standard bibliographic

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `citation_key` | ★ Required | str | Unique identifier, one per paper × language |
| `type` | ★ Required | str | `article`, `inproceedings`, `misc` |
| `author` | ★ Required | str | Full author list as a single string |
| `title` | ★ Required | str | Full paper title |
| `year` | ★ Required | int | 4-digit publication year |
| `journal` | ○ Optional | str | Journal name |
| `booktitle` | ○ Optional | str | Conference name |
| `url` | ○ Optional | str | Link to the paper |

#### Comparison

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `model` | ★ Required | str | Base model + adaptation method, e.g. `Whisper-large-v3 + LoRA` |
| `datasets` | ★ Required | list | All datasets the paper evaluates on |
| `language` | ★ Required | str | Target language — `Yoruba`, `Hausa`, `Igbo` |
| `directions` | ★ Required | list | All task directions, e.g. `[ASR]` or `["Source → English"]` |
| `metrics` | ★ Required | list | All metrics reported; `score` corresponds to `metrics[0]` |
| `score` | ★ Required | float\|null | Numeric score for `metrics[0]`; `null` = not yet filled |
| `summary` | ★ Required | str | 2–5 sentences on what the paper does and why it is relevant |
| `ft_method` | ★ Required | str | Adaptation method: `lora`, `adapter`, `full_finetune`, `prefix`, `none` |
| `notes` | ○ Optional | str | Where to find the score, training conditions |
| `pretrained_score` | ○ Optional | float | Score **before** adaptation — enables before/after comparison |
| `num_ft_samples` | ○ Optional | int | Number of training samples used for adaptation |
| `trainable_params_pct` | ○ Optional | float | % of parameters trained, e.g. `0.5` for 0.5% |

> **Tip:** If you have both pre- and post-adaptation scores from the same paper, fill in
> `pretrained_score` alongside `score`. The framework builds a before/after table automatically.

> **Zero-shot baselines** (no fine-tuning, e.g. SeamlessM4T pre-trained only): set `ft_method: none`.

---

## Working with `sota_results.csv`

One row per system × language × metric. Flat fields — no lists.

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `paper_title` | ★ Required | str | Full paper title |
| `authors` | ★ Required | str | First author + et al. |
| `year` | ★ Required | int | 4-digit year |
| `model` | ★ Required | str | Base model + adaptation method |
| `dataset` | ★ Required | str | Dataset name |
| `language` | ★ Required | str | Display language name |
| `direction` | ★ Required | str | `Source → English` or `English → Source` |
| `metric` | ★ Required | str | `BLEU`, `spBLEU`, `ChrF`, `WER`, or `CER` |
| `score` | ★ Required | float | Score after adaptation |
| `citation_key` | ★ Required | str | Key matching an entry in `references.yaml` |
| `ft_method` | ★ Required | str | Adaptation method |
| `pretrained_score` | ○ Optional | float | Score before adaptation |
| `num_ft_samples` | ○ Optional | int | Training samples used |
| `trainable_params_pct` | ○ Optional | float | % of parameters trained |
| `training_hours` | ○ Optional | float | GPU hours |
| `notes` | ○ Optional | str | Training conditions |

---

## Published baselines to fill in

### Zero-shot baselines (no fine-tuning)

| Model | Dataset | Languages | Metric | Where to find scores |
|-------|---------|-----------|--------|----------------------|
| SeamlessM4T-v2-large | FLEURS | Yoruba, Hausa | BLEU | Table B.1 of `arXiv:2312.05187` |

### ASR fine-tuning baselines

| Model/Method | Dataset | Languages | Metric | Where to find scores |
|--------------|---------|-----------|--------|----------------------|
| Wav2Vec2-XLSR (full FT) | MasakhaSpeech | Hausa, Yoruba | WER | Table 2 of `arXiv:2206.00253` |
| Whisper-large-v2 (full FT) | AfriSpeech-200 | African-accented English | WER | Table 4 of `arXiv:2104.02010` |

### Methodology references (cite, not compare)

| Paper | Purpose |
|-------|---------|
| LoRA — Hu et al. 2022 (`arXiv:2106.09685`) | LoRA adaptation method |
| Adapter — Bapna & Firat 2019 (`arXiv:1909.08478`) | Adapter method baseline |

---

## Usage in run script

```python
PAPER_MODE        = "adaptation"
ENABLE_FINETUNING = True
SOTA_FILE         = "sota/paper2_adaptation/references.yaml"
```

---

## Experiment Steps

**Research question:** How much does LoRA fine-tuning improve zero-shot baselines? How does adaptation efficiency scale with training data size?

### Which experiments to run

FLEURS only — fine-tuning requires a training split and African-Celtic is the evaluation domain, not the training domain:

| Experiment | Model | Dataset | Languages |
|---|---|---|---|
| `FLEURS__SeamlessM4Tv2` | SeamlessM4T-v2 + LoRA | FLEURS | Igbo, Yoruba, Swahili |
| `FLEURS__WhisperNLLB` | Whisper + NLLB + LoRA | FLEURS | Yoruba, Hausa, Swahili |

> African-Celtic can be added as a supplementary cross-domain evaluation after the main FLEURS run.

### Phase 1 — One-time setup (do once, reuse across all sessions)

Open `run_on_colab.ipynb` on Google Colab.

1. **Set runtime** → Runtime → Change runtime type → **T4 GPU**
2. **Run Step 1** — mount Google Drive
3. **Run Step 2** — clone the repository
4. **Run Step 3** — install dependencies (~3–5 min)
5. **Edit & run Step 4** — set cache paths:
   ```python
   EXPERIMENTS       = ["FLEURS__SeamlessM4Tv2", "FLEURS__WhisperNLLB"]
   HF_CACHE_DIR      = "/content/drive/MyDrive/LinguoMT-AfricaS2T/hf_cache"
   DATASET_CACHE_DIR = "/content/drive/MyDrive/LinguoMT-AfricaS2T/dataset_cache"
   MAX_CACHED_PAIRS  = 300
   ```
6. **Run Step 5** — download models to Drive (~45–60 min first time, < 1 min after)
7. **Run Step 6** — build dataset sample cache (~10–15 min first time, < 5 sec after)

### Phase 2 — Smoke test (always do this before the full run)

8. **Edit & run Step 7** — debug settings:
   ```python
   PAPER_MODE         = "adaptation"
   DEBUG_MODE         = True
   ENABLE_FINETUNING  = True
   FINETUNING_METHOD  = "lora"
   SCALING_BUDGETS    = [50]        # single small budget to confirm fine-tuning runs
   N_EVAL_RUNS        = 1
   EVAL_TEXT_SAMPLES  = [8]
   EVAL_AUDIO_SAMPLES = [3]
   SOTA_FILE          = ""
   FORCE_RERUN        = False
   ```
9. **Run Step 8** — debug run (~45–60 min; fine-tuning adds time even in debug mode)
10. **Run Steps 9, 10, 11** — verify before/after tables appear in the report.

### Phase 3 — Full paper run

11. **Edit & run Step 7** — full settings:
    ```python
    PAPER_MODE         = "adaptation"
    DEBUG_MODE         = False
    ENABLE_FINETUNING  = True
    FINETUNING_METHOD  = "lora"          # lora (T4 compatible) | adapter | full (A100 only)
    SCALING_BUDGETS    = [100, 500, 1000, 0]   # 0 = full training split
    N_EVAL_RUNS        = 3
    EVAL_TEXT_SAMPLES  = [100, 200, 300]
    EVAL_AUDIO_SAMPLES = [30,  75,  100]
    SOTA_FILE          = "sota/paper2_adaptation/sota_results.csv"
    FORCE_RERUN        = False
    ```
12. **Run Step 8** — full experiments (~4–6 h on T4; each scaling budget is a full fine-tuning pass)
13. **Run Step 9** — consolidate metrics
14. **Run Step 10** — generate results report
15. **Run Step 11** — download ZIP

> **Fine-tuning method note:** Use `"lora"` on T4 (free Colab). Switch to `"full"` only on Colab Pro A100 (~24 GB VRAM required). `"adapter"` is an intermediate option if LoRA is unstable.

### Outputs for the paper

| File in ZIP | Paper section |
|---|---|
| `consolidated_metrics/text_metrics.csv` | Before/after BLEU comparison table |
| `consolidated_metrics/asr_metrics.csv` | Before/after WER comparison table |
| `*/tables/scaling_curve*.md` | Data scaling learning curves |
| `*/tables/adaptation_summary*.md` | Adaptation gain per language |
| `*/plots/scaling_curve*.png` | Learning curve figures |
| `papers/adaptation/results_report.md` | Full discussion, efficiency analysis |

### If the session disconnects mid-run

Fine-tuning checkpoints are saved to Drive. On reconnect:

1. Re-run Steps 1–4
2. Skip Steps 5 and 6
3. In Step 7, comment out completed experiments
4. Set `FORCE_RERUN = False` — the framework will resume from saved checkpoints
5. Re-run Steps 8–11
