# Paper 4 — LinguoMT-Cascade: Cascade vs End-to-End Architectures for African Speech Translation

---

## Title

**LinguoMT-Cascade: Cascade vs End-to-End Architectures for African Speech Translation**

---

## Abstract

Speech translation can be achieved through two broad architectural paradigms: cascade systems that chain an ASR module with a machine translation module, and end-to-end (E2E) models that map speech directly to translated text. For high-resource European language pairs, cascade systems have been shown to remain competitive with or outperform E2E models [bentivogli2021cascade_vs_e2e], but this comparison has not been conducted for low-resource African languages. This paper evaluates a Whisper + NLLB-200 cascade pipeline [costajussa2022nllb_cascade_igbo] against end-to-end SeamlessM4T-v2 [barrault2023seamlessm4t_e2e_yoruba] on Yoruba, Hausa, and Igbo, using the FLEURS dataset. We analyse how ASR error propagation through the cascade affects final BLEU scores, compare inference latency across architectures, and identify the conditions under which each paradigm is preferable for African-language deployment. Our findings extend the cascade vs. E2E debate [bentivogli2021cascade_vs_e2e] to a low-resource, morphologically rich language setting where data conditions differ substantially from prior work.

---

## Chapter 1 — Introduction

### 1.1 Motivation

The choice between cascade and end-to-end speech translation architectures has significant practical consequences for African-language deployment. Cascade systems built from Whisper [barrault2023seamlessm4t_e2e_yoruba] and NLLB-200 [costajussa2022nllb_cascade_igbo] allow independent component upgrades and benefit from the rich ecosystem of ASR and MT research, while end-to-end models such as SeamlessM4T-v2 [barrault2023seamlessm4t_e2e_yoruba] offer simpler deployment and potentially lower error propagation. Studies on high-resource European pairs — using toolkits such as ESPnet-ST [inaguma2020espnetst] and datasets such as MuST-C and CoVoST-2 [wang2020covost2_yoruba] — show that cascade systems remain competitive [bentivogli2021cascade_vs_e2e], but the low-resource, tonally complex characteristics of Yoruba, Hausa, and Igbo may alter this balance. High ASR error rates in these languages could compound translation errors in cascade systems in ways not seen for high-resource pairs, making the architectural choice non-trivial.

### 1.2 Problem Statement

No study has directly compared cascade and end-to-end architectures for speech translation on low-resource African languages under a controlled evaluation protocol. Existing architectural comparisons [bentivogli2021cascade_vs_e2e][inaguma2020espnetst] focus on high-resource European pairs and do not generalise to languages with sparse training data and low ASR baseline quality. Meanwhile, African-language S2TT papers typically report results for a single architecture without ablating the cascade vs. E2E dimension. Without this comparison, practitioners deploying speech translation in Yoruba-, Hausa-, or Igbo-speaking contexts cannot make an evidence-based architectural choice, and researchers cannot assess whether investing in ASR quality or E2E joint training is the higher-leverage path.

### 1.3 Research Questions

**RQ1.** Does the Whisper + NLLB-200 cascade pipeline [costajussa2022nllb_cascade_igbo] achieve competitive BLEU scores compared to end-to-end SeamlessM4T-v2 [barrault2023seamlessm4t_e2e_yoruba] on Yoruba, Hausa, and Igbo, and how does the gap compare to that observed for high-resource European pairs [bentivogli2021cascade_vs_e2e]?

**RQ2.** How much of the BLEU gap between the cascade pipeline and the text-only MT ceiling from NLLB-200 [costajussa2022nllb_cascade_igbo] is attributable to ASR transcription errors, and how does this error propagation vary across the three target languages?

**RQ3.** Under what conditions — in terms of ASR WER threshold and available fine-tuning data — does a cascade architecture outperform an end-to-end model for African-language speech translation, and what does this imply for the development roadmap of systems like those evaluated in CoVoST-2 [wang2020covost2_yoruba]?

---

**Paper mode:** `cascade`  
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
Use one entry per paper × language × architecture combination.

**To activate a baseline in comparison tables: fill in `score`.**

### Entry format

```yaml
- citation_key: barrault2023seamlessm4t_e2e_yoruba
  type: article
  author: "Barrault, Loïc and others"
  title: "SeamlessM4T: Massively Multilingual & Multimodal Machine Translation"
  year: 2023
  journal: "arXiv preprint arXiv:2308.11596"
  url: "https://arxiv.org/abs/2308.11596"

  # --- comparison fields ---
  model: SeamlessM4T-v2-large
  datasets: [FLEURS]                             # list — all datasets evaluated on
  language: Yoruba
  directions: ["Source → English"]              # list — all task directions reported
  metrics: [BLEU]                               # list — score corresponds to metrics[0]
  score: null                                   # fill in E2E BLEU from Table B.1
  architecture: end_to_end                      # paper-specific required field
  summary: >
    SeamlessM4T-v2 E2E S2TT BLEU on FLEURS Yoruba. Used as the E2E upper bound for cascade
    comparison. The core question is whether our Whisper+NLLB cascade matches or exceeds
    this E2E performance on African languages.
  notes: "Fill in BLEU from Table B.1 of arXiv:2312.05187"
```

### Reference fields

#### Standard bibliographic

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `citation_key` | ★ Required | str | Unique identifier, one per paper × language × architecture |
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
| `model` | ★ Required | str | Full pipeline, e.g. `Whisper-large-v3 + NLLB-200-distilled-600M` |
| `datasets` | ★ Required | list | All datasets evaluated on, e.g. `[FLEURS]` or `[MuST-C]` |
| `language` | ★ Required | str | Target language — `Yoruba`, `Hausa`, `Igbo` |
| `directions` | ★ Required | list | All task directions, e.g. `["Source → English"]` |
| `metrics` | ★ Required | list | All metrics reported; `score` corresponds to `metrics[0]` |
| `score` | ★ Required | float\|null | Numeric score for `metrics[0]`; `null` = not yet filled |
| `summary` | ★ Required | str | 2–5 sentences on what the paper does and why it is relevant |
| `architecture` | ★ Required | str | `cascade`, `end_to_end`, or `direct` |
| `notes` | ○ Optional | str | Where to find the score, architecture details |
| `asr_model` | ○ Optional | str | ASR component in cascade, e.g. `Whisper-large-v3` |
| `mt_model` | ○ Optional | str | MT component in cascade, e.g. `NLLB-200-distilled-600M` |
| `asr_wer` | ○ Optional | float | Intermediate ASR WER before translation |
| `latency_ms` | ○ Optional | float | End-to-end inference latency in ms |

> **Architecture values:**
> - `cascade` — ASR + MT pipeline (separate components)
> - `end_to_end` — single model for direct speech-to-text translation (e.g. SeamlessM4T)
> - `direct` — direct ST without intermediate text (E2E non-autoregressive)

---

## Working with `sota_results.csv`

One row per system × language × architecture × metric. Flat fields — no lists.

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `paper_title` | ★ Required | str | Full paper title |
| `authors` | ★ Required | str | First author + et al. |
| `year` | ★ Required | int | 4-digit year |
| `model` | ★ Required | str | Full pipeline description |
| `dataset` | ★ Required | str | Dataset name |
| `language` | ★ Required | str | Display language name |
| `direction` | ★ Required | str | `Source → English` or `English → Source` |
| `metric` | ★ Required | str | `BLEU`, `spBLEU`, `ChrF`, `WER`, or `CER` |
| `score` | ★ Required | float | Numeric score |
| `citation_key` | ★ Required | str | Key matching an entry in `references.yaml` |
| `architecture` | ★ Required | str | `cascade`, `end_to_end`, or `direct` |
| `asr_model` | ○ Optional | str | ASR component (cascade only) |
| `mt_model` | ○ Optional | str | MT component (cascade only) |
| `asr_wer` | ○ Optional | float | Intermediate ASR WER |
| `latency_ms` | ○ Optional | float | Inference latency in ms |
| `notes` | ○ Optional | str | Architecture details |

---

## Published baselines to fill in

### End-to-end baselines (architecture: `end_to_end`)

| Model | Dataset | Languages | Metric | Where to find scores |
|-------|---------|-----------|--------|----------------------|
| SeamlessM4T-v2-large | FLEURS | Yoruba, Hausa | BLEU | Table B.1 of `arXiv:2312.05187` |

### MT ceiling — text-only (use as cascade upper bound)

| Model | Dataset | Languages | Metric | Where to find scores |
|-------|---------|-----------|--------|----------------------|
| NLLB-200-distilled-600M | Flores-200 | Igbo (`ibo_Latn`) | spBLEU | Table 2 of `arXiv:2207.04672` |

### Architecture framing references (cite, not compare directly)

| Paper | Relevance |
|-------|-----------|
| Bentivogli et al. 2021 (`arXiv:2106.01045`) | Cascade vs E2E analysis (European pairs) |
| ESPnet-ST — Inaguma et al. 2020 (`arXiv:2004.10234`) | Cascade pipeline toolkit reference |
| CoVoST-2 — Wang et al. 2021 (`arXiv:2007.10310`) | Multilingual ST cascade baseline (**check language coverage**) |

> **CoVoST-2 note:** Yoruba and Hausa are likely **not** in CoVoST-2. Verify Table 2 before
> filling in scores for those entries in `references.yaml`.

---

## Usage in run script

```python
PAPER_MODE = "cascade"
SOTA_FILE  = "sota/paper4_cascade/references.yaml"
```

---

## Experiment Steps

**Research question:** When does cascade (Whisper ASR → NLLB MT) outperform end-to-end (SeamlessM4T-v2), and vice versa? What is the ASR WER break-even point?

### Which experiments to run

Both FLEURS experiments — the cascade vs E2E comparison requires both architectures on the same dataset:

| Experiment | Architecture | Dataset | Languages |
|---|---|---|---|
| `FLEURS__SeamlessM4Tv2` | End-to-end | FLEURS | Igbo, Yoruba, Swahili |
| `FLEURS__WhisperNLLB` | Cascade | FLEURS | Yoruba, Hausa, Swahili |

The framework automatically runs three additional analyses when `PAPER_MODE = "cascade"`:
- **Oracle cascade** — NLLB MT on gold transcripts (no ASR errors) → cascade upper bound
- **Error propagation** — controlled WER noise injected into transcripts → BLEU drop curve
- **Break-even WER** — linear regression finds the ASR quality threshold where cascade ≈ E2E

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
   PAPER_MODE         = "cascade"
   DEBUG_MODE         = True
   ENABLE_FINETUNING  = False
   SCALING_BUDGETS    = []
   N_EVAL_RUNS        = 1
   EVAL_TEXT_SAMPLES  = [8]
   EVAL_AUDIO_SAMPLES = [3]
   SOTA_FILE          = ""
   FORCE_RERUN        = False
   ```
9. **Run Step 8** — debug run (~40–50 min)
10. **Run Steps 9, 10, 11** — verify E2E vs cascade table and error propagation curve appear in the report.

### Phase 3 — Full paper run

11. **Edit & run Step 7** — full settings:
    ```python
    PAPER_MODE         = "cascade"
    DEBUG_MODE         = False
    ENABLE_FINETUNING  = False
    SCALING_BUDGETS    = []
    N_EVAL_RUNS        = 3
    EVAL_TEXT_SAMPLES  = [100, 200, 300]
    EVAL_AUDIO_SAMPLES = [30,  75,  100]
    SOTA_FILE          = "sota/paper4_cascade/sota_results.csv"
    FORCE_RERUN        = False
    ```
12. **Run Step 8** — full experiments (~2–3 h on T4)
13. **Run Step 9** — consolidate metrics
14. **Run Step 10** — generate results report
15. **Run Step 11** — download ZIP

> **Optional — latency and VRAM profiling:** After the main run, open a terminal in Colab and run:
> ```bash
> python papers/paper4_cascade/run_cascade_analysis.py
> ```
> This measures median inference latency (ms) and peak VRAM (MB) per architecture and adds a hardware comparison table to the report.

### Outputs for the paper

| File in ZIP | Paper section |
|---|---|
| `consolidated_metrics/text_metrics.csv` | E2E vs cascade BLEU table |
| `consolidated_metrics/asr_metrics.csv` | Intermediate ASR WER (cascade component) |
| `*/tables/oracle_cascade*.md` | Oracle ceiling: cascade with perfect ASR |
| `*/tables/error_propagation*.md` | BLEU drop vs injected WER |
| `*/tables/breakeven*.md` | Break-even WER per language |
| `*/plots/error_propagation*.png` | Error propagation curve figures |
| `papers/cascade/results_report.md` | Architectural recommendation, discussion |

### If the session disconnects mid-run

1. Re-run Steps 1–4
2. Skip Steps 5 and 6
3. In Step 7, comment out the completed experiment
4. Re-run Steps 8–11
