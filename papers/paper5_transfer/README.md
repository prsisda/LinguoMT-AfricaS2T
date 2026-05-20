# Paper 5 — LinguoMT-Transfer: Cross-Lingual Adaptation in African Speech Translation

---

## Title

**LinguoMT-Transfer: Cross-Lingual Adaptation in African Speech Translation**

---

## Abstract

Cross-lingual transfer — leveraging representations learned from one language to improve performance on another — is a primary strategy for addressing the data scarcity that limits African-language speech technology. Large multilingual models such as mSLAM [bapna2022mslam_transfer_yoruba], XLS-R [babu2022xlsr_transfer_yoruba], and MMS [pratap2023mms_transfer_yoruba] acquire cross-lingual representations through joint pretraining on many languages, while systems such as MasakhaSpeech [dossou2022masakhaspeech_transfer_hausa] demonstrate that targeted fine-tuning with limited African-language data yields substantial gains. However, the effectiveness of cross-lingual transfer is known to depend on linguistic relatedness, and the African languages of interest — Yoruba and Igbo (Niger-Congo, Volta-Niger) versus Hausa (Afro-Asiatic, Chadic) — differ significantly in phonology and morphology. This paper investigates how linguistic family membership moderates cross-lingual transfer gains for ASR and speech translation, comparing zero-shot, few-shot, and cross-lingual fine-tuning strategies [barrault2023seamlessm4t_transfer_yoruba][pratap2023mms_transfer_yoruba][babu2022xlsr_transfer_yoruba] on the FLEURS dataset, and quantifying the relationship between typological similarity and transfer efficiency.

---

## Chapter 1 — Introduction

### 1.1 Motivation

The three African languages targeted by the LinguoMT project — Yoruba, Hausa, and Igbo — are among the most widely spoken on the continent, yet all three fall squarely in the low-resource regime for speech technology. Collecting sufficient target-language data to train or fully fine-tune models from scratch is rarely feasible, making cross-lingual transfer the dominant practical strategy. Systems such as XLS-R [babu2022xlsr_transfer_yoruba] and MMS [pratap2023mms_transfer_yoruba] are pretrained on dozens to thousands of languages, including some African ones, and can be fine-tuned with small amounts of target-language audio. However, the linguistic distance between the pretraining language distribution and the target language is known to modulate transfer effectiveness. Yoruba and Igbo are typologically related (both Niger-Congo, Volta-Niger branch), while Hausa belongs to the Afro-Asiatic family — a substantially different phonological and morphological profile. Understanding whether this typological difference translates into measurably different cross-lingual transfer efficiency is critical for designing data collection and model selection strategies for African-language speech systems.

### 1.2 Problem Statement

No study has systematically examined how linguistic family membership and typological distance moderate cross-lingual transfer gains specifically for African-language ASR and speech translation. Existing cross-lingual transfer evaluations — including those reported for MMS [pratap2023mms_transfer_yoruba], XLS-R [babu2022xlsr_transfer_yoruba], mSLAM [bapna2022mslam_transfer_yoruba], and MasakhaSpeech [dossou2022masakhaspeech_transfer_hausa] — report per-language results without controlling for linguistic relatedness or systematically comparing zero-shot, few-shot, and fine-tuning regimes under matched data conditions. Without this analysis, it is unclear whether the better zero-shot performance of SeamlessM4T-v2 [barrault2023seamlessm4t_transfer_yoruba] on Yoruba versus Hausa reflects model capacity, pretraining data imbalance, or genuine typological transfer effects.

### 1.3 Research Questions

**RQ1.** Do typologically related language pairs (Yoruba–Igbo, both Niger-Congo) exhibit higher cross-lingual transfer efficiency — measured as WER reduction per training sample — compared to typologically distant pairs (Yoruba–Hausa, Niger-Congo vs. Afro-Asiatic), when fine-tuning XLS-R [babu2022xlsr_transfer_yoruba] and MMS [pratap2023mms_transfer_yoruba] on FLEURS?

**RQ2.** Across zero-shot [barrault2023seamlessm4t_transfer_yoruba][bapna2022mslam_transfer_yoruba], few-shot, and cross-lingual fine-tuning [pratap2023mms_transfer_yoruba][dossou2022masakhaspeech_transfer_hausa] regimes on matched data budgets, which transfer strategy yields the most favourable performance–efficiency trade-off for Yoruba, Hausa, and Igbo ASR?

**RQ3.** How does the amount of target-language fine-tuning data interact with the source language family of the pretrained model to determine final ASR WER, and at what data size does cross-lingual fine-tuning [babu2022xlsr_transfer_hausa] converge towards the performance of full fine-tuning [dossou2022masakhaspeech_transfer_yoruba]?

---

**Paper mode:** `transfer`  
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
Use one entry per paper × language × transfer type combination.

**To activate a baseline in comparison tables: fill in `score`.**

### Entry format

```yaml
- citation_key: pratap2023mms_transfer_yoruba
  type: article
  author: "Pratap, Vineel and Tjandra, Andros and others"
  title: "Scaling Speech Technology to 1,000+ Languages"
  year: 2023
  journal: "arXiv preprint arXiv:2305.13516"
  url: "https://arxiv.org/abs/2305.13516"

  # --- comparison fields ---
  model: MMS-300M
  datasets: [FLEURS]                             # list — all datasets evaluated on
  language: Yoruba
  directions: [ASR]                              # list — all task directions reported
  metrics: [WER]                                 # list — score corresponds to metrics[0]
  score: null                                    # fill in WER from paper appendix
  transfer_type: cross_lingual_finetune          # paper-specific required field
  summary: >
    ASR model covering 1100+ languages via cross-lingual adapter fine-tuning on a shared
    multilingual backbone. Key cross-lingual transfer baseline — shows WER achievable with
    language-specific adapters on a shared multilingual model.
  notes: "Fill in WER from per-language FLEURS ASR appendix"
```

### Reference fields

#### Standard bibliographic

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `citation_key` | ★ Required | str | Unique identifier, one per paper × language × transfer type |
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
| `model` | ★ Required | str | Model name and variant |
| `datasets` | ★ Required | list | All datasets evaluated on, e.g. `[FLEURS]` or `[MasakhaSpeech]` |
| `language` | ★ Required | str | **Target** language — `Yoruba`, `Hausa`, `Igbo` |
| `directions` | ★ Required | list | All task directions, e.g. `[ASR]` or `["Source → English"]` |
| `metrics` | ★ Required | list | All metrics reported; `score` corresponds to `metrics[0]` |
| `score` | ★ Required | float\|null | Numeric score for `metrics[0]`; `null` = not yet filled |
| `summary` | ★ Required | str | 2–5 sentences on what the paper does and why it is relevant |
| `transfer_type` | ★ Required | str | `zero_shot`, `few_shot`, `cross_lingual_finetune`, `multilingual` |
| `notes` | ○ Optional | str | Where to find the score, transfer conditions |
| `source_lang_family` | ○ Optional | str | Language family of pretraining data: `Niger-Congo`, `Afro-Asiatic`, `mixed` |
| `target_lang_family` | ○ Optional | str | Language family of evaluation language |
| `num_ft_samples` | ○ Optional | int | Adaptation samples used — `0` for zero-shot |
| `pretrain_langs` | ○ Optional | str | Languages in pretraining, e.g. `128 languages incl. Yoruba` |
| `lang_similarity_score` | ○ Optional | float | Typological similarity score (e.g. URIEL cosine), range 0–1 |

> **Transfer type values:**
> - `zero_shot` — no target-language fine-tuning; relies on multilingual pretraining
> - `few_shot` — very limited target-language data (< 1h)
> - `cross_lingual_finetune` — fine-tuning with target-language data using a shared backbone
> - `multilingual` — joint training on multiple languages simultaneously

> **Language families for our target languages:**
> - Yoruba → Niger-Congo (Volta-Niger)
> - Igbo → Niger-Congo (Volta-Niger) — typologically related to Yoruba
> - Hausa → Afro-Asiatic (Chadic) — typologically distant from Yoruba/Igbo

---

## Working with `sota_results.csv`

One row per system × language × transfer type × metric. Flat fields — no lists.

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `paper_title` | ★ Required | str | Full paper title |
| `authors` | ★ Required | str | First author + et al. |
| `year` | ★ Required | int | 4-digit year |
| `model` | ★ Required | str | Model name |
| `dataset` | ★ Required | str | Dataset name |
| `language` | ★ Required | str | Target language display name |
| `direction` | ★ Required | str | `Source → English` or `English → Source` |
| `metric` | ★ Required | str | `BLEU`, `spBLEU`, `ChrF`, `WER`, or `CER` |
| `score` | ★ Required | float | Score on the target language |
| `citation_key` | ★ Required | str | Key matching an entry in `references.yaml` |
| `transfer_type` | ★ Required | str | `zero_shot`, `few_shot`, `cross_lingual_finetune`, `multilingual` |
| `source_lang_family` | ○ Optional | str | Language family of pretraining data |
| `target_lang_family` | ○ Optional | str | Language family of evaluation language |
| `num_ft_samples` | ○ Optional | int | Adaptation samples — `0` for zero-shot |
| `pretrain_langs` | ○ Optional | str | Languages in pretraining set |
| `lang_similarity_score` | ○ Optional | float | Typological similarity (URIEL cosine) |
| `notes` | ○ Optional | str | Transfer conditions, data sources |

> **Tip:** Add one row per transfer condition to build a learning-curve table:
> ```
> language=Hausa, transfer_type=zero_shot,           num_ft_samples=0,   WER=0.82
> language=Hausa, transfer_type=cross_lingual_finetune, num_ft_samples=500, WER=0.51
> ```

---

## Published baselines to fill in

### Zero-shot transfer baselines

| Model | Dataset | Languages | Metric | Where to find scores |
|-------|---------|-----------|--------|----------------------|
| SeamlessM4T-v2-large | FLEURS | Yoruba | BLEU | Table B.1 of `arXiv:2312.05187` |
| mSLAM | FLEURS | Yoruba, Hausa | BLEU | Supplementary S2TT table of `arXiv:2202.01374` |

### Cross-lingual fine-tune baselines

| Model | Dataset | Languages | Metric | Where to find scores |
|-------|---------|-----------|--------|----------------------|
| MMS-300M | FLEURS | Yoruba, Hausa | WER | Per-language appendix of `arXiv:2305.13516` |
| XLS-R-1B | FLEURS | Yoruba, Hausa | WER | Table 4 of `arXiv:2111.09296` |
| Wav2Vec2-XLSR (full FT) | MasakhaSpeech | Hausa, Yoruba | WER | Table 2 of `arXiv:2206.00253` |

### Language similarity resources

| Resource | Description |
|----------|-------------|
| URIEL / lang2vec | Typological feature vectors — `github.com/antonisa/lang2vec` |
| ASJP | Phonological distance database |

---

## Usage in run script

```python
PAPER_MODE = "transfer"
SOTA_FILE  = "sota/paper5_transfer/references.yaml"
```
