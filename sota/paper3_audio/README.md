# Paper 3 — LinguoMT-Audio: Audio Preprocessing and Robustness for African Speech Translation

**Paper mode:** `audio`  
**Schema file:** `schema.json` — read by the framework to validate your data before generating comparison tables.

---

## How to add results

1. Open `sota_results.csv` in this folder.
2. Add one row per system × language × **audio condition** × metric.
   Each audio condition (clean, noisy, normalized, etc.) should be a separate row.
3. Fill in every **★ Required** field. Leave **○ Optional** fields blank if unknown.
4. Add the citation key to `sota/paper_references.csv`.
5. Set `SOTA_FILE = "sota/paper3_audio/sota_results.csv"` in your run script.

The framework validates against `schema.json` at load time. Rows with missing required fields
are skipped with a printed warning.

---

## Field reference

### Base fields (all papers)

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `paper_title` | ★ Required | str | Full paper title |
| `authors` | ★ Required | str | First author + et al. |
| `year` | ★ Required | int | 4-digit publication year |
| `model` | ★ Required | str | Model name |
| `dataset` | ★ Required | str | Evaluation dataset name |
| `language` | ★ Required | str | Display name matching our system — `Yoruba`, `Hausa`, `Igbo` |
| `direction` | ★ Required | str | `Source → English` or `English → Source` |
| `metric` | ★ Required | str | `BLEU`, `ChrF`, `WER`, or `CER` |
| `score` | ★ Required | float | Numeric score — must not be empty |
| `citation_key` | ★ Required | str | BibTeX key also listed in `sota/paper_references.csv` |
| `notes` | ○ Optional | str | Details about the audio condition or evaluation setup |

### Paper-specific fields

| Field | Status | Type | Description |
|-------|--------|------|-------------|
| `audio_condition` | ★ Required | str | Label for the audio setup: `clean`, `normalized`, `trimmed`, `chunked`, `noisy` |
| `noise_type` | ○ Optional | str | Type of noise: `babble`, `music`, `street`, `white`, `none` |
| `snr_db` | ○ Optional | float | Signal-to-noise ratio in dB, e.g. `10.0` or `20.0` |
| `vad_applied` | ○ Optional | bool | Whether VAD was applied: `true` or `false` |
| `sample_rate_hz` | ○ Optional | int | Audio sample rate used, typically `16000` |

> **Tip:** Add one row per condition so the framework can build a pivot table
> of `audio_condition × language → metric`. Example:
> ```
> model=Whisper, condition=clean,      language=Yoruba, WER=0.42
> model=Whisper, condition=normalized, language=Yoruba, WER=0.38
> model=Whisper, condition=noisy,      language=Yoruba, WER=0.61
> ```

---

## What papers to look for

### Noise robustness and audio augmentation

| System | Authors | Year | arXiv / Venue |
|--------|---------|------|---------------|
| SpecAugment | Park et al. | 2019 | INTERSPEECH 2019 — standard augmentation baseline |
| Whisper noise analysis | Various | 2023–2024 | Search "Whisper noise robustness evaluation" |
| Robust ASR African languages | Various | 2022–2024 | INTERSPEECH / ICASSP |

### VAD and chunking

| System | Authors | Year | Source |
|--------|---------|------|--------|
| Silero VAD | Silero Team | 2021 | GitHub: snakers4/silero-vad |
| Whisper long-form transcription | Radford et al. | 2023 | ICML 2023 |

---

## Usage in run script

```python
PAPER_MODE = "audio"
SOTA_FILE  = "sota/paper3_audio/sota_results.csv"
```
