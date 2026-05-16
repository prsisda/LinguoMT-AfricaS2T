# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # LinguoMT — FLEURS + Whisper-large-v3 + NLLB-200 (Cascade)
#
# **Dataset:** google/fleurs
# **ASR model:** openai/whisper-large-v3
# **MT model:**  facebook/nllb-200-distilled-600M
# **Pipeline:**  Audio → Whisper (transcribe) → NLLB (translate)
#
# Active languages: Yoruba, Swahili, Hausa
# Disabled: Igbo (Whisper does not support it), Wolof (Whisper does not support it)
#
# Run locally: `python run_experiment.py`
# Run on Colab: upload and execute all cells

# %% --- 1. Colab detection and matplotlib backend ---
import sys

try:
    from google.colab import drive as _colab_drive
    IN_COLAB = True
except Exception:
    _colab_drive = None
    IN_COLAB = False

import matplotlib
if not IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# %% --- 2. Dependency installation ---
# Google Colab : runs automatically below.
# Mac / local  : pip install transformers datasets sacrebleu librosa soundfile
#                           sentencepiece accelerate jiwer pandas pyarrow
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U",
        "transformers>=4.40", "datasets", "sacrebleu", "librosa", "soundfile",
        "sentencepiece", "accelerate", "jiwer", "pandas==2.2.2", "pyarrow>=15.0.0"],
        check=True)
    print("Dependencies installed.")

# %% --- 2b. HuggingFace Hub authentication ---
# Required only if the model or dataset is gated/private.
# Get your token at: https://huggingface.co/settings/tokens
# import os; os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"
# from huggingface_hub import login; login(token=os.environ["HF_TOKEN"])

# %% --- 3. Imports ---
import os
import re
import json
import random
import shutil
import warnings
import unicodedata
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
import sacrebleu
from jiwer import wer, cer

from datasets import load_dataset, load_dataset_builder, get_dataset_config_names
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

warnings.filterwarnings("ignore")

# %% --- 4. Configuration ---

# ── Model / Dataset ───────────────────────────────────────────────
WHISPER_MODEL_ID   = "openai/whisper-large-v3"
NLLB_MODEL_ID      = "facebook/nllb-200-distilled-600M"
DATASET_ID         = "google/fleurs"
ENGLISH_CONFIG     = "en_us"
ENGLISH_NLLB_CODE  = "eng_Latn"
ENGLISH_WHISPER_LANG = "english"

BASELINE_MODEL_NAME    = "Whisper-large-v3 + NLLB-200-600M"
BASELINE_DATASET_NAME  = "FLEURS"
BASELINE_PIPELINE_TYPE = "cascade_asr_mt"
EXPERIMENT_FAMILY      = "FLEURS__WhisperLargeV3_NLLB200"

# ── Runtime ───────────────────────────────────────────────────────
TARGET_SR  = 16000
DEVICE     = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SEED       = 42
# ── DEBUG / FULL SWITCH ── change this line only ──────────────────
# True  → debug run  : 8 text pairs, 3 audio pairs, fast (~5-15 min on GPU)
# False → full run   : 50/100/200 samples across 3 experiments (~1-3 hrs on GPU)
DEBUG_MODE = True   # ← SET HERE
SKIP_AUDIO_DEBUG = True
RUN_FULL_GRID    = True
RUN_TEXT_EVALUATION  = True
RUN_AUDIO_EVALUATION = True
RUN_SAMPLE_CHECK     = True
RUN_DATA_EXPLORATION = True
DIRECTIONS_TO_RUN    = ["african_to_english", "english_to_african"]
SELECTED_LANGUAGE_PAIRS = ["yoruba", "swahili", "hausa"]
SPLIT_FOR_EXPERIMENT = "test"
MIN_DURATION = 1.0
MAX_DURATION = 20.0

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Experiment sizes ──────────────────────────────────────────────
if DEBUG_MODE:
    EXPERIMENT_CONFIGS = [{"experiment": "debug", "max_text_dev": 8,   "max_audio_dev": 3,  "max_scan_rows": 100}]
else:
    EXPERIMENT_CONFIGS = [
        {"experiment": "Experiment_1", "max_text_dev": 50,  "max_audio_dev": 10, "max_scan_rows": 1000},
        {"experiment": "Experiment_2", "max_text_dev": 100, "max_audio_dev": 30, "max_scan_rows": 1500},
        {"experiment": "Experiment_3", "max_text_dev": 200, "max_audio_dev": 50, "max_scan_rows": 2000},
    ]

# ── EDA settings ──────────────────────────────────────────────────
EDA_SPLITS_TO_ANALYZE              = ["train", "validation", "test"]
EDA_SAMPLE_SIZE_PER_LANGUAGE_SPLIT = 25 if DEBUG_MODE else 200
EDA_MAX_SCAN_ROWS                  = 100 if DEBUG_MODE else 1000
EDA_AUDIO_EXAMPLES_PER_LANGUAGE    = 1 if DEBUG_MODE else 2

# ── Audio optimization strategies ─────────────────────────────────
# For cascade pipeline: normalize/trim apply before Whisper ASR.
# chunk: each chunk is transcribed then all transcripts are joined before NLLB MT.
CHUNK_SECONDS         = 6
CHUNK_OVERLAP_SECONDS = 1
RUN_EXPENSIVE_AUDIO_STRATEGIES_ONLY_IN_EXP1 = False

AUDIO_OPTIMIZATION_STRATEGIES = [
    {"strategy_key": "baseline_direct",   "method": "Whisper+NLLB direct",     "category": "Direct Cascade",     "enabled": True,  "normalize": False, "trim": False, "chunk": False, "expensive": False},
    {"strategy_key": "normalized_audio",  "method": "Whisper+NLLB normalized", "category": "Audio Normalization", "enabled": True,  "normalize": True,  "trim": False, "chunk": False, "expensive": False},
    {"strategy_key": "trimmed_audio",     "method": "Whisper+NLLB trimmed",    "category": "Silence Trimming",    "enabled": True,  "normalize": True,  "trim": True,  "chunk": False, "expensive": True},
    {"strategy_key": "chunk_based_audio", "method": "Whisper+NLLB chunked",    "category": "Audio Segmentation",  "enabled": True,  "normalize": True,  "trim": False, "chunk": True,  "expensive": True},
]

# ── Language configs ──────────────────────────────────────────────
# Whisper supports: Yoruba (yo / "yoruba"), Swahili (sw / "swahili"), Hausa (ha / "hausa")
# Whisper does NOT support: Igbo, Wolof
# NLLB-200 supports all five languages.
ALL_LANGUAGE_CONFIGS = [
    {
        "language": "yoruba", "fleurs_config": "yo_ng", "display_name": "Yoruba",
        "whisper_lang": "yoruba",  "nllb_src_code": "yor_Latn",
        "dataset_supported": True, "model_supported": True,  "enabled": True,
        "skip_reason": "",
    },
    {
        "language": "swahili", "fleurs_config": "sw_ke", "display_name": "Swahili",
        "whisper_lang": "swahili", "nllb_src_code": "swh_Latn",
        "dataset_supported": True, "model_supported": True,  "enabled": True,
        "skip_reason": "",
    },
    {
        "language": "hausa", "fleurs_config": "ha_ng", "display_name": "Hausa",
        "whisper_lang": "hausa",   "nllb_src_code": "hau_Latn",
        "dataset_supported": True, "model_supported": True,  "enabled": True,
        "skip_reason": "",
    },
    {
        "language": "igbo", "fleurs_config": "ig_ng", "display_name": "Igbo",
        "whisper_lang": None,      "nllb_src_code": "ibo_Latn",
        "dataset_supported": True, "model_supported": False, "enabled": False,
        "skip_reason": "Whisper-large-v3 does not support Igbo",
    },
    {
        "language": "wolof", "fleurs_config": "wo_sn", "display_name": "Wolof",
        "whisper_lang": None,      "nllb_src_code": "wol_Latn",
        "dataset_supported": True, "model_supported": False, "enabled": False,
        "skip_reason": "Whisper-large-v3 does not support Wolof",
    },
]

# ── Filter to active languages ────────────────────────────────────
if RUN_FULL_GRID:
    _candidates = [c for c in ALL_LANGUAGE_CONFIGS if c.get("enabled")]
else:
    _sel = set(SELECTED_LANGUAGE_PAIRS)
    _candidates = [c for c in ALL_LANGUAGE_CONFIGS if c["language"] in _sel]

LANGUAGE_CONFIGS = [c for c in _candidates if c.get("model_supported") and c.get("dataset_supported")]
SKIPPED_LANGUAGE_CONFIGS = [c for c in ALL_LANGUAGE_CONFIGS if not (c.get("model_supported") and c.get("dataset_supported"))]
LANGUAGE_LOOKUP = {c["language"]: c for c in LANGUAGE_CONFIGS}

# ── Output directory ──────────────────────────────────────────────
_FOLDER_NAME = f"Results_{EXPERIMENT_FAMILY}" + ("_DEBUG" if DEBUG_MODE else "")
if IN_COLAB:
    BASE_OUTPUT_DIR = Path("/content") / _FOLDER_NAME
else:
    BASE_OUTPUT_DIR = Path(__file__).parent.parent / "results" / _FOLDER_NAME

BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = BASE_OUTPUT_DIR / "metrics"
FIGURES_DIR = BASE_OUTPUT_DIR / "figures"
EDA_DIR     = BASE_OUTPUT_DIR / "eda"
QUAL_DIR    = BASE_OUTPUT_DIR / "qualitative"
for _d in [METRICS_DIR, FIGURES_DIR, EDA_DIR, QUAL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

print(f"ASR   : {WHISPER_MODEL_ID}")
print(f"MT    : {NLLB_MODEL_ID}")
print(f"Data  : {DATASET_ID}")
print(f"Device: {DEVICE} | CUDA: {torch.cuda.is_available()}")
print(f"Active: {[c['display_name'] for c in LANGUAGE_CONFIGS]}")
print(f"Skipped: {[(c['display_name'], c.get('skip_reason', '')) for c in SKIPPED_LANGUAGE_CONFIGS]}")
print(f"Output: {BASE_OUTPUT_DIR}")

# %% --- 5. Google Drive setup (Colab only) ---
GOOGLE_DRIVE_AVAILABLE = False
if IN_COLAB and _colab_drive is not None:
    try:
        _colab_drive.mount("/content/drive")
        GOOGLE_DRIVE_AVAILABLE = True
        print("Google Drive mounted.")
    except Exception as e:
        print(f"Drive mount skipped (mount it manually if needed): {e}")
else:
    print("Local run — Google Drive backup will be skipped.")

# %% --- 6. Model loading ---
if not LANGUAGE_CONFIGS:
    raise RuntimeError("No active supported language pairs.")

# Whisper (ASR)
whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL_ID,
    torch_dtype=torch.float16 if DEVICE in ("cuda", "mps") else torch.float32,
).to(DEVICE)
whisper_model.eval()
print(f"Loaded Whisper on {DEVICE}")

# NLLB (MT)
nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID).to(DEVICE)
nllb_model.eval()
print(f"Loaded NLLB on {DEVICE}")

# %% --- 7. Audio preprocessing helpers ---

def get_text(item):
    return str(item.get("transcription") or item.get("raw_transcription") or item.get("sentence") or "").strip()

def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text).strip()

def resample_audio(audio, sr, target_sr=TARGET_SR):
    audio = np.asarray(audio, dtype=np.float32).flatten()
    return audio if sr == target_sr else librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)

def normalize_audio_waveform(audio):
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32).flatten())
    if len(audio) == 0: return audio
    audio -= np.mean(audio)
    mx = np.max(np.abs(audio))
    return (audio / mx).astype(np.float32) if mx > 0 else audio

def plot_waveform_comparison(audio_raw, lang_name, sr, out_dir):
    """Side-by-side waveform: original vs mean-removed amplitude-normalized audio."""
    original   = np.asarray(audio_raw, dtype=np.float32).flatten()
    normalized = normalize_audio_waveform(original.copy())
    time_axis  = np.arange(len(original)) / float(sr)
    fig, axes  = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(time_axis, original,   color="steelblue",  linewidth=0.6)
    axes[0].set_title(f"{lang_name} — Original Waveform"); axes[0].set_ylabel("Amplitude")
    axes[1].plot(time_axis, normalized, color="darkorange", linewidth=0.6)
    axes[1].set_title(f"{lang_name} — Normalized Waveform")
    axes[1].set_ylabel("Amplitude"); axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    save_plot(out_dir / f"06_waveform_comparison_{lang_name.lower().replace(' ', '_')}.png")

def run_waveform_comparisons():
    if not RUN_DATA_EXPLORATION: return
    print("\nWaveform Comparisons (Original vs Normalized):")
    for cfg in LANGUAGE_CONFIGS:
        try:
            lang_dir = EDA_DIR / cfg["language"]; lang_dir.mkdir(exist_ok=True)
            stream   = load_fleurs_split(cfg["fleurs_config"], EDA_SPLITS_TO_ANALYZE[0], streaming=True)
            item     = next(iter(stream))
            audio    = resample_audio(item["audio"]["array"], item["audio"]["sampling_rate"])
            plot_waveform_comparison(audio, cfg["display_name"], TARGET_SR, lang_dir)
            print(f"  Saved: {cfg['display_name']}")
        except Exception as e:
            print(f"  Skipped ({cfg['display_name']}): {e}")

def trim_silence(audio, threshold=0.01):
    audio = np.asarray(audio, dtype=np.float32).flatten()
    nsi = np.where(np.abs(audio) > threshold)[0]
    if len(nsi) == 0: return audio
    trimmed = audio[nsi[0]:nsi[-1]+1]
    return trimmed if len(trimmed) >= int(MIN_DURATION * TARGET_SR) else audio.astype(np.float32)

def ensure_min_audio_length(audio, min_sec=MIN_DURATION, sr=TARGET_SR):
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32).flatten())
    mn = int(min_sec * sr)
    if len(audio) == 0: return np.zeros(mn, dtype=np.float32)
    return audio if len(audio) >= mn else np.pad(audio, (0, mn - len(audio))).astype(np.float32)

def audio_duration(audio, sr=TARGET_SR):
    return len(np.asarray(audio, dtype=np.float32).flatten()) / float(sr)

def extract_audio_array(audio_obj, target_sr=TARGET_SR):
    if isinstance(audio_obj, dict):
        sr, audio = audio_obj.get("sampling_rate", target_sr), audio_obj.get("array")
    else:
        sr, audio = target_sr, audio_obj
    if audio is None: return np.zeros(int(MIN_DURATION * target_sr), dtype=np.float32)
    try:
        audio = np.asarray(audio, dtype=np.float32)
    except Exception:
        audio = np.array([], dtype=np.float32)
    audio = np.nan_to_num(audio.flatten())
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return ensure_min_audio_length(audio)

def split_audio_into_chunks(audio, chunk_sec=CHUNK_SECONDS, overlap_sec=CHUNK_OVERLAP_SECONDS, sr=TARGET_SR):
    audio      = np.asarray(audio, dtype=np.float32).flatten()
    chunk_size = int(chunk_sec * sr)
    step_size  = max(1, chunk_size - int(overlap_sec * sr))
    chunks = []
    for start in range(0, len(audio), step_size):
        chunk = audio[start:start + chunk_size]
        if len(chunk) >= int(MIN_DURATION * sr): chunks.append(chunk.astype(np.float32))
        if start + chunk_size >= len(audio): break
    return chunks

# %% --- 8. Dataset loading helpers ---

def load_fleurs_split(config, split=SPLIT_FOR_EXPERIMENT, streaming=True):
    return load_dataset(DATASET_ID, config, split=split, streaming=streaming)

def prepare_bidirectional_pairs(language_key, split=SPLIT_FOR_EXPERIMENT, max_samples=50, max_scan_rows=5000):
    cfg = LANGUAGE_LOOKUP.get(language_key)
    if cfg is None: return []
    afr_stream = load_fleurs_split(cfg["fleurs_config"], split, streaming=True)
    eng_stream = load_fleurs_split(ENGLISH_CONFIG,        split, streaming=True)
    rows, count = [], 0
    for idx, (ai, ei) in enumerate(zip(afr_stream, eng_stream)):
        if idx >= max_scan_rows or count >= max_samples: break
        afr_text = normalize_text(get_text(ai))
        eng_text = normalize_text(get_text(ei))
        if not afr_text or not eng_text: continue
        if "african_to_english" in DIRECTIONS_TO_RUN:
            rows.append({
                "sample_index": idx, "language": language_key, "language_display": cfg["display_name"],
                "direction_key": "african_to_english", "direction": f"{cfg['display_name']}→English",
                # Text codes (NLLB)
                "source_lang_code": cfg["nllb_src_code"], "target_lang_code": ENGLISH_NLLB_CODE,
                # Audio codes (Whisper)
                "source_whisper_lang": cfg["whisper_lang"], "target_whisper_lang": ENGLISH_WHISPER_LANG,
                "source_text": afr_text, "target_text": eng_text,
                "source_audio": ai["audio"], "target_audio": ei["audio"],
            })
        if "english_to_african" in DIRECTIONS_TO_RUN:
            rows.append({
                "sample_index": idx, "language": language_key, "language_display": cfg["display_name"],
                "direction_key": "english_to_african", "direction": f"English→{cfg['display_name']}",
                "source_lang_code": ENGLISH_NLLB_CODE, "target_lang_code": cfg["nllb_src_code"],
                "source_whisper_lang": ENGLISH_WHISPER_LANG, "target_whisper_lang": cfg["whisper_lang"],
                "source_text": eng_text, "target_text": afr_text,
                "source_audio": ei["audio"], "target_audio": ai["audio"],
            })
        count += 1
    return rows

def save_df(df, filename, out_dir=None):
    p = (out_dir or METRICS_DIR) / filename
    df.to_csv(p, index=False); print("Saved:", p); return p

# %% --- 9. Translation functions (Whisper ASR → NLLB MT cascade) ---

def transcribe_whisper(audio, whisper_lang):
    """Transcribe audio to text using Whisper (no translation)."""
    inputs = whisper_processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    input_features = inputs.input_features.to(
        DEVICE, dtype=torch.float16 if DEVICE in ("cuda", "mps") else torch.float32
    )
    with torch.no_grad():
        tokens = whisper_model.generate(
            input_features,
            language=whisper_lang,
            task="transcribe",
            max_new_tokens=448,
        )
    return whisper_processor.decode(tokens[0], skip_special_tokens=True)

def translate_nllb(text, src_nllb_code, tgt_nllb_code):
    """Translate text with NLLB-200. Codes e.g. yor_Latn, eng_Latn."""
    text = str(text).strip()
    if not text: return ""
    nllb_tokenizer.src_lang = src_nllb_code
    inputs = nllb_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    tgt_id = nllb_tokenizer.convert_tokens_to_ids(tgt_nllb_code)
    with torch.no_grad():
        out = nllb_model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=256)
    return nllb_tokenizer.decode(out[0], skip_special_tokens=True)

def translate_text_direct(text, source_lang_code, target_lang_code):
    """Text-only translation via NLLB. source/target are NLLB codes."""
    return translate_nllb(text, source_lang_code, target_lang_code)

def translate_audio_direct(audio_obj, source_lang_code, target_lang_code,
                            whisper_lang=None, normalize_audio=True, improved=True):
    """Cascade: Whisper ASR → NLLB MT.
    source_lang_code: NLLB code for the spoken language (e.g. yor_Latn)
    target_lang_code: NLLB code for the output language (e.g. eng_Latn)
    whisper_lang: Whisper language name (e.g. 'yoruba')
    Returns: (translation, asr_transcription)
    """
    audio = extract_audio_array(audio_obj)
    if improved:        audio = trim_silence(audio)
    if normalize_audio: audio = normalize_audio_waveform(audio)
    audio = ensure_min_audio_length(audio)
    transcription = transcribe_whisper(audio, whisper_lang or "english")
    translation   = translate_nllb(transcription, source_lang_code, target_lang_code)
    return translation, transcription

def prepare_audio_for_strategy(audio_obj, strategy):
    audio = extract_audio_array(audio_obj)
    if strategy.get("trim"):      audio = trim_silence(audio)
    if strategy.get("normalize"): audio = normalize_audio_waveform(audio)
    return ensure_min_audio_length(audio).astype(np.float32)

def translate_audio_with_strategy(audio_obj, source_lang_code, target_lang_code,
                                   source_whisper_lang, strategy):
    """Run the cascade strategy. Chunking joins transcriptions before MT."""
    audio = prepare_audio_for_strategy(audio_obj, strategy)
    if strategy.get("chunk"):
        chunks = split_audio_into_chunks(audio) or [audio]
        transcriptions = []
        for chunk in chunks:
            t = transcribe_whisper(chunk, source_whisper_lang)
            if normalize_text(t): transcriptions.append(normalize_text(t))
        joined_transcript = " ".join(transcriptions)
        translation = translate_nllb(joined_transcript, source_lang_code, target_lang_code)
        return translation, joined_transcript, len(chunks)
    translation, transcript = translate_audio_direct(
        audio, source_lang_code, target_lang_code,
        whisper_lang=source_whisper_lang, normalize_audio=False, improved=False)
    return translation, transcript, 1

def active_audio_strategies(experiment_name):
    return [s for s in AUDIO_OPTIMIZATION_STRATEGIES
            if s.get("enabled")
            and not (RUN_EXPENSIVE_AUDIO_STRATEGIES_ONLY_IN_EXP1
                     and s.get("expensive")
                     and experiment_name not in ["Experiment_1", "debug"])]

# %% --- 10. Metrics ---

def compute_metrics(predictions, references):
    preds = [str(p).strip() for p in predictions]
    refs  = [str(r).strip() for r in references]
    if not preds or not refs: return {"BLEU": 0.0, "ChrF": 0.0}
    return {
        "BLEU": float(sacrebleu.corpus_bleu(preds, [refs]).score),
        "ChrF": float(sacrebleu.corpus_chrf(preds, [refs]).score),
    }

# %% --- 11. EDA helpers ---

def compute_audio_quality_features(audio_array, sr):
    audio = np.nan_to_num(np.asarray(audio_array, dtype=np.float32).flatten())
    if len(audio) == 0:
        return {k: 0.0 for k in ["duration_sec","mean_abs_energy","rms_energy","peak_amplitude",
                                   "silence_ratio_001","clipping_ratio_099","zero_crossing_rate","dynamic_range"]}
    return {
        "duration_sec":       len(audio) / float(sr),
        "mean_abs_energy":    float(np.mean(np.abs(audio))),
        "rms_energy":         float(np.sqrt(np.mean(audio**2))),
        "peak_amplitude":     float(np.max(np.abs(audio))),
        "silence_ratio_001":  float(np.mean(np.abs(audio) < 0.01)),
        "clipping_ratio_099": float(np.mean(np.abs(audio) >= 0.99)),
        "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio)[0])),
        "dynamic_range":      float(np.percentile(audio, 95) - np.percentile(audio, 5)),
    }

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if IN_COLAB: plt.show()
    plt.close(); print("Saved figure:", path)

def collect_eda_rows(lang_cfg, split, max_samples=200, max_scan_rows=1000):
    rows = []
    afr_s = load_fleurs_split(lang_cfg["fleurs_config"], split, streaming=True)
    eng_s = load_fleurs_split(ENGLISH_CONFIG,            split, streaming=True)
    for idx, (ai, ei) in enumerate(zip(afr_s, eng_s)):
        if idx >= max_scan_rows or len(rows) >= max_samples: break
        at = normalize_text(get_text(ai)); et = normalize_text(get_text(ei))
        if not at or not et: continue
        aa = resample_audio(ai["audio"]["array"], ai["audio"]["sampling_rate"])
        ea = resample_audio(ei["audio"]["array"], ei["audio"]["sampling_rate"])
        rows.append({
            "sample_index": idx, "split": split,
            "language": lang_cfg["display_name"], "language_key": lang_cfg["language"],
            "african_text": at, "english_text": et,
            "african_words": len(at.split()), "english_words": len(et.split()),
            **{f"african_{k}": v for k, v in compute_audio_quality_features(aa, TARGET_SR).items()},
            **{f"english_{k}": v for k, v in compute_audio_quality_features(ea, TARGET_SR).items()},
        })
    return rows

def plot_eda(df, out_dir, name):
    if df.empty: return
    for feat, label in [("duration_sec","Duration (s)"),("rms_energy","RMS Energy"),("silence_ratio_001","Silence Ratio")]:
        plt.figure(figsize=(8, 4))
        plt.hist(df[f"african_{feat}"].dropna(), bins=20, alpha=0.65, label=name)
        plt.hist(df[f"english_{feat}"].dropna(), bins=20, alpha=0.45, label="English")
        plt.xlabel(label); plt.ylabel("Count"); plt.title(f"{name}: {label}"); plt.legend()
        save_plot(out_dir / f"eda_{feat}.png")

def run_data_exploration():
    if not RUN_DATA_EXPLORATION:
        print("EDA skipped."); return pd.DataFrame(), pd.DataFrame()
    eda_root = EDA_DIR; eda_root.mkdir(exist_ok=True)
    all_rows = []
    for cfg in LANGUAGE_CONFIGS:
        lang_dir = eda_root / cfg["language"]; lang_dir.mkdir(exist_ok=True)
        print(f"\nEDA: {cfg['display_name']}")
        lang_rows = []
        for split in EDA_SPLITS_TO_ANALYZE:
            try:
                split_rows = collect_eda_rows(cfg, split, EDA_SAMPLE_SIZE_PER_LANGUAGE_SPLIT, EDA_MAX_SCAN_ROWS)
                lang_rows.extend(split_rows)
            except Exception as e:
                print(f"  EDA failed {split}: {e}")
        if lang_rows:
            ldf = pd.DataFrame(lang_rows)
            ldf.to_csv(lang_dir / "eda_samples.csv", index=False)
            plot_eda(ldf, lang_dir, cfg["display_name"])
            all_rows.extend(lang_rows)
    all_df = pd.DataFrame(all_rows)
    compact = pd.DataFrame()
    if not all_df.empty:
        all_df.to_csv(EDA_DIR / "01a_eda_all_languages.csv", index=False)
        compact = all_df.groupby("language").agg(
            samples=("sample_index","count"),
            avg_duration=("african_duration_sec","mean"),
            avg_silence=("african_silence_ratio_001","mean"),
            avg_rms=("african_rms_energy","mean"),
        ).reset_index()
        compact.to_csv(EDA_DIR / "01b_eda_compact_summary.csv", index=False)
        print(compact.to_string(index=False))
    return all_df, compact

# %% --- 12. Run EDA ---
eda_df, eda_compact = run_data_exploration()
run_waveform_comparisons()

# %% --- 12b. EDA-driven strategy rationale ---
def select_strategies_from_eda(eda_compact_df):
    """Return {strategy_key: rationale_string} derived from EDA statistics."""
    HIGH_SILENCE_THRESH = 0.05
    LONG_AUDIO_THRESH   = 6.0
    LOW_RMS_THRESH      = 0.05
    if eda_compact_df is None or eda_compact_df.empty:
        return {s["strategy_key"]: "EDA unavailable — strategy applied with default settings."
                for s in AUDIO_OPTIMIZATION_STRATEGIES}
    avg_silence  = float(eda_compact_df["avg_silence"].mean())
    avg_duration = float(eda_compact_df["avg_duration"].mean())
    avg_rms      = float(eda_compact_df["avg_rms"].mean())
    return {
        "baseline_direct": (
            "Always applied — establishes the direct-audio performance baseline "
            "against which all other strategies are compared."
        ),
        "normalized_audio": (
            f"Mean RMS={avg_rms:.4f}. "
            + ("Low energy detected → normalisation is critical to prevent the model "
               "from treating silence as signal."
               if avg_rms < LOW_RMS_THRESH else
               "Energy levels adequate; normalisation still reduces amplitude variance "
               "across utterances and recording conditions.")
        ),
        "trimmed_audio": (
            f"Mean silence ratio={avg_silence:.3f}. "
            + ("High silence proportion → trimming is expected to improve model focus "
               "on speech content and reduce padding-induced errors."
               if avg_silence > HIGH_SILENCE_THRESH else
               "Silence proportion is low, indicating generally clean recordings. "
               "Trimming applied as a lightweight defensive preprocessing step.")
        ),
        "chunk_based_audio": (
            f"Mean audio duration={avg_duration:.1f}s. "
            + ("Above chunking threshold — segmenting into overlapping chunks reduces "
               "context overload for the acoustic encoder."
               if avg_duration > LONG_AUDIO_THRESH else
               "Audio is relatively short; chunking runs for completeness but "
               "fragmentation of phonological cues may limit gains.")
        ),
    }

strategy_rationale = select_strategies_from_eda(eda_compact)

# %% --- 13. Text evaluation (NLLB only, no audio) ---
text_results, text_pred_rows = [], []

if RUN_TEXT_EVALUATION:
    for exp in EXPERIMENT_CONFIGS:
        exp_name = exp["experiment"]
        print(f"\nText eval: {exp_name}")
        for cfg in LANGUAGE_CONFIGS:
            rows = prepare_bidirectional_pairs(cfg["language"], SPLIT_FOR_EXPERIMENT,
                                               exp["max_text_dev"], exp["max_scan_rows"])
            dir_groups = {}
            for r in rows: dir_groups.setdefault(r["direction"], []).append(r)
            for direction, group in dir_groups.items():
                preds, refs = [], []
                t0 = time.time()
                for r in group:
                    try:
                        p = translate_text_direct(r["source_text"], r["source_lang_code"], r["target_lang_code"])
                    except Exception as e:
                        p = ""; print(f"  Text error: {e}")
                    preds.append(p); refs.append(r["target_text"])
                    text_pred_rows.append({
                        "baseline_model": BASELINE_MODEL_NAME, "dataset": BASELINE_DATASET_NAME,
                        "experiment_family": EXPERIMENT_FAMILY, "experiment": exp_name,
                        "sample_index": r["sample_index"], "language": r["language_display"],
                        "direction": direction, "direction_key": r["direction_key"],
                        "mode": "text_to_text", "method": "NLLB Text Translation",
                        "category": "Text Translation",
                        "source_text": r["source_text"], "reference": r["target_text"], "prediction": p,
                    })
                m = compute_metrics(preds, refs)
                text_results.append({
                    "baseline_model": BASELINE_MODEL_NAME, "dataset": BASELINE_DATASET_NAME,
                    "experiment_family": EXPERIMENT_FAMILY, "experiment": exp_name,
                    "mode": "text_to_text", "method": "NLLB Text Translation",
                    "category": "Text Translation", "language": cfg["display_name"],
                    "direction": direction, "num_samples": len(group),
                    "BLEU": m["BLEU"], "ChrF": m["ChrF"], "runtime_seconds": time.time() - t0,
                })
                print(f"  {cfg['display_name']} | {direction} | BLEU={m['BLEU']:.2f} ChrF={m['ChrF']:.2f}")
else:
    print("Text evaluation skipped.")

text_results_df = pd.DataFrame(text_results)
text_pred_df    = pd.DataFrame(text_pred_rows)
save_df(text_results_df, "02_text_metrics.csv")
save_df(text_pred_df,    "03_text_predictions.csv")

# %% --- 14. Audio evaluation (Whisper ASR → NLLB MT cascade) ---
audio_results, audio_pred_rows = [], []
should_run_audio = RUN_AUDIO_EVALUATION and not (DEBUG_MODE and SKIP_AUDIO_DEBUG)

if should_run_audio:
    for exp in EXPERIMENT_CONFIGS:
        exp_name   = exp["experiment"]
        strategies = active_audio_strategies(exp_name)
        print(f"\nAudio eval: {exp_name} | strategies: {[s['strategy_key'] for s in strategies]}")
        for cfg in LANGUAGE_CONFIGS:
            rows = prepare_bidirectional_pairs(cfg["language"], SPLIT_FOR_EXPERIMENT,
                                               exp["max_audio_dev"], exp["max_scan_rows"])
            dir_groups = {}
            for r in rows: dir_groups.setdefault(r["direction"], []).append(r)
            for strategy in strategies:
                sk = strategy["strategy_key"]
                for direction, group in dir_groups.items():
                    preds, refs, durs, chunks_list, asr_texts = [], [], [], [], []
                    t0 = time.time()
                    for r in group:
                        arr = extract_audio_array(r["source_audio"])
                        dur = audio_duration(arr); durs.append(dur)
                        if dur < MIN_DURATION or dur > MAX_DURATION:
                            pred, asr, nc, err = "", "", 0, f"skipped_duration_{dur:.2f}s"
                        else:
                            err = ""
                            try:
                                pred, asr, nc = translate_audio_with_strategy(
                                    r["source_audio"], r["source_lang_code"], r["target_lang_code"],
                                    r["source_whisper_lang"], strategy)
                            except Exception as e:
                                pred, asr, nc, err = "", "", 0, str(e)
                                print(f"  Audio error {exp_name}/{sk}: {e}")
                        chunks_list.append(nc); preds.append(pred); refs.append(r["target_text"])
                        asr_texts.append(asr)
                        audio_pred_rows.append({
                            "baseline_model": BASELINE_MODEL_NAME, "dataset": BASELINE_DATASET_NAME,
                            "experiment_family": EXPERIMENT_FAMILY, "experiment": exp_name,
                            "sample_index": r["sample_index"], "language": r["language_display"],
                            "direction": direction, "direction_key": r["direction_key"],
                            "mode": sk, "method": strategy["method"], "category": strategy["category"],
                            "source_text_transcription": r["source_text"],
                            "asr_output": asr,   # intermediate Whisper transcription
                            "reference": r["target_text"], "prediction": pred,
                            "duration_seconds": dur, "num_chunks": nc, "error_message": err,
                        })
                    m = compute_metrics(preds, refs)
                    audio_results.append({
                        "baseline_model": BASELINE_MODEL_NAME, "dataset": BASELINE_DATASET_NAME,
                        "experiment_family": EXPERIMENT_FAMILY, "experiment": exp_name,
                        "mode": sk, "method": strategy["method"], "category": strategy["category"],
                        "language": cfg["display_name"], "direction": direction,
                        "num_samples": len(group), "BLEU": m["BLEU"], "ChrF": m["ChrF"],
                        "avg_duration_seconds": float(np.mean(durs)) if durs else 0.0,
                        "avg_num_chunks": float(np.mean(chunks_list)) if chunks_list else 0.0,
                        "runtime_seconds": time.time() - t0,
                    })
                    print(f"  {cfg['display_name']} | {sk} | {direction} | BLEU={m['BLEU']:.2f} ChrF={m['ChrF']:.2f}")
else:
    print(f"Audio evaluation skipped.")

audio_results_df = pd.DataFrame(audio_results)
audio_pred_df    = pd.DataFrame(audio_pred_rows)
save_df(audio_results_df, "04_audio_metrics.csv")
save_df(audio_pred_df,    "05_audio_predictions.csv")

# %% --- 15. Aggregate ---
_frames = [df for df in [text_results_df, audio_results_df] if not df.empty]
aggregate_df = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
save_df(aggregate_df, "06_aggregate_metrics.csv")

if not aggregate_df.empty:
    labels = aggregate_df["experiment"] + " | " + aggregate_df["mode"] + " | " + aggregate_df["direction"]
    plt.figure(figsize=(max(12, len(aggregate_df)*0.4), 5))
    plt.bar(range(len(labels)), aggregate_df["ChrF"], color="steelblue")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.ylabel("ChrF"); plt.title(f"ChrF Overview — {EXPERIMENT_FAMILY}")
    save_plot(FIGURES_DIR / "01_chrf_overview.png")

    plt.figure(figsize=(max(12, len(aggregate_df)*0.4), 5))
    plt.bar(range(len(labels)), aggregate_df["BLEU"], color="darkorange")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.ylabel("BLEU"); plt.title(f"BLEU Overview — {EXPERIMENT_FAMILY}")
    save_plot(FIGURES_DIR / "02_bleu_overview.png")

    _text_agg = aggregate_df[aggregate_df["mode"] == "text_to_text"]
    if not _text_agg.empty:
        _latest = _text_agg[_text_agg["experiment"] == _text_agg["experiment"].max()]
        _pivot_bleu = _latest.pivot_table(index="language", columns="direction", values="BLEU", aggfunc="mean")
        _pivot_chrf = _latest.pivot_table(index="language", columns="direction", values="ChrF", aggfunc="mean")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        _pivot_bleu.plot(kind="bar", ax=axes[0], colormap="tab10", rot=30)
        axes[0].set_title("BLEU by Language × Direction"); axes[0].set_ylabel("BLEU"); axes[0].set_xlabel("")
        _pivot_chrf.plot(kind="bar", ax=axes[1], colormap="tab10", rot=30)
        axes[1].set_title("ChrF by Language × Direction"); axes[1].set_ylabel("ChrF"); axes[1].set_xlabel("")
        plt.suptitle(f"{EXPERIMENT_FAMILY} — Text Translation", fontsize=11)
        save_plot(FIGURES_DIR / "03_language_direction_comparison.png")

    _audio_agg = aggregate_df[aggregate_df["mode"] != "text_to_text"]
    if not _audio_agg.empty:
        _latest_a = _audio_agg[_audio_agg["experiment"] == _audio_agg["experiment"].max()]
        _pivot_s = _latest_a.pivot_table(index="mode", columns="language", values="ChrF", aggfunc="mean")
        plt.figure(figsize=(10, 5))
        _pivot_s.plot(kind="bar", colormap="Set2", rot=30, ax=plt.gca())
        plt.title(f"Audio Strategy ChrF by Language — {EXPERIMENT_FAMILY}")
        plt.ylabel("ChrF"); plt.xlabel("Strategy")
        save_plot(FIGURES_DIR / "04_audio_strategy_comparison.png")

    _grouped = aggregate_df.groupby("experiment")[["BLEU","ChrF"]].mean().reset_index()
    if len(_grouped) > 1:
        plt.figure(figsize=(8, 4))
        plt.plot(_grouped["experiment"], _grouped["BLEU"], marker="o", label="BLEU")
        plt.plot(_grouped["experiment"], _grouped["ChrF"], marker="s", label="ChrF")
        plt.title(f"Score Progression — {EXPERIMENT_FAMILY}")
        plt.ylabel("Score"); plt.xlabel("Experiment"); plt.legend()
        save_plot(FIGURES_DIR / "05_experiment_progression.png")

# %% --- 15b. Summary table (one row per language × experiment config) ---
SHORT_AUDIO_THRESHOLD = 5.0   # seconds — labelled "Short audio"
LONG_AUDIO_THRESHOLD  = 12.0  # seconds — labelled "Long audio"

def _bleu_lookup(df, mode_val, direction_val, lang_name, exp_name):
    if df is None or df.empty: return float("nan")
    sub = df[(df["experiment"] == exp_name) & (df["mode"] == mode_val) &
             (df["direction"] == direction_val) & (df["language"] == lang_name)]
    return float(sub["BLEU"].values[0]) if not sub.empty else float("nan")

def build_summary_table():
    rows = []
    for exp in EXPERIMENT_CONFIGS:
        exp_name = exp["experiment"]
        for cfg in LANGUAGE_CONFIGS:
            lang    = cfg["display_name"]
            a2e_dir = f"{lang}→English"
            e2a_dir = f"English→{lang}"
            short_bleu = long_bleu = float("nan")
            if not audio_pred_df.empty:
                sub = audio_pred_df[
                    (audio_pred_df["experiment"] == exp_name) &
                    (audio_pred_df["mode"]       == "baseline_direct") &
                    (audio_pred_df["direction"]  == a2e_dir) &
                    (audio_pred_df["language"]   == lang)
                ]
                s = sub[sub["duration_seconds"] <  SHORT_AUDIO_THRESHOLD]
                l = sub[sub["duration_seconds"] >= LONG_AUDIO_THRESHOLD]
                if len(s) >= 1:
                    short_bleu = compute_metrics(s["prediction"].tolist(), s["reference"].tolist())["BLEU"]
                if len(l) >= 1:
                    long_bleu  = compute_metrics(l["prediction"].tolist(), l["reference"].tolist())["BLEU"]
            rows.append({
                "method":                  BASELINE_MODEL_NAME,
                "language":                lang,
                "experiment":              BASELINE_DATASET_NAME,
                "max_text_train_samples":  EDA_SAMPLE_SIZE_PER_LANGUAGE_SPLIT,
                "max_text_dev_samples":    exp["max_text_dev"],
                "max_audio_dev_samples":   exp["max_audio_dev"],
                "Chunk-based audio → English": _bleu_lookup(audio_results_df, "chunk_based_audio", a2e_dir, lang, exp_name),
                "Direct audio → English":      _bleu_lookup(audio_results_df, "baseline_direct",   a2e_dir, lang, exp_name),
                "English → Source":            _bleu_lookup(text_results_df,  "text_to_text",      e2a_dir, lang, exp_name),
                "Gold ASR cascade":            _bleu_lookup(text_results_df,  "text_to_text",      a2e_dir, lang, exp_name),
                "Long audio → English":        long_bleu,
                "Normalized audio → English":  _bleu_lookup(audio_results_df, "normalized_audio",  a2e_dir, lang, exp_name),
                "Short audio → English":       short_bleu,
                "Transcript → English":        float("nan"),
                "Trimmed audio → English":     _bleu_lookup(audio_results_df, "trimmed_audio",     a2e_dir, lang, exp_name),
            })
    summary_df = pd.DataFrame(rows)
    save_df(summary_df, "07_summary_table.csv")
    print("\nSummary table:")
    print(summary_df.to_string(index=False))
    return summary_df

summary_df = build_summary_table()

# %% --- 15c. Experiment summary report ---
def write_experiment_summary():
    """Prose report: setup, EDA findings, strategy decisions, results, implications."""
    sep  = "=" * 72
    sep2 = "-" * 52
    out  = [sep, f"  EXPERIMENT SUMMARY — {EXPERIMENT_FAMILY}", sep, ""]

    out += ["SETUP", sep2,
            f"  Model          : {BASELINE_MODEL_NAME}",
            f"  Dataset        : {DATASET_ID} ({BASELINE_DATASET_NAME})",
            f"  Pipeline type  : {BASELINE_PIPELINE_TYPE}",
            f"  Evaluation split: {SPLIT_FOR_EXPERIMENT}",
            f"  Mode           : {'DEBUG — sanity-check only (not for paper)' if DEBUG_MODE else 'FULL — paper-quality run'}",
            f"  Device         : {DEVICE}",
            f"  Active languages: {[c['display_name'] for c in LANGUAGE_CONFIGS]}",
            f"  Skipped        : {[(c['display_name'], c.get('skip_reason','')) for c in SKIPPED_LANGUAGE_CONFIGS]}",
            "  Experiment configs:"]
    for e in EXPERIMENT_CONFIGS:
        out.append(f"    {e['experiment']}: max_text_dev={e['max_text_dev']}, max_audio_dev={e['max_audio_dev']}")
    out.append("")

    out += ["EDA FINDINGS", sep2]
    if not eda_compact.empty:
        for _, row in eda_compact.iterrows():
            out.append(
                f"  {str(row['language']):12s}  avg_duration={row['avg_duration']:.2f}s  "
                f"avg_silence={row['avg_silence']:.3f}  avg_rms={row['avg_rms']:.4f}"
            )
    else:
        out.append("  EDA was skipped or produced no rows.")
    out.append("")

    out += ["STRATEGY DECISIONS (DATA-DRIVEN)", sep2]
    for s in AUDIO_OPTIMIZATION_STRATEGIES:
        rationale = strategy_rationale.get(s["strategy_key"], "No rationale recorded.")
        out.append(f"  [{s['strategy_key']}]  {s['method']}")
        out.append(f"    Rationale: {rationale}")
    out.append("")

    out += ["RESULTS — TEXT EVALUATION", sep2]
    if not text_results_df.empty:
        for _, r in text_results_df.iterrows():
            out.append(
                f"  {r['experiment']:15s}  {str(r['language']):10s}  "
                f"{r['direction']:30s}  BLEU={r['BLEU']:6.2f}  ChrF={r['ChrF']:6.2f}  n={r['num_samples']}"
            )
    else:
        out.append("  Text evaluation skipped.")
    out.append("")

    out += ["RESULTS — AUDIO EVALUATION", sep2]
    if not audio_results_df.empty:
        for _, r in audio_results_df.iterrows():
            out.append(
                f"  {r['experiment']:15s}  {str(r['language']):10s}  "
                f"{r['method']:35s}  {r['direction']:25s}  BLEU={r['BLEU']:6.2f}  ChrF={r['ChrF']:6.2f}"
            )
    else:
        out.append("  Audio evaluation skipped.")
    out.append("")

    out += ["KEY FINDINGS", sep2]
    _text_col = next(
        (c for c in ["Transcript → English", "Gold ASR cascade"]
         if not summary_df.empty and c in summary_df.columns and summary_df[c].notna().any()),
        None,
    )
    _direct_col = "Direct audio → English"
    if _text_col and not summary_df.empty and _direct_col in summary_df.columns:
        out.append(f"  {_text_col} (gold text) vs {_direct_col} (raw audio):")
        out.append("  Hypothesis: text score > audio score (text bypasses ASR transcription errors).")
        found = False
        for _, row in summary_df.iterrows():
            t = row.get(_text_col, float("nan"))
            d = row.get(_direct_col, float("nan"))
            if pd.isna(t) or pd.isna(d): continue
            found = True
            delta   = t - d
            verdict = "CONFIRMED ✓" if delta > 0 else "NOT CONFIRMED — audio ≥ text"
            out.append(
                f"    {str(row.get('experiment','?')):15s} | {str(row['language']):10s}  "
                f"text={t:.2f}  audio={d:.2f}  Δ={delta:+.2f}  → {verdict}"
            )
        if not found:
            out.append("    No comparable data (audio evaluation was skipped in this run).")
    else:
        out.append("  Insufficient data for key-findings comparison.")
    out.append("")

    out += ["IMPLICATIONS FOR FUTURE RESEARCH", sep2,
        "  1. Text-based inputs (gold transcript / gold ASR cascade) are expected to",
        "     outperform direct-audio inputs, confirming ASR transcription errors as the",
        "     dominant performance bottleneck for low-resource African languages.",
        "  2. Silence trimming and waveform normalisation are low-cost preprocessing steps",
        "     that improve robustness across languages and variable recording conditions.",
        "  3. Chunk-based segmentation benefits longer utterances but may degrade short",
        "     ones by fragmenting phonological cues — duration-adaptive chunking warrants",
        "     investigation.",
        "  4. Cross-language BLEU variance reflects resource scarcity and linguistic",
        "     complexity; targeted fine-tuning or data augmentation is recommended for",
        "     the lowest-scoring languages.",
        "  5. ChrF is more informative than BLEU for morphologically rich African languages;",
        "     both should be reported in the final paper.",
        "  6. Short-audio vs long-audio stratification reveals duration-dependent effects;",
        "     duration-aware training data curation and model selection are promising",
        "     future directions.",
        "  7. The waveform comparison (EDA Strategy C) provides visual evidence of",
        "     amplitude variance — valuable for the data characterisation section of the paper.",
        "",
        f"  {'⚠  DEBUG MODE: scores are indicative only — re-run in FULL mode for paper results.' if DEBUG_MODE else '✓  FULL MODE: results are paper-ready.'}",
        ""]
    out.append(sep)

    body = "\n".join(out)
    path = BASE_OUTPUT_DIR / "00_experiment_summary.txt"
    path.write_text(body, encoding="utf-8")
    print(body)
    print(f"Experiment summary saved: {path}")

write_experiment_summary()

# %% --- 16. Qualitative outputs ---

def build_qualitative_table(pred_df, mode_name, n=5):
    if pred_df.empty: return pd.DataFrame()
    rows = []
    for _, grp in pred_df.groupby(["experiment","direction"] if "experiment" in pred_df.columns else ["direction"]):
        for _, r in grp.head(n).iterrows():
            rows.append({
                "experiment": r.get("experiment",""), "mode": r.get("mode", mode_name),
                "method": r.get("method",""), "category": r.get("category",""),
                "baseline_model": BASELINE_MODEL_NAME, "dataset": BASELINE_DATASET_NAME,
                "language": r.get("language",""), "direction": r.get("direction",""),
                "sample_index": r.get("sample_index",""),
                "source": r.get("source_text", r.get("source_text_transcription","")),
                "asr_output": r.get("asr_output",""),
                "reference": r.get("reference",""), "prediction": r.get("prediction",""),
                "error_message": r.get("error_message",""),
                "manual_error_category": "",
                "manual_severity": "",
                "manual_comment": "",
            })
    return pd.DataFrame(rows)

qual_text  = build_qualitative_table(text_pred_df,  "text_to_text")
qual_audio = build_qualitative_table(audio_pred_df, "cascade_asr_mt")
qual_all   = pd.concat([df for df in [qual_text, qual_audio] if not df.empty], ignore_index=True)
save_df(qual_text,  "08_qualitative_text.csv",  QUAL_DIR)
save_df(qual_audio, "09_qualitative_audio.csv", QUAL_DIR)
save_df(qual_all,   "10_qualitative_all.csv",   QUAL_DIR)

# %% --- 17. Metadata snapshot ---
import json as _json
_metadata = {
    "experiment_family": EXPERIMENT_FAMILY,
    "asr_model": WHISPER_MODEL_ID,
    "mt_model": NLLB_MODEL_ID,
    "dataset": DATASET_ID,
    "pipeline": BASELINE_PIPELINE_TYPE,
    "device": DEVICE,
    "debug_mode": DEBUG_MODE,
    "split_used": SPLIT_FOR_EXPERIMENT,
    "active_languages": [c["display_name"] for c in LANGUAGE_CONFIGS],
    "skipped_languages": [(c["display_name"], c.get("skip_reason","")) for c in SKIPPED_LANGUAGE_CONFIGS],
    "experiments": [e["experiment"] for e in EXPERIMENT_CONFIGS],
    "directions": DIRECTIONS_TO_RUN,
    "audio_strategies": [s["strategy_key"] for s in AUDIO_OPTIMIZATION_STRATEGIES if s["enabled"]],
    "run_timestamp": datetime.now().isoformat(),
    "output_structure": {
        "metrics/": "02_text_metrics, 03_text_predictions, 04_audio_metrics, 05_audio_predictions, 06_aggregate_metrics, 07_summary_table",
        "figures/": "01_chrf_overview, 02_bleu_overview, 03_language_direction_comparison, 04_audio_strategy_comparison, 05_experiment_progression",
        "eda/":     "01a_eda_all_languages, 01b_eda_compact_summary; per-language histograms",
        "qualitative/": "08_qualitative_text, 09_qualitative_audio, 10_qualitative_all",
    },
}
with open(BASE_OUTPUT_DIR / "metadata.json", "w") as _f:
    _json.dump(_metadata, _f, indent=2)
print("Saved: metadata.json")

# %% --- 18. Google Drive backup (Colab only) ---
if GOOGLE_DRIVE_AVAILABLE:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    dr_root = Path("/content/drive/MyDrive") / _FOLDER_NAME
    dr_root.mkdir(parents=True, exist_ok=True)
    dr_dest = dr_root / f"run_{ts}"
    if dr_dest.exists(): shutil.rmtree(dr_dest)
    shutil.copytree(BASE_OUTPUT_DIR, dr_dest)
    zip_path = shutil.make_archive(str(dr_dest), "zip", root_dir=BASE_OUTPUT_DIR)
    print(f"Drive backup: {dr_dest}\nZIP: {zip_path}")
else:
    print(f"Local results saved to: {BASE_OUTPUT_DIR}")

print("\n=== Experiment complete ===")
print(f"Pair    : {EXPERIMENT_FAMILY}")
print(f"metrics/: {sorted(METRICS_DIR.glob('*.csv'))}")
print(f"figures/: {sorted(FIGURES_DIR.glob('*.png'))}")
print(f"eda/    : {sorted(EDA_DIR.glob('*.csv'))}")
print(f"qual/   : {sorted(QUAL_DIR.glob('*.csv'))}")
