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
# # LinguoMT — African-Celtic + Whisper-large-v3 + NLLB-200 (Cascade)
#
# **Dataset:** McGill-NLP/african_celtic_dataset  (IWSLT 2026 African & Celtic ST track)
# **ASR model:** openai/whisper-large-v3
# **MT model:**  facebook/nllb-200-distilled-600M
# **Pipeline:**  Audio → Whisper (transcribe) → NLLB (translate)
#
# Dataset languages: Igbo, Yoruba, Hausa
# Active languages:  Yoruba, Hausa
# Disabled: Igbo — Whisper-large-v3 does not support it
#
# IMPORTANT: The African-Celtic dataset is associated with IWSLT 2026.
# If it requires authentication, run: huggingface-cli login
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
#                           sentencepiece accelerate jiwer pandas pyarrow torchcodec
# Note: torchcodec is required by datasets>=4.x to decode Audio columns.
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U",
        "transformers>=4.40", "datasets", "sacrebleu", "librosa", "soundfile",
        "sentencepiece", "accelerate", "jiwer", "pandas", "pyarrow>=15.0.0"],
        check=True)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "torchcodec",
        "--extra-index-url", "https://download.pytorch.org/whl/cu121"], check=False)
    print("Dependencies installed.")

# %% --- 2b. HuggingFace Hub authentication ---
# The african_celtic_dataset (IWSLT 2026) may require HF authentication.
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

from datasets import load_dataset, load_dataset_builder, Audio
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
DATASET_ID         = "McGill-NLP/african_celtic_dataset"
ENGLISH_NLLB_CODE  = "eng_Latn"
ENGLISH_WHISPER_LANG = "english"

BASELINE_MODEL_NAME    = "Whisper-large-v3 + NLLB-200-600M"
BASELINE_DATASET_NAME  = "African-Celtic"
BASELINE_PIPELINE_TYPE = "cascade_asr_mt"
EXPERIMENT_FAMILY      = "AfricanCeltic__WhisperLargeV3_NLLB200"

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
SELECTED_LANGUAGE_PAIRS = ["yoruba", "hausa"]
SPLIT_FOR_EXPERIMENT = "dev"    # African-Celtic has only "train" and "dev" splits
MIN_DURATION = 1.0
MAX_DURATION = 20.0

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Experiment sizes ──────────────────────────────────────────────
if DEBUG_MODE:
    # dev split has ~5500 items; English items appear first (~1375 rows) so we must scan
    # past them to find African items with matching text_ids — 3000 rows covers yoruba/hausa blocks
    EXPERIMENT_CONFIGS = [{"experiment": "debug", "max_text_dev": 8,   "max_audio_dev": 3,  "max_scan_rows": 3000}]
else:
    EXPERIMENT_CONFIGS = [
        {"experiment": "Experiment_1", "max_text_dev": 50,  "max_audio_dev": 10, "max_scan_rows": 1000},
        {"experiment": "Experiment_2", "max_text_dev": 100, "max_audio_dev": 30, "max_scan_rows": 1500},
        {"experiment": "Experiment_3", "max_text_dev": 200, "max_audio_dev": 50, "max_scan_rows": 2000},
    ]

# ── EDA settings ──────────────────────────────────────────────────
EDA_SPLITS_TO_ANALYZE              = ["train", "dev"]  # African-Celtic only has "train" and "dev"
EDA_SAMPLE_SIZE_PER_LANGUAGE_SPLIT = 10 if DEBUG_MODE else 200
EDA_MAX_SCAN_ROWS                  = 3000 if DEBUG_MODE else 4000  # must scan past English block
EDA_AUDIO_EXAMPLES_PER_LANGUAGE    = 1 if DEBUG_MODE else 2

# ── Audio optimization strategies ─────────────────────────────────
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
# African-Celtic dataset schema (confirmed via metadata):
#   - Single HuggingFace config: "default"
#   - Language field: item["language"] — values like "igbo", "yoruba", "hausa", "english"
#   - Source text field: item["text"]
#   - Alignment key: item["text_id"] — shared across all language variants of the same sentence
#   - English items (same split, language="english") are the translation references
#
# Whisper supports: Yoruba ("yoruba"), Hausa ("hausa"). Does NOT support Igbo.
# NLLB-200 supports all three.
#
# language_value: exact string in item["language"]
ENGLISH_LANGUAGE_VALUE = "english"  # value of item["language"] for English items

ALL_LANGUAGE_CONFIGS = [
    {
        "language": "yoruba", "language_value": "yoruba", "display_name": "Yoruba",
        "whisper_lang": "yoruba",  "nllb_src_code": "yor_Latn",
        "dataset_supported": True, "model_supported": True,  "enabled": True,
        "skip_reason": "",
    },
    {
        "language": "hausa",  "language_value": "hausa",  "display_name": "Hausa",
        "whisper_lang": "hausa",   "nllb_src_code": "hau_Latn",
        "dataset_supported": True, "model_supported": True,  "enabled": True,
        "skip_reason": "",
    },
    {
        "language": "igbo",   "language_value": "igbo",   "display_name": "Igbo",
        "whisper_lang": None,      "nllb_src_code": "ibo_Latn",
        "dataset_supported": True, "model_supported": False, "enabled": False,
        "skip_reason": "Whisper-large-v3 does not support Igbo",
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
print(f"Skipped: {[(c['display_name'], c.get('skip_reason','')) for c in SKIPPED_LANGUAGE_CONFIGS]}")
print(f"Output: {BASE_OUTPUT_DIR}")

# %% --- 5. Google Drive setup (Colab only) ---
GOOGLE_DRIVE_AVAILABLE = False
if IN_COLAB and _colab_drive is not None:
    _colab_drive.mount("/content/drive")
    GOOGLE_DRIVE_AVAILABLE = True
    print("Google Drive mounted.")
else:
    print("Local run — Google Drive backup will be skipped.")

# %% --- 6. Dataset schema discovery ---
# Run this cell first to inspect the actual column names and splits.

def discover_dataset_schema(split=None):
    print(f"\n=== Schema discovery: {DATASET_ID} / default ===")
    try:
        from datasets import get_dataset_config_names
        print(f"Available configs: {get_dataset_config_names(DATASET_ID)}")
    except Exception as e:
        print(f"Could not list configs: {e}")
    for try_split in ([split] if split else ["dev", "train"]):
        try:
            ds = load_dataset(DATASET_ID, "default", split=try_split, streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
            item = next(iter(ds))
            print(f"\nSplit '{try_split}' columns: {list(item.keys())}")
            for k, v in item.items():
                if k != "audio":
                    print(f"  {k}: {repr(v)[:120]}")
                else:
                    print(f"  audio: {repr(v)[:120]}")
            return try_split, list(item.keys())
        except Exception as e:
            print(f"  Split '{try_split}' failed: {e}")
    print("Could not load any split.")
    return None, []

# Uncomment to run schema discovery before the full experiment:
# discover_dataset_schema()

# %% --- 7. Model loading ---
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
nllb_model     = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID).to(DEVICE)
nllb_model.eval()
print(f"Loaded NLLB on {DEVICE}")

# %% --- 8. Audio preprocessing helpers ---

def get_item_text(item):
    """Return the transcription text from an African-Celtic item (field: 'text')."""
    return str(item.get("text") or "").strip()

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
    try: audio = np.asarray(audio, dtype=np.float32)
    except Exception: audio = np.array([], dtype=np.float32)
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

# %% --- 9. Dataset loading (African-Celtic) ---
# Single "default" config; all languages mixed in one split.
# African and English items share text_id — alignment is done by groupby join.

def prepare_bidirectional_pairs(language_key, split=SPLIT_FOR_EXPERIMENT, max_samples=50, max_scan_rows=5000):
    """Align African + English items by text_id and return translation pair dicts."""
    cfg = LANGUAGE_LOOKUP.get(language_key)
    if cfg is None:
        return []
    lang_val = cfg["language_value"]

    stream = load_dataset(DATASET_ID, "default", split=split, streaming=True)

    id_to_items: dict = {}  # text_id -> {language_value: item}
    scanned = 0
    for item in stream:
        if scanned >= max_scan_rows:
            break
        lang = item.get("language", "")
        tid  = item.get("text_id",  "")
        if lang in (lang_val, ENGLISH_LANGUAGE_VALUE) and tid:
            if tid not in id_to_items:
                id_to_items[tid] = {}
            if lang not in id_to_items[tid]:
                id_to_items[tid][lang] = item
        scanned += 1

    rows = []
    pair_count = 0
    for tid, lang_items in id_to_items.items():
        if pair_count >= max_samples:
            break
        afr_item = lang_items.get(lang_val)
        eng_item = lang_items.get(ENGLISH_LANGUAGE_VALUE)
        if not afr_item or not eng_item:
            continue
        afr_text = normalize_text(get_item_text(afr_item))
        eng_text = normalize_text(get_item_text(eng_item))
        if not afr_text or not eng_text:
            continue
        afr_audio = afr_item.get("audio")
        eng_audio = eng_item.get("audio")
        if "african_to_english" in DIRECTIONS_TO_RUN:
            rows.append({
                "sample_index": pair_count, "language": language_key,
                "language_display": cfg["display_name"],
                "direction_key": "african_to_english",
                "direction": f"{cfg['display_name']}→English",
                "source_lang_code": cfg["nllb_src_code"], "target_lang_code": ENGLISH_NLLB_CODE,
                "source_whisper_lang": cfg["whisper_lang"], "target_whisper_lang": ENGLISH_WHISPER_LANG,
                "source_text": afr_text, "target_text": eng_text,
                "source_audio": afr_audio, "target_audio": eng_audio,
            })
        if "english_to_african" in DIRECTIONS_TO_RUN:
            rows.append({
                "sample_index": pair_count, "language": language_key,
                "language_display": cfg["display_name"],
                "direction_key": "english_to_african",
                "direction": f"English→{cfg['display_name']}",
                "source_lang_code": ENGLISH_NLLB_CODE, "target_lang_code": cfg["nllb_src_code"],
                "source_whisper_lang": ENGLISH_WHISPER_LANG, "target_whisper_lang": cfg["whisper_lang"],
                "source_text": eng_text, "target_text": afr_text,
                "source_audio": eng_audio, "target_audio": afr_audio,
            })
        pair_count += 1
    return rows

def save_df(df, filename, out_dir=None):
    p = (out_dir or METRICS_DIR) / filename
    df.to_csv(p, index=False); print("Saved:", p); return p

# %% --- 10. Translation functions (Whisper ASR → NLLB MT cascade) ---

def transcribe_whisper(audio, whisper_lang):
    """Transcribe audio to text in source language using Whisper."""
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
    """Translate text using NLLB-200. Codes e.g. yor_Latn, eng_Latn."""
    text = str(text).strip()
    if not text: return ""
    nllb_tokenizer.src_lang = src_nllb_code
    inputs = nllb_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    tgt_id = nllb_tokenizer.convert_tokens_to_ids(tgt_nllb_code)
    with torch.no_grad():
        out = nllb_model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=256)
    return nllb_tokenizer.decode(out[0], skip_special_tokens=True)

def translate_text_direct(text, source_lang_code, target_lang_code):
    """Text-only translation via NLLB."""
    return translate_nllb(text, source_lang_code, target_lang_code)

def translate_audio_direct(audio_obj, source_lang_code, target_lang_code,
                            whisper_lang=None, normalize_audio=True, improved=True):
    """Cascade: Whisper ASR → NLLB MT. Returns (translation, asr_transcription)."""
    audio = extract_audio_array(audio_obj)
    if improved:        audio = trim_silence(audio)
    if normalize_audio: audio = normalize_audio_waveform(audio)
    audio = ensure_min_audio_length(audio)
    transcription = transcribe_whisper(audio, whisper_lang or ENGLISH_WHISPER_LANG)
    translation   = translate_nllb(transcription, source_lang_code, target_lang_code)
    return translation, transcription

def prepare_audio_for_strategy(audio_obj, strategy):
    audio = extract_audio_array(audio_obj)
    if strategy.get("trim"):      audio = trim_silence(audio)
    if strategy.get("normalize"): audio = normalize_audio_waveform(audio)
    return ensure_min_audio_length(audio).astype(np.float32)

def translate_audio_with_strategy(audio_obj, source_lang_code, target_lang_code,
                                   source_whisper_lang, strategy):
    audio = prepare_audio_for_strategy(audio_obj, strategy)
    if strategy.get("chunk"):
        chunks = split_audio_into_chunks(audio) or [audio]
        transcriptions = []
        for chunk in chunks:
            t = transcribe_whisper(chunk, source_whisper_lang)
            if normalize_text(t): transcriptions.append(normalize_text(t))
        joined = " ".join(transcriptions)
        translation = translate_nllb(joined, source_lang_code, target_lang_code)
        return translation, joined, len(chunks)
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

# %% --- 11. Metrics ---

def compute_metrics(predictions, references):
    preds = [str(p).strip() for p in predictions]
    refs  = [str(r).strip() for r in references]
    if not preds or not refs: return {"BLEU": 0.0, "ChrF": 0.0}
    return {
        "BLEU": float(sacrebleu.corpus_bleu(preds, [refs]).score),
        "ChrF": float(sacrebleu.corpus_chrf(preds, [refs]).score),
    }

# %% --- 12. EDA helpers ---

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
    lang_val = lang_cfg["language_value"]
    try:
        stream = load_dataset(DATASET_ID, "default", split=split, streaming=True)
        stream = stream.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"  EDA load failed {split}: {e}"); return rows

    id_to_items: dict = {}
    scanned = 0
    for item in stream:
        if scanned >= max_scan_rows:
            break
        lang = item.get("language", "")
        tid  = item.get("text_id",  "")
        if lang in (lang_val, ENGLISH_LANGUAGE_VALUE) and tid:
            if tid not in id_to_items:
                id_to_items[tid] = {}
            if lang not in id_to_items[tid]:
                id_to_items[tid][lang] = item
        scanned += 1

    for pair_idx, (tid, lang_items) in enumerate(id_to_items.items()):
        if len(rows) >= max_samples:
            break
        afr_item = lang_items.get(lang_val)
        eng_item = lang_items.get(ENGLISH_LANGUAGE_VALUE)
        if not afr_item or not eng_item:
            continue
        afr_text = normalize_text(get_item_text(afr_item))
        eng_text = normalize_text(get_item_text(eng_item))
        if not afr_text or not eng_text:
            continue
        afr_audio_obj = afr_item.get("audio") or {}
        if isinstance(afr_audio_obj, dict) and afr_audio_obj.get("array") is not None:
            afr_arr = resample_audio(
                afr_audio_obj.get("array", np.array([])),
                afr_audio_obj.get("sampling_rate", TARGET_SR),
            )
        else:
            afr_arr = np.array([])
        aq = compute_audio_quality_features(afr_arr, TARGET_SR)
        rows.append({
            "sample_index": pair_idx, "split": split,
            "language": lang_cfg["display_name"], "language_key": lang_cfg["language"],
            "african_text": afr_text, "english_text": eng_text,
            "african_words": len(afr_text.split()), "english_words": len(eng_text.split()),
            **{f"african_{k}": v for k, v in aq.items()},
        })
    return rows

def plot_eda(df, out_dir, name):
    if df.empty: return
    for feat, label in [("duration_sec","Duration (s)"),("rms_energy","RMS Energy"),("silence_ratio_001","Silence Ratio")]:
        col = f"african_{feat}"
        if col not in df.columns: continue
        plt.figure(figsize=(8, 4))
        plt.hist(df[col].dropna(), bins=20, alpha=0.8, label=name)
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
                print(f"  {split}: {len(split_rows)} rows")
            except Exception as e:
                print(f"  EDA failed {split}: {e}")
        if lang_rows:
            ldf = pd.DataFrame(lang_rows)
            ldf.to_csv(lang_dir / "eda_samples.csv", index=False)
            plot_eda(ldf, lang_dir, cfg["display_name"])
            all_rows.extend(lang_rows)
    all_df  = pd.DataFrame(all_rows)
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

# %% --- 13. Run EDA ---
eda_df, eda_compact = run_data_exploration()

# %% --- 14. Text evaluation (NLLB) ---
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

# %% --- 15. Audio evaluation (Whisper → NLLB cascade) ---
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
                    preds, refs, durs, chunks_list, asr_list = [], [], [], [], []
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
                        asr_list.append(asr)
                        audio_pred_rows.append({
                            "baseline_model": BASELINE_MODEL_NAME, "dataset": BASELINE_DATASET_NAME,
                            "experiment_family": EXPERIMENT_FAMILY, "experiment": exp_name,
                            "sample_index": r["sample_index"], "language": r["language_display"],
                            "direction": direction, "direction_key": r["direction_key"],
                            "mode": sk, "method": strategy["method"], "category": strategy["category"],
                            "source_text_transcription": r["source_text"],
                            "asr_output": asr,
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
    print("Audio evaluation skipped.")

audio_results_df = pd.DataFrame(audio_results)
audio_pred_df    = pd.DataFrame(audio_pred_rows)
save_df(audio_results_df, "04_audio_metrics.csv")
save_df(audio_pred_df,    "05_audio_predictions.csv")

# %% --- 16. Aggregate ---
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

# %% --- 17. Qualitative outputs ---

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

# %% --- 18. Metadata snapshot ---
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
        "metrics/": "02_text_metrics, 03_text_predictions, 04_audio_metrics, 05_audio_predictions, 06_aggregate_metrics",
        "figures/": "01_chrf_overview, 02_bleu_overview, 03_language_direction_comparison, 04_audio_strategy_comparison, 05_experiment_progression",
        "eda/":     "01a_eda_all_languages, 01b_eda_compact_summary; per-language histograms",
        "qualitative/": "08_qualitative_text, 09_qualitative_audio, 10_qualitative_all",
    },
}
with open(BASE_OUTPUT_DIR / "metadata.json", "w") as _f:
    _json.dump(_metadata, _f, indent=2)
print("Saved: metadata.json")

# %% --- 19. Google Drive backup (Colab only) ---
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
print(f"Pair : {EXPERIMENT_FAMILY}")
print(f"Files: {list(BASE_OUTPUT_DIR.glob('*.csv'))}")
