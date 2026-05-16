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
# # LinguoMT — FLEURS + SeamlessM4T-v2-Large
#
# **Dataset:** google/fleurs  |  **Model:** facebook/seamless-m4t-v2-large
#
# Active languages: Igbo, Yoruba, Swahili
# Disabled: Hausa (SeamlessM4T-v2 does not support it), Wolof (not supported)
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
        "sentencepiece", "accelerate", "jiwer", "pandas", "pyarrow>=15.0.0"],
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
from transformers import AutoProcessor, SeamlessM4Tv2Model

warnings.filterwarnings("ignore")

# %% --- 4. Configuration ---

# ── Model / Dataset ───────────────────────────────────────────────
MODEL_ID              = "facebook/seamless-m4t-v2-large"
DATASET_ID            = "google/fleurs"
ENGLISH_CONFIG        = "en_us"
BASELINE_MODEL_NAME   = "SeamlessM4T-v2 Large"
BASELINE_DATASET_NAME = "FLEURS"
BASELINE_PIPELINE_TYPE = "end_to_end_speech_translation"
EXPERIMENT_FAMILY     = "FLEURS__SeamlessM4Tv2_Large"

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
SELECTED_LANGUAGE_PAIRS = ["igbo", "yoruba", "swahili"]
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
EDA_SPLITS_TO_ANALYZE               = ["train", "validation", "test"]
EDA_SAMPLE_SIZE_PER_LANGUAGE_SPLIT  = 25 if DEBUG_MODE else 200
EDA_MAX_SCAN_ROWS                   = 100 if DEBUG_MODE else 1000
EDA_AUDIO_EXAMPLES_PER_LANGUAGE     = 1 if DEBUG_MODE else 2

# ── Audio optimization strategies ─────────────────────────────────
CHUNK_SECONDS         = 6
CHUNK_OVERLAP_SECONDS = 1
RUN_EXPENSIVE_AUDIO_STRATEGIES_ONLY_IN_EXP1 = False

AUDIO_OPTIMIZATION_STRATEGIES = [
    {"strategy_key": "baseline_direct",   "method": "Direct audio → English",     "category": "Direct Speech Translation", "enabled": True,  "normalize": False, "trim": False, "chunk": False, "expensive": False},
    {"strategy_key": "normalized_audio",  "method": "Normalized audio → English",  "category": "Audio Normalization",       "enabled": True,  "normalize": True,  "trim": False, "chunk": False, "expensive": False},
    {"strategy_key": "trimmed_audio",     "method": "Trimmed audio → English",     "category": "Silence Trimming",          "enabled": True,  "normalize": True,  "trim": True,  "chunk": False, "expensive": True},
    {"strategy_key": "chunk_based_audio", "method": "Chunk-based audio → English", "category": "Audio Segmentation",        "enabled": True,  "normalize": True,  "trim": False, "chunk": True,  "expensive": True},
]

# ── Language configs ──────────────────────────────────────────────
# SeamlessM4T-v2 supports: Igbo (ibo), Yoruba (yor), Swahili (swh)
# NOT supported: Hausa, Wolof
ALL_LANGUAGE_CONFIGS = [
    {"language": "igbo",    "fleurs_config": "ig_ng", "source_lang_code": "ibo", "display_name": "Igbo",    "dataset_supported": True,  "model_supported": True,  "enabled": True},
    {"language": "yoruba",  "fleurs_config": "yo_ng", "source_lang_code": "yor", "display_name": "Yoruba",  "dataset_supported": True,  "model_supported": True,  "enabled": True},
    {"language": "swahili", "fleurs_config": "sw_ke", "source_lang_code": "swh", "display_name": "Swahili", "dataset_supported": True,  "model_supported": True,  "enabled": True},
    {"language": "hausa",   "fleurs_config": "ha_ng", "source_lang_code": None,  "display_name": "Hausa",   "dataset_supported": True,  "model_supported": False, "enabled": False},
    {"language": "wolof",   "fleurs_config": "wo_sn", "source_lang_code": None,  "display_name": "Wolof",   "dataset_supported": True,  "model_supported": False, "enabled": False},
]

# ── Filter to active languages ────────────────────────────────────
if RUN_FULL_GRID:
    _candidates = [c for c in ALL_LANGUAGE_CONFIGS if c.get("enabled")]
else:
    _sel = set(SELECTED_LANGUAGE_PAIRS)
    _candidates = [c for c in ALL_LANGUAGE_CONFIGS if c["language"] in _sel]

LANGUAGE_CONFIGS, SKIPPED_LANGUAGE_CONFIGS = [], []
for _cfg in _candidates:
    _reasons = []
    if not _cfg.get("dataset_supported"): _reasons.append("not in FLEURS")
    if not _cfg.get("model_supported"):   _reasons.append("not supported by SeamlessM4T-v2")
    if _cfg.get("source_lang_code") is None: _reasons.append("missing SeamlessM4T language code")
    if _reasons:
        SKIPPED_LANGUAGE_CONFIGS.append({**_cfg, "skip_reason": "; ".join(_reasons)})
    else:
        LANGUAGE_CONFIGS.append(_cfg)
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

print(f"Model : {MODEL_ID}")
print(f"Dataset: {DATASET_ID}")
print(f"Device : {DEVICE} | CUDA: {torch.cuda.is_available()}")
print(f"Active : {[c['display_name'] for c in LANGUAGE_CONFIGS]}")
print(f"Skipped: {[(c['display_name'], c['skip_reason']) for c in SKIPPED_LANGUAGE_CONFIGS]}")
print(f"Output : {BASE_OUTPUT_DIR}")

# %% --- 5. Google Drive setup (Colab only) ---
GOOGLE_DRIVE_AVAILABLE = False
if IN_COLAB and _colab_drive is not None:
    _colab_drive.mount("/content/drive")
    GOOGLE_DRIVE_AVAILABLE = True
    print("Google Drive mounted.")
else:
    print("Local run — Google Drive backup will be skipped.")

# %% --- 6. Model loading ---
if not LANGUAGE_CONFIGS:
    raise RuntimeError("No active supported language pairs. Check language config flags.")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model     = SeamlessM4Tv2Model.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print(f"Loaded {BASELINE_MODEL_NAME} on {DEVICE}")

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
    """Return flat float32 waveform from a HuggingFace audio dict or raw array."""
    if isinstance(audio_obj, dict):
        sr    = audio_obj.get("sampling_rate", target_sr)
        audio = audio_obj.get("array")
    else:
        sr, audio = target_sr, audio_obj
    if audio is None:
        return np.zeros(int(MIN_DURATION * target_sr), dtype=np.float32)
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
        if len(chunk) >= int(MIN_DURATION * sr):
            chunks.append(chunk.astype(np.float32))
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
            rows.append({"sample_index": idx, "language": language_key, "language_display": cfg["display_name"],
                         "direction_key": "african_to_english", "direction": f"{cfg['display_name']}→English",
                         "source_lang_code": cfg["source_lang_code"], "target_lang_code": "eng",
                         "source_text": afr_text, "target_text": eng_text,
                         "source_audio": ai["audio"], "target_audio": ei["audio"]})
        if "english_to_african" in DIRECTIONS_TO_RUN:
            rows.append({"sample_index": idx, "language": language_key, "language_display": cfg["display_name"],
                         "direction_key": "english_to_african", "direction": f"English→{cfg['display_name']}",
                         "source_lang_code": "eng", "target_lang_code": cfg["source_lang_code"],
                         "source_text": eng_text, "target_text": afr_text,
                         "source_audio": ei["audio"], "target_audio": ai["audio"]})
        count += 1
    return rows

def save_df(df, filename, out_dir=None):
    p = (out_dir or METRICS_DIR) / filename
    df.to_csv(p, index=False)
    print("Saved:", p)
    return p

# %% --- 9. Translation functions (SeamlessM4T-v2) ---

def translate_text_direct(text, source_lang_code, target_lang_code):
    text = str(text).strip()
    if not text: return ""
    inputs = processor(text=text, src_lang=source_lang_code, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        tokens = model.generate(**inputs, tgt_lang=target_lang_code, generate_speech=False)
    return processor.decode(tokens[0].tolist()[0], skip_special_tokens=True)

def translate_audio_direct(audio_obj, source_lang_code, target_lang_code, normalize_audio=True, improved=True):
    audio = extract_audio_array(audio_obj)
    if improved:       audio = trim_silence(audio)
    if normalize_audio: audio = normalize_audio_waveform(audio)
    audio = ensure_min_audio_length(audio)
    inputs = processor(audio=audio, sampling_rate=TARGET_SR, src_lang=source_lang_code, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        tokens = model.generate(**inputs, tgt_lang=target_lang_code, generate_speech=False)
    return processor.decode(tokens[0].tolist()[0], skip_special_tokens=True)

def prepare_audio_for_strategy(audio_obj, strategy):
    audio = extract_audio_array(audio_obj)
    if strategy.get("trim"):      audio = trim_silence(audio)
    if strategy.get("normalize"): audio = normalize_audio_waveform(audio)
    return ensure_min_audio_length(audio).astype(np.float32)

def translate_audio_with_strategy(audio_obj, source_lang_code, target_lang_code, strategy):
    audio = prepare_audio_for_strategy(audio_obj, strategy)
    if strategy.get("chunk"):
        chunks = split_audio_into_chunks(audio) or [audio]
        preds = []
        for chunk in chunks:
            p = translate_audio_direct(chunk, source_lang_code, target_lang_code, normalize_audio=False, improved=False)
            if normalize_text(p): preds.append(normalize_text(p))
        return " ".join(preds), len(chunks)
    return translate_audio_direct(audio, source_lang_code, target_lang_code, normalize_audio=False, improved=False), 1

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
        "duration_sec":        len(audio) / float(sr),
        "mean_abs_energy":     float(np.mean(np.abs(audio))),
        "rms_energy":          float(np.sqrt(np.mean(audio**2))),
        "peak_amplitude":      float(np.max(np.abs(audio))),
        "silence_ratio_001":   float(np.mean(np.abs(audio) < 0.01)),
        "clipping_ratio_099":  float(np.mean(np.abs(audio) >= 0.99)),
        "zero_crossing_rate":  float(np.mean(librosa.feature.zero_crossing_rate(audio)[0])),
        "dynamic_range":       float(np.percentile(audio, 95) - np.percentile(audio, 5)),
    }

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if IN_COLAB: plt.show()
    plt.close()
    print("Saved figure:", path)

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
        aq = compute_audio_quality_features(aa, TARGET_SR)
        eq = compute_audio_quality_features(ea, TARGET_SR)
        rows.append({
            "sample_index": idx, "split": split,
            "language": lang_cfg["display_name"], "language_key": lang_cfg["language"],
            "african_text": at, "english_text": et,
            "african_words": len(at.split()), "english_words": len(et.split()),
            **{f"african_{k}": v for k, v in aq.items()},
            **{f"english_{k}": v for k, v in eq.items()},
        })
    return rows

def plot_eda(df, out_dir, name):
    if df.empty: return
    for feat, label in [("duration_sec", "Duration (s)"), ("rms_energy", "RMS Energy"), ("silence_ratio_001", "Silence Ratio")]:
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
    if not all_df.empty:
        all_df.to_csv(EDA_DIR / "01a_eda_all_languages.csv", index=False)
        compact = all_df.groupby("language").agg(
            samples=("sample_index", "count"),
            avg_duration=("african_duration_sec", "mean"),
            avg_silence=("african_silence_ratio_001", "mean"),
            avg_rms=("african_rms_energy", "mean"),
        ).reset_index()
        compact.to_csv(EDA_DIR / "01b_eda_compact_summary.csv", index=False)
        print(compact.to_string(index=False))
    return all_df, compact if not all_df.empty else pd.DataFrame()

# %% --- 12. Run EDA ---
eda_df, eda_compact = run_data_exploration()

# %% --- 13. Text evaluation ---
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
                        p = ""; print(f"  Text error {exp_name}/{direction}/{r['sample_index']}: {e}")
                    preds.append(p); refs.append(r["target_text"])
                    text_pred_rows.append({
                        "baseline_model": BASELINE_MODEL_NAME, "dataset": BASELINE_DATASET_NAME,
                        "experiment_family": EXPERIMENT_FAMILY, "experiment": exp_name,
                        "sample_index": r["sample_index"], "language": r["language_display"],
                        "direction": direction, "direction_key": r["direction_key"],
                        "mode": "text_to_text", "method": "Transcript ↔ Text Translation",
                        "category": "Text Translation",
                        "source_text": r["source_text"], "reference": r["target_text"], "prediction": p,
                    })
                m = compute_metrics(preds, refs)
                text_results.append({
                    "baseline_model": BASELINE_MODEL_NAME, "dataset": BASELINE_DATASET_NAME,
                    "experiment_family": EXPERIMENT_FAMILY, "experiment": exp_name,
                    "mode": "text_to_text", "method": "Transcript ↔ Text Translation",
                    "category": "Text Translation", "language": cfg["display_name"],
                    "direction": direction, "num_samples": len(group),
                    "BLEU": m["BLEU"], "ChrF": m["ChrF"],
                    "runtime_seconds": time.time() - t0,
                })
                print(f"  {cfg['display_name']} | {direction} | BLEU={m['BLEU']:.2f} ChrF={m['ChrF']:.2f} n={len(group)}")
else:
    print("Text evaluation skipped.")

text_results_df   = pd.DataFrame(text_results)
text_pred_df      = pd.DataFrame(text_pred_rows)
save_df(text_results_df, "02_text_metrics.csv")
save_df(text_pred_df,    "03_text_predictions.csv")

# %% --- 14. Audio evaluation ---
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
                    preds, refs, durs, chunks_list = [], [], [], []
                    t0 = time.time()
                    for r in group:
                        arr = extract_audio_array(r["source_audio"])
                        dur = audio_duration(arr)
                        durs.append(dur)
                        if dur < MIN_DURATION or dur > MAX_DURATION:
                            pred, nc, err = "", 0, f"skipped_duration_{dur:.2f}s"
                        else:
                            err = ""
                            try:
                                pred, nc = translate_audio_with_strategy(
                                    r["source_audio"], r["source_lang_code"], r["target_lang_code"], strategy)
                            except Exception as e:
                                pred, nc, err = "", 0, str(e)
                                print(f"  Audio error {exp_name}/{sk}/{r['sample_index']}: {e}")
                        chunks_list.append(nc); preds.append(pred); refs.append(r["target_text"])
                        audio_pred_rows.append({
                            "baseline_model": BASELINE_MODEL_NAME, "dataset": BASELINE_DATASET_NAME,
                            "experiment_family": EXPERIMENT_FAMILY, "experiment": exp_name,
                            "sample_index": r["sample_index"], "language": r["language_display"],
                            "direction": direction, "direction_key": r["direction_key"],
                            "mode": sk, "method": strategy["method"], "category": strategy["category"],
                            "source_text_transcription": r["source_text"],
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
    print(f"Audio evaluation skipped. RUN_AUDIO_EVALUATION={RUN_AUDIO_EVALUATION}")

audio_results_df = pd.DataFrame(audio_results)
audio_pred_df    = pd.DataFrame(audio_pred_rows)
save_df(audio_results_df, "04_audio_metrics.csv")
save_df(audio_pred_df,    "05_audio_predictions.csv")

# %% --- 15. Aggregate results ---
_frames = [df for df in [text_results_df, audio_results_df] if not df.empty]
aggregate_df = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
save_df(aggregate_df, "06_aggregate_metrics.csv")

if not aggregate_df.empty:
    # Figure 1 — ChrF by experiment × mode × direction
    plt.figure(figsize=(max(12, len(aggregate_df)*0.4), 5))
    labels = aggregate_df["experiment"] + " | " + aggregate_df["mode"] + " | " + aggregate_df["direction"]
    plt.bar(range(len(labels)), aggregate_df["ChrF"], color="steelblue")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.ylabel("ChrF"); plt.title(f"ChrF Overview — {EXPERIMENT_FAMILY}")
    save_plot(FIGURES_DIR / "01_chrf_overview.png")

    # Figure 2 — BLEU by experiment × mode × direction
    plt.figure(figsize=(max(12, len(aggregate_df)*0.4), 5))
    plt.bar(range(len(labels)), aggregate_df["BLEU"], color="darkorange")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.ylabel("BLEU"); plt.title(f"BLEU Overview — {EXPERIMENT_FAMILY}")
    save_plot(FIGURES_DIR / "02_bleu_overview.png")

    # Figure 3 — Language × Direction: BLEU and ChrF side-by-side (text eval only, latest experiment)
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

    # Figure 4 — Strategy comparison (audio eval)
    _audio_agg = aggregate_df[aggregate_df["mode"] != "text_to_text"]
    if not _audio_agg.empty:
        _latest_a = _audio_agg[_audio_agg["experiment"] == _audio_agg["experiment"].max()]
        _pivot_s = _latest_a.pivot_table(index="mode", columns="language", values="ChrF", aggfunc="mean")
        plt.figure(figsize=(10, 5))
        _pivot_s.plot(kind="bar", colormap="Set2", rot=30, ax=plt.gca())
        plt.title(f"Audio Strategy ChrF by Language — {EXPERIMENT_FAMILY}")
        plt.ylabel("ChrF"); plt.xlabel("Strategy")
        save_plot(FIGURES_DIR / "04_audio_strategy_comparison.png")

    # Figure 5 — Experiment progression (BLEU and ChrF across Exp1→2→3)
    _grouped = aggregate_df.groupby("experiment")[["BLEU","ChrF"]].mean().reset_index()
    if len(_grouped) > 1:
        plt.figure(figsize=(8, 4))
        plt.plot(_grouped["experiment"], _grouped["BLEU"],  marker="o", label="BLEU")
        plt.plot(_grouped["experiment"], _grouped["ChrF"],  marker="s", label="ChrF")
        plt.title(f"Score Progression — {EXPERIMENT_FAMILY}")
        plt.ylabel("Score"); plt.xlabel("Experiment"); plt.legend()
        save_plot(FIGURES_DIR / "05_experiment_progression.png")

# %% --- 16. Qualitative outputs ---

def build_qualitative_table(pred_df, mode_name, n=5):
    if pred_df.empty: return pd.DataFrame()
    rows = []
    for _, grp in pred_df.groupby(["experiment", "direction"] if "experiment" in pred_df.columns else ["direction"]):
        for _, r in grp.head(n).iterrows():
            rows.append({
                "experiment": r.get("experiment", ""), "mode": r.get("mode", mode_name),
                "method": r.get("method", ""), "category": r.get("category", ""),
                "baseline_model": BASELINE_MODEL_NAME, "dataset": BASELINE_DATASET_NAME,
                "language": r.get("language", ""), "direction": r.get("direction", ""),
                "sample_index": r.get("sample_index", ""),
                "source": r.get("source_text", r.get("source_text_transcription", "")),
                "reference": r.get("reference", ""), "prediction": r.get("prediction", ""),
                "error_message": r.get("error_message", ""),
                "manual_error_category": "",
                "manual_severity": "",
                "manual_comment": "",
            })
    return pd.DataFrame(rows)

qual_text  = build_qualitative_table(text_pred_df,  "text_to_text")
qual_audio = build_qualitative_table(audio_pred_df, "speech_to_text")
qual_all   = pd.concat([df for df in [qual_text, qual_audio] if not df.empty], ignore_index=True)
save_df(qual_text,  "08_qualitative_text.csv",  QUAL_DIR)
save_df(qual_audio, "09_qualitative_audio.csv", QUAL_DIR)
save_df(qual_all,   "10_qualitative_all.csv",   QUAL_DIR)

# %% --- 17. Metadata snapshot ---
import json as _json
_metadata = {
    "experiment_family": EXPERIMENT_FAMILY,
    "model": MODEL_ID,
    "dataset": DATASET_ID,
    "pipeline": BASELINE_PIPELINE_TYPE,
    "device": DEVICE,
    "debug_mode": DEBUG_MODE,
    "split_used": SPLIT_FOR_EXPERIMENT,
    "active_languages": [c["display_name"] for c in LANGUAGE_CONFIGS],
    "skipped_languages": [(c["display_name"], c["skip_reason"]) for c in SKIPPED_LANGUAGE_CONFIGS],
    "experiments": [e["experiment"] for e in EXPERIMENT_CONFIGS],
    "directions": DIRECTIONS_TO_RUN,
    "audio_strategies": [s["strategy_key"] for s in AUDIO_OPTIMIZATION_STRATEGIES if s["enabled"]],
    "run_timestamp": datetime.now().isoformat(),
    "output_structure": {
        "metrics/": "CSVs: 02_text_metrics, 03_text_predictions, 04_audio_metrics, 05_audio_predictions, 06_aggregate_metrics",
        "figures/": "PNG charts: 01_chrf_overview, 02_bleu_overview, 03_language_direction_comparison, 04_audio_strategy_comparison, 05_experiment_progression",
        "eda/":     "CSVs: 01a_eda_all_languages, 01b_eda_compact_summary; per-language histograms",
        "qualitative/": "CSVs: 08_qualitative_text, 09_qualitative_audio, 10_qualitative_all (annotate manually)",
    },
}
with open(BASE_OUTPUT_DIR / "metadata.json", "w") as _f:
    _json.dump(_metadata, _f, indent=2)
print("Saved: metadata.json")

# %% --- 18. Google Drive backup (Colab only) ---
if GOOGLE_DRIVE_AVAILABLE:
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    dr_root  = Path("/content/drive/MyDrive") / _FOLDER_NAME
    dr_root.mkdir(parents=True, exist_ok=True)
    dr_dest  = dr_root / f"run_{ts}"
    if dr_dest.exists(): shutil.rmtree(dr_dest)
    shutil.copytree(BASE_OUTPUT_DIR, dr_dest)
    zip_path = shutil.make_archive(str(dr_dest), "zip", root_dir=BASE_OUTPUT_DIR)
    print(f"Drive backup: {dr_dest}")
    print(f"ZIP:          {zip_path}")
else:
    print(f"Local results saved to: {BASE_OUTPUT_DIR}")

# %% --- 19. Summary ---
print("\n=== Experiment complete ===")
print(f"Pair    : {EXPERIMENT_FAMILY}")
print(f"metrics/: {sorted(METRICS_DIR.glob('*.csv'))}")
print(f"figures/: {sorted(FIGURES_DIR.glob('*.png'))}")
print(f"eda/    : {sorted(EDA_DIR.glob('*.csv'))}")
print(f"qual/   : {sorted(QUAL_DIR.glob('*.csv'))}")
