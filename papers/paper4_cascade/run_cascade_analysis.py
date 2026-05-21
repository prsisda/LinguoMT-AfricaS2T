#!/usr/bin/env python3
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
# # LinguoMT — Paper 4 Cascade Analysis
#
# Runs Paper 4 (LinguoMT-Cascade) specific analyses:
#   1. Oracle cascade  — NLLB on gold FLEURS transcripts (text MT ceiling)
#   2. Error propagation — corrupt ASR transcripts at controlled WER rates
#   3. Break-even WER  — regression to find where cascade matches E2E
#   4. Latency profile — median/P95 latency + peak VRAM for each architecture
#
# Prerequisites:
#   - Run Paper 1 (benchmark) experiments first to get E2E BLEU numbers
#   - Set E2E_BLEU_BY_LANG below with values from results/paper1_benchmark/results.csv
#
# Outputs (written to RESULTS_DIR):
#   oracle_cascade_metrics.csv
#   error_propagation_metrics.csv
#   breakeven_metrics.csv
#   latency_metrics.csv
#
# Run on Colab: set DEBUG_MODE = False, then Run All
# Run locally : python papers/paper4_cascade/run_cascade_analysis.py

# %% --- bootstrap ---
import sys, subprocess
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

try:
    import google.colab  # noqa
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

if _IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U",
        "transformers>=4.40", "datasets", "sacrebleu", "librosa", "soundfile",
        "sentencepiece", "accelerate", "jiwer", "pandas==2.2.2", "pyarrow>=15.0.0",
        "protobuf",
    ], check=True)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "torchcodec",
        "--extra-index-url", "https://download.pytorch.org/whl/cu121"], check=False)
    # Pull latest code
    subprocess.run(["git", "-C", str(_REPO), "pull", "--ff-only"], check=False)

from framework import detect_environment
ENV = detect_environment()

# %% --- imports ---
import random, warnings
import numpy as np
import torch
from transformers import (
    AutoProcessor, SeamlessM4Tv2Model,
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
)
from framework import (
    detect_model_capabilities, select_supported_languages, get_adapter_type,
    DatasetCache, StepMonitor, create_run_dirs,
    zip_run_outputs, drive_backup, colab_download, mount_google_drive,
)
from framework.cascade_analysis import (
    run_oracle_cascade,
    run_error_propagation,
    run_breakeven_analysis,
    run_latency_profile,
)
warnings.filterwarnings("ignore")

# %% --- configuration ---
DEBUG_MODE = True   # True → fast test (few samples)  |  False → full paper run

# ── E2E BLEU by language (fill from results/paper1_benchmark/results.csv) ─────
# These are the SeamlessM4T-v2 zero-shot BLEU scores used to compute break-even.
# Key: language_key (lowercase), Value: BLEU score (float)
E2E_BLEU_BY_LANG = {
    "yoruba":  None,   # fill from fleurs_seamless.yoruba.bleu in paper1 results
    "igbo":    None,   # fill from fleurs_seamless.igbo.bleu
    "swahili": None,   # fill from fleurs_seamless.swahili.bleu
}
# Example (fill these after running Paper 1):
# E2E_BLEU_BY_LANG = {"yoruba": 8.3, "igbo": 5.1, "swahili": 14.7}

# ── Models ────────────────────────────────────────────────────────────────────
SEAMLESS_ID   = "facebook/seamless-m4t-v2-large"
WHISPER_ID    = "openai/whisper-large-v3"
NLLB_ID       = "facebook/nllb-200-distilled-600M"
DATASET_ID    = "google/fleurs"
SPLIT         = "validation"
MANUAL_LANGUAGES_SEAMLESS = ["yoruba", "igbo", "swahili"]
MANUAL_LANGUAGES_WHISPER  = ["yoruba", "hausa", "swahili"]
SEED          = 42

RESULTS_DIR = _REPO / "results" / "paper4_cascade"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_WARMUP  = 3 if DEBUG_MODE else 10
N_MEASURE = 10 if DEBUG_MODE else 100
MAX_PAIRS = 20 if DEBUG_MODE else 200

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# %% --- environment ---
DEVICE = ENV["device"]
drive_available = mount_google_drive(ENV.get("_colab_drive")) if _IN_COLAB else False

monitor = StepMonitor(RESULTS_DIR / "monitoring")
monitor.step("Paper 4 cascade analysis started", f"debug={DEBUG_MODE}")

# %% --- load models ---
monitor.step("Loading models")

# SeamlessM4T (E2E — for latency profiling)
seamless_proc  = AutoProcessor.from_pretrained(SEAMLESS_ID)
seamless_model = SeamlessM4Tv2Model.from_pretrained(SEAMLESS_ID).to(DEVICE).eval()
print("SeamlessM4T-v2 loaded.")

# Whisper + NLLB (cascade — for oracle, error propagation, latency)
whisper_proc   = WhisperProcessor.from_pretrained(WHISPER_ID)
whisper_model  = WhisperForConditionalGeneration.from_pretrained(WHISPER_ID).to(DEVICE).eval()
nllb_tok       = AutoTokenizer.from_pretrained(NLLB_ID)
nllb_model     = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID).to(DEVICE).eval()
print("Whisper + NLLB loaded.")

# %% --- language configs ---
seamless_caps  = detect_model_capabilities(SEAMLESS_ID)
whisper_caps   = detect_model_capabilities("whisper_nllb")

seamless_langs = select_supported_languages(
    seamless_caps, DATASET_ID, processor=seamless_proc,
    max_langs=3, manual_override=MANUAL_LANGUAGES_SEAMLESS,
)
whisper_langs  = select_supported_languages(
    whisper_caps, DATASET_ID, processor=whisper_proc,
    max_langs=3, manual_override=MANUAL_LANGUAGES_WHISPER,
)
print(f"SeamlessM4T languages: {[c['display'] for c in seamless_langs]}")
print(f"Whisper+NLLB languages: {[c['display'] for c in whisper_langs]}")

# %% --- data caches ---
monitor.step("Building data caches")
cache_dir = _REPO / "FLEURS__WhisperNLLB" / "cache"

whisper_cache = DatasetCache(
    dataset_id=DATASET_ID, adapter_type=get_adapter_type(DATASET_ID),
    language_configs=whisper_langs, split=SPLIT,
    max_pairs=MAX_PAIRS, max_scan_rows=5000 if DEBUG_MODE else 50000,
    cache_dir=cache_dir, force_rerun=False,
)
whisper_cache.build(monitor)
print("Cache built:", whisper_cache.stats())

# %% --- model callables ---

def nllb_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    text = str(text).strip()
    if not text:
        return ""
    nllb_tok.src_lang = src_lang
    inp    = nllb_tok(text, return_tensors="pt").to(DEVICE)
    tgt_id = nllb_tok.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        ids = nllb_model.generate(**inp, forced_bos_token_id=tgt_id, max_new_tokens=256)
    return nllb_tok.decode(ids[0], skip_special_tokens=True)


def whisper_asr(audio_arr: np.ndarray, whisper_lang_code: str) -> str:
    inp    = whisper_proc(audio_arr, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
    forced = whisper_proc.get_decoder_prompt_ids(language=whisper_lang_code, task="transcribe")
    with torch.no_grad():
        ids = whisper_model.generate(inp, forced_decoder_ids=forced)
    return whisper_proc.batch_decode(ids, skip_special_tokens=True)[0]


def cascade_translate_audio(audio_arr: np.ndarray, src_lang: str, tgt_lang: str) -> tuple[str, int]:
    nllb_to_whisper = {c["nllb_code"]: c["whisper_code"]
                       for c in whisper_langs if c.get("nllb_code") and c.get("whisper_code")}
    whisper_code = nllb_to_whisper.get(src_lang)
    if not whisper_code:
        return "", 1
    transcript  = whisper_asr(audio_arr, whisper_code)
    translation = nllb_translate(transcript, src_lang, tgt_lang)
    return translation, 1


def seamless_translate_audio(audio_arr: np.ndarray, src_lang: str, tgt_lang: str) -> tuple[str, int]:
    inp = seamless_proc(audio=audio_arr, sampling_rate=16000, src_lang=src_lang,
                        return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        tok = seamless_model.generate(**inp, tgt_lang=tgt_lang, generate_speech=False)
    return seamless_proc.decode(tok[0].tolist()[0], skip_special_tokens=True), 1

# %% --- 1. Oracle cascade ---
monitor.step("Running oracle cascade")
oracle_df = run_oracle_cascade(
    translate_text_fn=nllb_translate,
    dev_cache=whisper_cache,
    lang_cfgs=whisper_langs,
    caps=whisper_caps,
    dirs=type("D", (), {"metrics": RESULTS_DIR})(),
    monitor=monitor,
)
print(oracle_df.to_string(index=False) if not oracle_df.empty else "  No oracle results")

# %% --- 2. Error propagation ---
monitor.step("Running error propagation")
wer_rates = [0.0, 0.1, 0.2, 0.3] if DEBUG_MODE else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
error_df = run_error_propagation(
    translate_text_fn=nllb_translate,
    dev_cache=whisper_cache,
    lang_cfgs=whisper_langs,
    caps=whisper_caps,
    dirs=type("D", (), {"metrics": RESULTS_DIR})(),
    monitor=monitor,
    wer_rates=wer_rates,
    seed=SEED,
)
print(error_df.to_string(index=False) if not error_df.empty else "  No error propagation results")

# %% --- 3. Break-even WER ---
monitor.step("Running break-even analysis")
if not error_df.empty:
    valid_e2e = {k: v for k, v in E2E_BLEU_BY_LANG.items() if v is not None}
    if not valid_e2e:
        print("  WARNING: E2E_BLEU_BY_LANG is empty — fill in Paper 1 results first.")
        print("  Skipping break-even analysis.")
    else:
        breakeven_df = run_breakeven_analysis(
            error_prop_df=error_df,
            e2e_bleu_by_lang=valid_e2e,
            dirs=type("D", (), {"metrics": RESULTS_DIR})(),
        )
        print(breakeven_df.to_string(index=False) if not breakeven_df.empty else "  No break-even results")
else:
    print("  Skipping break-even: no error propagation results")

# %% --- 4. Latency profile ---
monitor.step("Running latency profile")
latency_df = run_latency_profile(
    translate_audio_fn=seamless_translate_audio,
    asr_fn=whisper_asr,
    translate_text_fn=nllb_translate,
    dev_cache=whisper_cache,
    lang_cfgs=whisper_langs,
    caps=whisper_caps,
    dirs=type("D", (), {"metrics": RESULTS_DIR})(),
    monitor=monitor,
    n_warmup=N_WARMUP,
    n_measure=N_MEASURE,
)
print(latency_df.to_string(index=False) if not latency_df.empty else "  No latency results")

# %% --- done ---
monitor.step("Paper 4 cascade analysis complete")
print(f"\nAll results written to: {RESULTS_DIR}")
print("Next step: python papers/extract_results.py paper4_cascade")

if _IN_COLAB:
    import shutil
    zip_path = _REPO / "paper4_cascade_analysis.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(RESULTS_DIR))
    from google.colab import files
    files.download(str(zip_path))
