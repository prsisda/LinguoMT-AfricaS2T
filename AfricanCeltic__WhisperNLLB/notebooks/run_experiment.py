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
# # LinguoMT — African-Celtic + Whisper-large-v3 + NLLB-200-600M
#
# Cascade ASR+MT pipeline on the African-Celtic dataset.
# Whisper transcribes; NLLB translates.  Languages must be supported by both.
#
# Run locally : python run_experiment.py
# Run on Colab: set DEBUG_MODE below, then Run All

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

from framework import detect_environment, install_colab_dependencies
ENV = detect_environment()

# %% --- imports ---
import random, warnings
import numpy as np
import torch
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
)
from framework import (
    detect_model_capabilities, select_supported_languages, get_adapter_type,
    DatasetCache, StepMonitor, create_run_dirs, save_config,
    zip_run_outputs, drive_backup, colab_download, mount_google_drive,
    ExperimentRunner, RunConfig, default_experiment_configs,
)
warnings.filterwarnings("ignore")

# %% --- configuration ---
DEBUG_MODE       = True
FAST_MODE        = False
SKIP_AUDIO_DEBUG = True
FORCE_RERUN      = False
RUN_FULL_GRID    = True   # False → only Experiment_1 in full mode
WHISPER_ID       = "openai/whisper-large-v3"
NLLB_ID          = "facebook/nllb-200-600M"
MODEL_ID         = "whisper_nllb"
DATASET_ID       = "McGill-NLP/african_celtic_dataset"
MODEL_NAME       = "Whisper-large-v3 + NLLB-600M"
DATASET_NAME     = "African-Celtic"
EXPERIMENT_FAMILY= "AfricanCeltic__WhisperNLLB"
SPLIT            = "dev"
MANUAL_LANGUAGES = None   # None = auto | e.g. ["yoruba", "hausa"]
SEED             = 42

if FAST_MODE: DEBUG_MODE = True
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# %% --- capabilities + languages ---
DEVICE = ENV["device"]
caps   = detect_model_capabilities(MODEL_ID)
print(f"Loading processors: {WHISPER_ID}, {NLLB_ID}")
whisper_proc = WhisperProcessor.from_pretrained(WHISPER_ID)
nllb_tok     = AutoTokenizer.from_pretrained(NLLB_ID)
lang_cfgs    = select_supported_languages(caps, DATASET_ID, processor=whisper_proc,
                                           max_langs=3, manual_override=MANUAL_LANGUAGES)
if not lang_cfgs:
    raise RuntimeError("No supported languages found.")
print(f"Languages : {[c['display'] for c in lang_cfgs]}")
print(f"Strategies: {caps.enabled_strategies}  |  Device: {DEVICE}")

# %% --- models ---
whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_ID).to(DEVICE)
whisper_model.eval()
nllb_model    = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID).to(DEVICE)
nllb_model.eval()
print("Models loaded.")

# %% --- output setup ---
dirs    = create_run_dirs("whispernllb", "african_celtic", DEBUG_MODE, ENV["in_colab"], script_path=Path(__file__))
monitor = StepMonitor(dirs.monitoring)
monitor.step("Experiment started", f"mode={'DEBUG' if DEBUG_MODE else 'FULL'}")
drive_available = mount_google_drive(ENV["_colab_drive"]) if ENV["in_colab"] else False

# %% --- callables ---
_nllb_to_whisper = {c["nllb_code"]: c["whisper_code"]
                    for c in lang_cfgs if c.get("nllb_code") and c.get("whisper_code")}

def _asr(audio_arr: np.ndarray, whisper_code: str) -> str:
    inp = whisper_proc(audio_arr, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
    forced = whisper_proc.get_decoder_prompt_ids(language=whisper_code, task="transcribe")
    with torch.no_grad():
        ids = whisper_model.generate(inp, forced_decoder_ids=forced)
    return whisper_proc.batch_decode(ids, skip_special_tokens=True)[0]

def _mt(text: str, nllb_src: str, nllb_tgt: str) -> str:
    text = str(text).strip()
    if not text: return ""
    nllb_tok.src_lang = nllb_src
    inp = nllb_tok(text, return_tensors="pt").to(DEVICE)
    tgt_id = nllb_tok.lang_code_to_id[nllb_tgt]
    with torch.no_grad():
        ids = nllb_model.generate(**inp, forced_bos_token_id=tgt_id, max_new_tokens=256)
    return nllb_tok.decode(ids[0], skip_special_tokens=True)


def translate_text(text, src_lang, tgt_lang):
    return _mt(text, src_lang, tgt_lang)

def translate_audio(audio_arr, src_lang, tgt_lang):
    # src_lang is nllb_code; derive whisper_code for ASR
    wcode = _nllb_to_whisper.get(src_lang)
    if not wcode:
        return "", 1
    transcript = _asr(audio_arr, wcode)
    return _mt(transcript, src_lang, tgt_lang), 1

def asr_transcribe(audio_arr, src_lang):
    # src_lang is whisper_code (asr_lang_attr)
    return _asr(audio_arr, src_lang)

# %% --- data cache ---
exp_cfgs  = default_experiment_configs(DEBUG_MODE)
max_pairs = max(e["max_text_dev"] for e in exp_cfgs)
cache = DatasetCache(
    dataset_id=DATASET_ID, adapter_type=get_adapter_type(DATASET_ID),
    language_configs=lang_cfgs, split=SPLIT,
    max_pairs=max_pairs, max_scan_rows=6000, force_rerun=FORCE_RERUN,
)
cache.build(monitor)
print("Cache:", cache.stats())

# %% --- run ---
ExperimentRunner(
    config=RunConfig(
        model_name=MODEL_NAME, dataset_name=DATASET_NAME,
        experiment_family=EXPERIMENT_FAMILY, model_id=MODEL_ID, dataset_id=DATASET_ID,
        split=SPLIT, debug_mode=DEBUG_MODE, fast_mode=FAST_MODE,
        skip_audio_debug=SKIP_AUDIO_DEBUG, force_rerun=FORCE_RERUN,
        run_full_grid=RUN_FULL_GRID,
        in_colab=ENV["in_colab"], eda_sample_size=25 if DEBUG_MODE else 200,
        directions=["source_to_english", "english_to_source"],
    ),
    capabilities=caps, data_cache=cache, language_configs=lang_cfgs,
    dirs=dirs, monitor=monitor, experiment_configs=exp_cfgs,
    translate_text_fn=translate_text, translate_audio_fn=translate_audio, asr_fn=asr_transcribe,
).run()

# %% --- archive ---
save_config(dirs, {
    "asr_model": WHISPER_ID, "mt_model": NLLB_ID,
    "dataset_id": DATASET_ID, "experiment_family": EXPERIMENT_FAMILY,
    "debug_mode": DEBUG_MODE, "device": DEVICE, "split": SPLIT,
    "languages": [c["language_key"] for c in lang_cfgs],
    "capabilities": caps.enabled_strategies,
})
zip_run_outputs(dirs)
drive_backup(dirs, drive_available)
if ENV["in_colab"]:
    colab_download(dirs)
