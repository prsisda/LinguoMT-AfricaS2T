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
# # LinguoMT — African-Celtic + SeamlessM4T-v2-Large
#
# End-to-end speech translation on the African-Celtic dataset.
# Supported African languages auto-detected from both model card and dataset.
#
# Run locally : python run_experiment.py
# Run on Colab: set DEBUG_MODE below, then Run All

# %% --- bootstrap ---
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from framework import detect_environment, install_colab_dependencies
ENV = detect_environment()
if ENV["in_colab"]:
    install_colab_dependencies()

# %% --- imports ---
import random, warnings
import numpy as np
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model
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
MODEL_ID         = "facebook/seamless-m4t-v2-large"
DATASET_ID       = "McGill-NLP/african_celtic_dataset"
MODEL_NAME       = "SeamlessM4T-v2 Large"
DATASET_NAME     = "African-Celtic"
EXPERIMENT_FAMILY= "AfricanCeltic__SeamlessM4Tv2_Large"
SPLIT            = "dev"
MANUAL_LANGUAGES = None   # None = auto | e.g. ["igbo", "yoruba", "hausa"]
SEED             = 42

if FAST_MODE: DEBUG_MODE = True
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# %% --- capabilities + languages ---
DEVICE = ENV["device"]
caps   = detect_model_capabilities(MODEL_ID)
print(f"Loading processor: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
lang_cfgs = select_supported_languages(caps, DATASET_ID, processor=processor,
                                        max_langs=3, manual_override=MANUAL_LANGUAGES)
if not lang_cfgs:
    raise RuntimeError("No supported languages found.")
print(f"Languages : {[c['display'] for c in lang_cfgs]}")
print(f"Strategies: {caps.enabled_strategies}  |  Device: {DEVICE}")

# %% --- model ---
model = SeamlessM4Tv2Model.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print("Model loaded.")

# %% --- output setup ---
dirs    = create_run_dirs("seamlessm4t", "african_celtic", DEBUG_MODE, ENV["in_colab"], script_path=Path(__file__))
monitor = StepMonitor(dirs.monitoring)
monitor.step("Experiment started", f"mode={'DEBUG' if DEBUG_MODE else 'FULL'}")
drive_available = mount_google_drive(ENV["_colab_drive"]) if ENV["in_colab"] else False

# %% --- callables ---
def translate_text(text, src_lang, tgt_lang):
    text = str(text).strip()
    if not text: return ""
    inp = processor(text=text, src_lang=src_lang, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        tok = model.generate(**inp, tgt_lang=tgt_lang, generate_speech=False)
    return processor.decode(tok[0].tolist()[0], skip_special_tokens=True)

def translate_audio(audio_arr, src_lang, tgt_lang):
    inp = processor(audio=audio_arr, sampling_rate=16000, src_lang=src_lang, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        tok = model.generate(**inp, tgt_lang=tgt_lang, generate_speech=False)
    return processor.decode(tok[0].tolist()[0], skip_special_tokens=True), 1

def asr_transcribe(audio_arr, src_lang):
    inp = processor(audio=audio_arr, sampling_rate=16000, src_lang=src_lang, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        tok = model.generate(**inp, tgt_lang=src_lang, generate_speech=False)
    return processor.decode(tok[0].tolist()[0], skip_special_tokens=True)

# %% --- data cache ---
exp_cfgs  = default_experiment_configs(DEBUG_MODE)
max_pairs = max(e["max_text_dev"] for e in exp_cfgs)
cache = DatasetCache(
    dataset_id=DATASET_ID, adapter_type=get_adapter_type(DATASET_ID),
    language_configs=lang_cfgs, split=SPLIT,
    max_pairs=max_pairs, max_scan_rows=5000, force_rerun=FORCE_RERUN,
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
    "model_id": MODEL_ID, "dataset_id": DATASET_ID,
    "experiment_family": EXPERIMENT_FAMILY, "debug_mode": DEBUG_MODE,
    "device": DEVICE, "split": SPLIT,
    "languages": [c["language_key"] for c in lang_cfgs],
    "capabilities": caps.enabled_strategies,
})
zip_run_outputs(dirs)
drive_backup(dirs, drive_available)
if ENV["in_colab"]:
    colab_download(dirs)
