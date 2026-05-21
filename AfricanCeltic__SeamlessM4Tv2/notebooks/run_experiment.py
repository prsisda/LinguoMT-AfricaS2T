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
import sys, subprocess
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

# Install deps BEFORE importing framework (framework imports sacrebleu/librosa at module level)
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
from transformers import AutoProcessor, SeamlessM4Tv2Model
from framework import (
    detect_model_capabilities, select_supported_languages, get_adapter_type,
    DatasetCache, StepMonitor, create_run_dirs, save_config,
    zip_run_outputs, drive_backup, colab_download, mount_google_drive,
    ExperimentRunner, RunConfig, default_experiment_configs,
    FinetuneConfig, SeamlessM4TFineTuner,
)
warnings.filterwarnings("ignore")

# %% --- configuration ---
DEBUG_MODE        = True
FAST_MODE         = False
SKIP_AUDIO_DEBUG  = True
FORCE_RERUN       = False
RUN_FULL_GRID     = True    # False → only Experiment_1 in full mode
ENABLE_FINETUNING = False   # set True once baseline is stable
# ── Fine-tuning configuration (only used when ENABLE_FINETUNING = True) ──────
FINETUNING_METHOD              = "lora"
FINETUNE_TEXT_TRANSLATION      = True
FINETUNE_REVERSE_TRANSLATION   = True
FINETUNE_ASR                   = True
FINETUNE_DIRECT_SPEECH_TRANSLATION = False   # set True to also fine-tune S2TT
TEXT_FINETUNE_SAMPLES          = 1000
ASR_FINETUNE_SAMPLES           = 500
ST_FINETUNE_SAMPLES            = 200
TEXT_EPOCHS                    = 3
TEXT_BATCH_SIZE                = 8
TEXT_LR                        = 5e-5
ASR_EPOCHS                     = 3
ASR_BATCH_SIZE                 = 4
ASR_LR                         = 1e-5
ST_EPOCHS                      = 3
ST_BATCH_SIZE                  = 4
ST_LR                          = 1e-5
GRADIENT_ACCUMULATION_STEPS    = 4
FP16                           = True
EARLY_STOPPING_PATIENCE        = 2
SAVE_CHECKPOINTS               = True
EVAL_BEFORE_AFTER              = True
# ─────────────────────────────────────────────────────────────────────────────
# ── Publication settings ──────────────────────────────────────────────────────
# PAPER_MODE selects the output emphasis for the LinguoMT journal series:
#   "benchmark"  → LinguoMT (pretrained baseline evaluation + SOTA comparison)
#   "adaptation" → LinguoMT-Adapt (fine-tuning before/after)
#   "audio"      → LinguoMT-Audio (audio strategy analysis)
#   "cascade"    → LinguoMT-Cascade (cascade vs end-to-end)
#   "transfer"   → LinguoMT-Transfer (cross-lingual analysis)
PAPER_MODE        = "benchmark"
# Path to sota_results.csv or published_baselines.json (leave "" to skip SOTA tables)
SOTA_FILE         = ""    # e.g. "sota/sota_results.csv"
# Paper 2 only: data scaling budgets. Set to [] to skip.
SCALING_BUDGETS   = []
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID          = "facebook/seamless-m4t-v2-large"
DATASET_ID        = "McGill-NLP/african_celtic_dataset"
MODEL_NAME        = "SeamlessM4T-v2 Large"
DATASET_NAME      = "African-Celtic"
EXPERIMENT_FAMILY = "AfricanCeltic__SeamlessM4Tv2_Large"
SPLIT             = "dev"
TRAIN_SPLIT       = "train"
MANUAL_LANGUAGES  = ["igbo", "yoruba"]   # Hausa excluded: not in African-Celtic dataset
SEED              = 42

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

# %% --- fine-tuner setup ---
ft_cfg = FinetuneConfig(
    finetune_text_translation=FINETUNE_TEXT_TRANSLATION,
    finetune_reverse_translation=FINETUNE_REVERSE_TRANSLATION,
    finetune_asr=FINETUNE_ASR,
    finetune_direct_speech_translation=FINETUNE_DIRECT_SPEECH_TRANSLATION,
    finetuning_method=FINETUNING_METHOD,
    text_finetune_samples=TEXT_FINETUNE_SAMPLES,
    asr_finetune_samples=ASR_FINETUNE_SAMPLES,
    st_finetune_samples=ST_FINETUNE_SAMPLES,
    text_epochs=TEXT_EPOCHS, text_batch_size=TEXT_BATCH_SIZE, text_lr=TEXT_LR,
    asr_epochs=ASR_EPOCHS,   asr_batch_size=ASR_BATCH_SIZE,   asr_lr=ASR_LR,
    st_epochs=ST_EPOCHS,     st_batch_size=ST_BATCH_SIZE,     st_lr=ST_LR,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    fp16=FP16, early_stopping_patience=EARLY_STOPPING_PATIENCE,
    save_checkpoints=SAVE_CHECKPOINTS, eval_before_after=EVAL_BEFORE_AFTER,
) if ENABLE_FINETUNING else None

tuner = None
if ENABLE_FINETUNING:
    import torch
    _device = torch.device(DEVICE)
    tuner   = SeamlessM4TFineTuner(model, processor, ft_cfg, _device)

# %% --- data cache ---
exp_cfgs      = default_experiment_configs(DEBUG_MODE)
max_dev_pairs = max(e["max_text_dev"]   for e in exp_cfgs)
max_trn_pairs = max(e["max_text_train"] for e in exp_cfgs)

dev_cache = DatasetCache(
    dataset_id=DATASET_ID, adapter_type=get_adapter_type(DATASET_ID),
    language_configs=lang_cfgs, split=SPLIT,
    max_pairs=max_dev_pairs,
    max_scan_rows=20000 if DEBUG_MODE else 50000,
    cache_dir=Path(__file__).parent.parent / "cache",
    force_rerun=FORCE_RERUN,
)
dev_cache.build(monitor)
print("Dev cache:", dev_cache.stats())

train_cache = None
if ENABLE_FINETUNING and max_trn_pairs > 0:
    train_cache = DatasetCache(
        dataset_id=DATASET_ID, adapter_type=get_adapter_type(DATASET_ID),
        language_configs=lang_cfgs, split=TRAIN_SPLIT,
        max_pairs=max_trn_pairs,
        max_scan_rows=200000,
        force_rerun=FORCE_RERUN,
    )
    train_cache.build(monitor)
    print("Train cache:", train_cache.stats())

# %% --- run ---
if tuner and ft_cfg:
    ft_cfg.checkpoint_dir = str(dirs.base / "checkpoints")

ExperimentRunner(
    config=RunConfig(
        model_name=MODEL_NAME, dataset_name=DATASET_NAME,
        experiment_family=EXPERIMENT_FAMILY, model_id=MODEL_ID, dataset_id=DATASET_ID,
        split=SPLIT, train_split=TRAIN_SPLIT,
        debug_mode=DEBUG_MODE, fast_mode=FAST_MODE,
        skip_audio_debug=SKIP_AUDIO_DEBUG, force_rerun=FORCE_RERUN,
        run_full_grid=RUN_FULL_GRID, enable_finetuning=ENABLE_FINETUNING,
        in_colab=ENV["in_colab"], eda_sample_size=25 if DEBUG_MODE else 200,
        directions=["source_to_english", "english_to_source"],
        paper_mode=PAPER_MODE,
        sota_path=SOTA_FILE,
        scaling_budgets=SCALING_BUDGETS,
    ),
    capabilities=caps, data_cache=dev_cache, train_cache=train_cache,
    language_configs=lang_cfgs, dirs=dirs, monitor=monitor, experiment_configs=exp_cfgs,
    translate_text_fn=translate_text, translate_audio_fn=translate_audio, asr_fn=asr_transcribe,
    finetune_fn=tuner.finetune if tuner else None,
).run()

# %% --- archive ---
save_config(dirs, {
    "model_id": MODEL_ID, "dataset_id": DATASET_ID,
    "experiment_family": EXPERIMENT_FAMILY,
    "debug_mode": DEBUG_MODE, "enable_finetuning": ENABLE_FINETUNING,
    "device": DEVICE, "split": SPLIT, "train_split": TRAIN_SPLIT,
    "languages": [c["language_key"] for c in lang_cfgs],
    "capabilities": caps.enabled_strategies,
    "dev_cache_stats": dev_cache.stats(),
    "train_cache_stats": train_cache.stats() if train_cache else {},
})
zip_run_outputs(dirs)
drive_backup(dirs, drive_available)
if ENV["in_colab"]:
    colab_download(dirs)
