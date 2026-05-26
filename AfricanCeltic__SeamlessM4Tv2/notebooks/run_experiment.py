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
# ── How to run this script directly on Colab ──────────────────────────────────
# 1. Open this file as a Colab notebook (File → Open → upload or open from Drive)
# 2. Set runtime to T4 or A100: Runtime → Change runtime type → GPU
# 3. Edit the configuration block below — minimum required changes:
#      DEBUG_MODE        = False   ← False for a paper run, True for a quick test
#      HF_CACHE_DIR      = "/content/drive/MyDrive/LinguoMT-AfricaS2T/hf_cache"
#      DATASET_CACHE_DIR = "/content/drive/MyDrive/LinguoMT-AfricaS2T/dataset_cache"
# 4. Runtime → Run all
#
# If HF_CACHE_DIR points to the cache built in Step 5 of run_on_colab.ipynb,
# models load from Drive (no re-download). If DATASET_CACHE_DIR points to the
# cache built in Step 6, dataset scanning is skipped (loads in < 5 sec).
# ─────────────────────────────────────────────────────────────────────────────
# Run locally : python run_experiment.py

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

# ── Run mode ──────────────────────────────────────────────────────────────────
# DEBUG_MODE = True  → evaluates ~20 samples per language in ~10 min.
#                      Results are statistically meaningless at this scale.
#                      Use ONLY to verify the pipeline runs without errors
#                      before committing to a full run.
# DEBUG_MODE = False → evaluates the full dev set (30 min – 4 h on GPU).
#                      REQUIRED for any number you intend to report in a paper.
DEBUG_MODE        = True
# FAST_MODE = True forces DEBUG_MODE = True regardless of the setting above.
# Use as a shortcut when you just want to re-enter the pipeline quickly.
FAST_MODE         = False
# False → audio eval runs in all modes, including DEBUG_MODE.
#         Required for Paper 3 (audio strategies) — True will suppress
#         audio_metrics.csv and audio_metrics_all.csv entirely.
#         Recommended for Paper 2 (adaptation) to capture before/after audio results.
# True  → skips audio evaluation when DEBUG_MODE=True (faster smoke test).
#         Safe for Papers 1, 4, 5 where text/ASR metrics are the primary output
#         and a quick pipeline check does not need audio strategy results.
SKIP_AUDIO_DEBUG  = False
# True → delete and rebuild the dataset cache even if one already exists.
# False → reuse the existing cache (saves 5–15 min on re-runs). Only set True
#         if the dataset has changed or you suspect a corrupted cache file.
FORCE_RERUN       = False
# Number of evaluation passes to run: 1, 2, or 3. Default is 3.
# Each pass evaluates on a progressively larger sample, producing three
# data points that show metric stability as sample size grows.
#   1 pass  → fastest; one result per language (Experiment_1 only)
#   2 passes → two results per language (Experiment_1 + Experiment_2)
#   3 passes → three results per language — recommended for paper results
N_EVAL_RUNS       = 3

# Text pairs (source+reference) evaluated in each pass.
# The Nth value is used for the Nth pass. Must have at least N_EVAL_RUNS values.
# Increase these for more reliable metrics; decrease to save time.
EVAL_TEXT_SAMPLES  = [100, 200, 300]

# Audio utterances evaluated in each pass (speech-to-text and ASR tasks).
# Keep proportional to EVAL_TEXT_SAMPLES. Audio evaluation is slower than text.
EVAL_AUDIO_SAMPLES = [ 30,  75, 100]

# True → run all N_EVAL_RUNS passes. False → run only pass 1 (legacy override).
# Prefer setting N_EVAL_RUNS = 1 over RUN_FULL_GRID = False for clarity.
RUN_FULL_GRID     = True
# True → fine-tune the model before evaluation (Papers 2 & 3 only).
# False → evaluate the pretrained model zero-shot (Papers 1, 4, 5).
ENABLE_FINETUNING = False

# ── Fine-tuning configuration (only used when ENABLE_FINETUNING = True) ──────
# Fine-tuning strategy — trade-off between quality, GPU memory, and training time:
#   "lora"    → low-rank adapters; most weights frozen. Runs on T4 (~10 GB). Recommended.
#   "adapter" → bottleneck adapters inserted between layers. Needs ~14 GB GPU.
#   "full"    → all weights updated. Best quality; needs A100 (24+ GB). Slowest.
FINETUNING_METHOD              = "lora"
FINETUNE_TEXT_TRANSLATION      = True    # fine-tune text-to-text translation (improves BLEU/ChrF)
FINETUNE_REVERSE_TRANSLATION   = True    # also fine-tune the English→source direction
FINETUNE_ASR                   = True    # fine-tune speech recognition (improves WER/CER)
FINETUNE_DIRECT_SPEECH_TRANSLATION = False   # fine-tune SeamlessM4T S2TT end-to-end path
# Max training samples per task. Capped by dataset size if the split is smaller.
TEXT_FINETUNE_SAMPLES          = 1000
ASR_FINETUNE_SAMPLES           = 500
ST_FINETUNE_SAMPLES            = 200
# Training hyperparameters. Conservative defaults that run on a T4 GPU.
# Reduce batch sizes if you see an out-of-memory (OOM) error.
TEXT_EPOCHS                    = 3
TEXT_BATCH_SIZE                = 8
TEXT_LR                        = 5e-5
ASR_EPOCHS                     = 3
ASR_BATCH_SIZE                 = 4
ASR_LR                         = 1e-5
ST_EPOCHS                      = 3
ST_BATCH_SIZE                  = 4
ST_LR                          = 1e-5
GRADIENT_ACCUMULATION_STEPS    = 4    # effective batch size = batch_size × this value
FP16                           = True    # mixed-precision training; set False if NaN losses appear
EARLY_STOPPING_PATIENCE        = 2    # stop training if dev loss does not improve for N epochs
SAVE_CHECKPOINTS               = True    # save model weights to disk after fine-tuning
EVAL_BEFORE_AFTER              = True    # record metrics before AND after fine-tuning for comparison
# ─────────────────────────────────────────────────────────────────────────────

# ── Publication settings ──────────────────────────────────────────────────────
# PAPER_MODE controls which analyses run and which report sections are produced.
# Set this to match the paper you are writing:
#   "benchmark"  → Paper 1: zero-shot baselines + SOTA gap analysis
#   "adaptation" → Paper 2: before/after fine-tuning + data scaling curves
#   "audio"      → Paper 3: audio strategy comparison (S2TT, normalise, trim, chunk)
#   "cascade"    → Paper 4: oracle cascade, error propagation, break-even WER
#   "transfer"   → Paper 5: typological similarity, cross-lingual transfer, few-shot
PAPER_MODE        = "benchmark"
# Path to a CSV of published baselines for SOTA comparison (used by Paper 1).
# Required columns: system, language, BLEU, venue, year
# Leave "" to skip the SOTA comparison section in the report entirely.
SOTA_FILE         = ""    # e.g. "sota/paper1_benchmark/sota_results.csv"
# SCALING_BUDGETS — Paper 2 only.
# Answers the question: "how much does performance improve as the model gets more training data?"
# Each value triggers a separate fine-tuning pass at that many training examples.
# The result is a learning curve (samples on X axis, BLEU/WER on Y axis) that shows
# whether the model is still improving with more data or has already plateaued.
#
# Example — [100, 500, 1000, 0] runs 4 fine-tuning passes:
#   100  → fine-tune on 100 randomly sampled examples  → record BLEU/WER
#   500  → fine-tune on 500 randomly sampled examples  → record BLEU/WER
#   1000 → fine-tune on 1000 randomly sampled examples → record BLEU/WER
#   0    → fine-tune on the full training split        → record BLEU/WER
#
# Special values:
#   any positive integer → use exactly that many training examples
#   0                    → use the entire training split ("full set" sentinel)
#   []                   → skip scaling entirely (correct for Papers 1, 3, 4, 5)
#
# Runtime: each budget is a full fine-tuning pass.
# [100, 500, 1000, 0] multiplies total runtime by 4.
# Budget 0 alone can take 1–2 h on a T4 GPU — plan accordingly.
SCALING_BUDGETS   = []
# ── HuggingFace cache directory ──────────────────────────────────────────────
# Set to persist model and dataset downloads across sessions (e.g. on Drive).
# Patched automatically by run_on_colab.ipynb — set "" for ephemeral cache.
HF_CACHE_DIR      = ""
# Pre-built dataset sample cache directory (from Step 6 in run_on_colab.ipynb).
# Patched automatically by run_on_colab.ipynb — set "" to build cache locally.
DATASET_CACHE_DIR = ""
# Max text pairs pre-cached in Step 6; must match MAX_CACHED_PAIRS in notebook.
# Patched automatically by run_on_colab.ipynb.
MAX_CACHED_PAIRS  = 300
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
if HF_CACHE_DIR:
    import os; os.makedirs(HF_CACHE_DIR, exist_ok=True); os.environ["HF_HOME"] = HF_CACHE_DIR
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
exp_cfgs      = default_experiment_configs(DEBUG_MODE, N_EVAL_RUNS, EVAL_TEXT_SAMPLES, EVAL_AUDIO_SAMPLES)
max_dev_pairs = max(e["max_text_dev"]   for e in exp_cfgs)
max_trn_pairs = max(e["max_text_train"] for e in exp_cfgs)

dev_cache = DatasetCache(
    dataset_id=DATASET_ID, adapter_type=get_adapter_type(DATASET_ID),
    language_configs=lang_cfgs, split=SPLIT,
    max_pairs=MAX_CACHED_PAIRS if DATASET_CACHE_DIR else max_dev_pairs,
    max_scan_rows=50000 if DATASET_CACHE_DIR else (20000 if DEBUG_MODE else 50000),
    cache_dir=Path(DATASET_CACHE_DIR) / Path(__file__).parent.parent.name if DATASET_CACHE_DIR else Path(__file__).parent.parent / "cache",
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
