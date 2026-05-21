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
# # LinguoMT — African-Celtic + Whisper-large-v3 + NLLB-200-distilled-600M
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
    FinetuneConfig, WhisperNLLBCascadeFineTuner,
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
# True → skip audio evaluation steps in debug runs to save time.
# False → run audio evaluation in all modes.
SKIP_AUDIO_DEBUG  = True
# True → delete and rebuild the dataset cache even if one already exists.
# False → reuse the existing cache (saves 5–15 min on re-runs). Only set True
#         if the dataset has changed or you suspect a corrupted cache file.
FORCE_RERUN       = False
# True → run all experiment configurations (recommended for paper results).
# False → run only Experiment_1, even in full mode. Useful for a quick
#         single-configuration sanity check without the full grid.
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
FINETUNE_DIRECT_SPEECH_TRANSLATION = False   # not applicable to cascade; kept for API compatibility
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
WHISPER_ID        = "openai/whisper-large-v3"
NLLB_ID           = "facebook/nllb-200-distilled-600M"
MODEL_ID          = "whisper_nllb"
DATASET_ID        = "McGill-NLP/african_celtic_dataset"
MODEL_NAME        = "Whisper-large-v3 + NLLB-distilled-600M"
DATASET_NAME      = "African-Celtic"
EXPERIMENT_FAMILY = "AfricanCeltic__WhisperNLLB"
SPLIT             = "dev"
TRAIN_SPLIT       = "train"
MANUAL_LANGUAGES  = ["yoruba", "hausa"]   # Igbo excluded: no Whisper token
SEED              = 42

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
SOTA_FILE         = ""            # e.g. "sota/paper1_benchmark/sota_results.csv"
# Paper 2 only: training-set sizes for the data scaling analysis.
# Each value triggers a separate fine-tuning run at that many training examples.
# 0 = use the full available training set. Leave [] to disable scaling experiments.
# e.g. SCALING_BUDGETS = [100, 500, 1000, 0]
SCALING_BUDGETS   = []
# ─────────────────────────────────────────────────────────────────────────────

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
    tgt_id = nllb_tok.convert_tokens_to_ids(nllb_tgt)
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

# %% --- fine-tuner setup ---
ft_cfg = FinetuneConfig(
    finetune_text_translation=FINETUNE_TEXT_TRANSLATION,
    finetune_reverse_translation=FINETUNE_REVERSE_TRANSLATION,
    finetune_asr=FINETUNE_ASR,
    finetune_direct_speech_translation=False,
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
    tuner   = WhisperNLLBCascadeFineTuner(
        whisper_model, whisper_proc,
        nllb_model, nllb_tok,
        ft_cfg, _device,
    )

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
    "asr_model": WHISPER_ID, "mt_model": NLLB_ID,
    "dataset_id": DATASET_ID, "experiment_family": EXPERIMENT_FAMILY,
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
