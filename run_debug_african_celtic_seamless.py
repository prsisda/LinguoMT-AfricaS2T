"""
Debug runner: AfricanCeltic + SeamlessM4T-v2-large
Loads model once; runs each language in sequence with its own output folder.
Each language run: both directions, debug sample size (8 text / 3 audio pairs).
"""
import sys, random, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model

from framework import (
    detect_environment, detect_model_capabilities,
    select_supported_languages, get_adapter_type,
    DatasetCache, StepMonitor, create_run_dirs, save_config,
    zip_run_outputs, drive_backup, ExperimentRunner, RunConfig,
    default_experiment_configs,
)

warnings.filterwarnings("ignore")

# ── settings ──────────────────────────────────────────────────────────────────
MODEL_ID         = "facebook/seamless-m4t-v2-large"
DATASET_ID       = "McGill-NLP/african_celtic_dataset"
MODEL_NAME       = "SeamlessM4T-v2 Large"
DATASET_NAME     = "African-Celtic"
EXPERIMENT_FAMILY= "AfricanCeltic__SeamlessM4Tv2_Large"
SPLIT            = "dev"
LANGUAGES        = ["igbo", "yoruba", "hausa"]   # run in this order
DIRECTIONS       = ["source_to_english", "english_to_source"]
SEED             = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ENV    = detect_environment()
DEVICE = ENV["device"]
caps   = detect_model_capabilities(MODEL_ID)

# ── load model once ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Loading {MODEL_NAME} on {DEVICE} ...")
print(f"{'='*60}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model     = SeamlessM4Tv2Model.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print("Model ready.\n")

# ── translation callables ──────────────────────────────────────────────────────
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

# ── run each language ─────────────────────────────────────────────────────────
all_caps    = caps
adapter     = get_adapter_type(DATASET_ID)
exp_cfgs    = default_experiment_configs(debug_mode=True)
max_pairs   = max(e["max_text_dev"] for e in exp_cfgs)

for lang_key in LANGUAGES:
    lang_cfgs = select_supported_languages(
        all_caps, DATASET_ID, processor=processor, max_langs=1,
        manual_override=[lang_key],
    )
    if not lang_cfgs:
        print(f"\n[SKIP] {lang_key} — not supported for this model+dataset pair.\n")
        continue

    lang_display = lang_cfgs[0]["display"]
    print(f"\n{'='*60}")
    print(f"  Language: {lang_display}  ({lang_key})")
    print(f"  Directions: {DIRECTIONS}")
    print(f"  Mode: DEBUG")
    print(f"{'='*60}\n")

    # Dataset cache for this language only
    cache = DatasetCache(
        dataset_id       = DATASET_ID,
        adapter_type     = adapter,
        language_configs = lang_cfgs,
        split            = SPLIT,
        max_pairs        = max_pairs,
        max_scan_rows    = 20000,
    )

    # Output folder per language
    dirs    = create_run_dirs(
        model_slug   = "seamlessm4t",
        dataset_slug = f"african_celtic_{lang_key}",
        debug_mode   = True,
        in_colab     = False,
        base_root    = Path(__file__).parent / "AfricanCeltic__SeamlessM4Tv2" / "outputs",
    )
    monitor = StepMonitor(dirs.monitoring)
    monitor.step(f"Starting {lang_display}", f"directions={DIRECTIONS}")

    cache.build(monitor)
    stats = cache.stats()
    print(f"Cache: {stats}")
    if stats.get(lang_key, 0) == 0:
        print(f"[WARNING] No pairs found for {lang_display} — skipping.\n")
        continue

    ExperimentRunner(
        config=RunConfig(
            model_name        = MODEL_NAME,
            dataset_name      = DATASET_NAME,
            experiment_family = EXPERIMENT_FAMILY,
            model_id          = MODEL_ID,
            dataset_id        = DATASET_ID,
            split             = SPLIT,
            debug_mode        = True,
            skip_audio_debug  = False,   # run audio strategies in debug
            in_colab          = False,
            eda_sample_size   = 10,
            directions        = DIRECTIONS,
        ),
        capabilities       = all_caps,
        data_cache         = cache,
        language_configs   = lang_cfgs,
        dirs               = dirs,
        monitor            = monitor,
        experiment_configs = exp_cfgs,
        translate_text_fn  = translate_text,
        translate_audio_fn = translate_audio,
        asr_fn             = asr_transcribe,
    ).run()

    save_config(dirs, {
        "model_id":           MODEL_ID,
        "dataset_id":         DATASET_ID,
        "experiment_family":  EXPERIMENT_FAMILY,
        "language":           lang_key,
        "debug_mode":         True,
        "device":             DEVICE,
        "split":              SPLIT,
        "directions":         DIRECTIONS,
        "capabilities":       all_caps.enabled_strategies,
    })
    zip_run_outputs(dirs)
    print(f"\nResults saved → {dirs.base}\n")

print("\n" + "="*60)
print("All languages complete.")
print("="*60)
