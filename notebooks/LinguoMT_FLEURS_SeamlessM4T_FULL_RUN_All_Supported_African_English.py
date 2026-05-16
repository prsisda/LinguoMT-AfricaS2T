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

# %% [markdown] id="fca6aa58"
# # LinguoMT — FLEURS + SeamlessM4T-v2 Comparable Experiment Template
#
# This notebook is a paper-comparable baseline experiment for multilingual African speech-to-text translation using **FLEURS** as the dataset and **SeamlessM4T-v2** as the model. It is designed so the same experimental structure can be reused for other dataset–model pairs, such as African-Celtic + SeamlessM4T, FLEURS + Whisper/NLLB, African-Celtic + Whisper/NLLB, or MMS/NLLB variants.
#
# The notebook intentionally keeps the same stages, output filenames, metrics, optimization strategies, and aggregation format across experiments. This makes it easier to compare results fairly in a paper. When creating a new baseline notebook, change only the model-loading and dataset-loading sections while keeping the experiment runner, metric names, output schemas, and interpretation sections consistent.
#
# Expected final outputs include per-language CSV files, aggregate metric tables, audio quality tables, duration analysis tables, qualitative prediction samples, plots, and a Google Drive backup.

# %% [markdown] id="87aeed90"
# # Experimental Guidelines for Comparable African Speech Translation Research
#
# This notebook follows a controlled experimental design for low-resource African speech translation. The central research question is whether translation errors mainly come from the **speech understanding stage**, the **translation/generation stage**, or the **audio preprocessing strategy**.
#
# For comparability across models and datasets, each baseline should use the same sample sizes, language-pair definitions, evaluation metrics, and optimization strategies. The only variables that should change between comparable notebooks are the selected model and dataset.
#
# Interpretation rule for the paper: do not report a single BLEU score in isolation. Always interpret model quality together with transcript translation, direct audio translation, normalized audio translation, trimmed audio translation, chunk-based audio translation, and the gold-transcript cascade. This helps separate acoustic limitations from translation limitations.

# %% [markdown] id="b6482e29"
# # 1 Introduction and Motivation
#
# This section explains the research motivation and frames the notebook as a reusable experimental pipeline. The main task is multilingual African language speech-to-text translation, with both African → English and English → African text directions included where supported.
#
# For paper writing, this section should support the introduction and experimental setup. It explains why low-resource African speech translation is difficult: limited paired speech data, speaker and accent variation, variable audio quality, silence regions, and uneven multilingual model coverage.

# %% [markdown] id="9a80d60a"
# ## 1.1 Problem Definition
#
# The problem is to evaluate whether a multilingual speech translation model can translate African-language speech and transcripts into English, and optionally generate African-language text from English.
#
# In the paper, this section can be used to define the task formally:
#
# - Input 1: African-language transcript text.
# - Input 2: African-language speech audio.
# - Output: English translation.
# - Reverse-direction diagnostic: English text → African-language text.
#
# Expected insight: if transcript translation performs much better than audio translation, then the main bottleneck is likely speech understanding rather than text translation.

# %% [markdown] id="18c100f3"
# ## 1.2 Experiment Design
#
# The notebook runs the same experiment grid for every enabled language and configuration. Each configuration controls the number of text and audio samples used.
#
# The design includes:
#
# 1. Dataset loading and sanity checks.
# 2. Sample-pair inspection for human verification.
# 3. Text-data exploration.
# 4. Audio-data exploration and audio-quality analysis.
# 5. Text translation evaluation.
# 6. Direct speech translation evaluation.
# 7. Audio optimization evaluations: normalization, silence trimming, chunking, and duration-based analysis.
# 8. Gold-transcript cascade evaluation.
# 9. Aggregated result export.
#
# Interpretation rule: improvements from normalization or trimming suggest sensitivity to audio amplitude or silence; improvements from chunking suggest the model struggles with longer audio contexts.

# %% [markdown] id="ca7a1e5a"
# ## 1.3 Environment Setup and Baseline Model
#
# This section installs dependencies, defines the runtime configuration, sets reproducibility seeds, specifies enabled languages, and loads the model.
#
# For consistent comparisons across baselines, keep the following settings aligned between notebooks:
#
# - Random seed.
# - Debug/full-run mode.
# - Sample sizes per experiment.
# - Minimum and maximum audio duration.
# - Output folder structure.
# - Metric names and CSV schemas.
#
# When comparing another model, only change the model-loading block and the translation helper functions. The rest of the pipeline should remain structurally identical.

# %% [markdown] id="e6100c03"
# ### 1.3.1 Library Installation
#
# This cell installs the libraries needed for dataset loading, audio preprocessing, model inference, evaluation, and visualization.
#
# Expected output: package installation logs. If Colab restarts or reports version conflicts, rerun this cell and then rerun the notebook from the runtime configuration cell.

# %% id="ae7b954f" colab={"base_uri": "https://localhost:8080/"} outputId="8081a2b8-07af-4d9a-8ddc-c53db51d2628"
# !pip -q install -U transformers datasets sacrebleu librosa soundfile sentencepiece accelerate matplotlib jiwer "pandas==2.2.2" "pyarrow>=15.0.0"

# %% [markdown] id="7dfe7571"
# ### 1.3.2 Baseline and Runtime Configuration
#
# This is the main control panel for the notebook. It determines which model, dataset, languages, directions, sample sizes, and optimization strategies are used.
#
# Use `DEBUG_MODE=True` for a quick smoke test with very small sample sizes. Use the full-run setting only after confirming that the sample check, model loading, and one-language evaluation work.
#
# For paper comparability, keep the experiment names and sample sizes consistent across all baseline notebooks unless you explicitly report a different setup.
#
# Important interpretation note: if a language is unsupported by either the dataset or the model, disable that pair rather than forcing an invalid language code. Invalid language codes can produce misleading outputs and unfair comparisons.

# %% colab={"base_uri": "https://localhost:8080/"} id="50cd03db" outputId="515e7356-7810-4b78-d3a3-89d83b72ea71"
import os
import re
import json
import random
import warnings
import unicodedata
import time
from pathlib import Path


import numpy as np
import pandas as pd

import torch
import librosa
import sacrebleu
import matplotlib.pyplot as plt
from jiwer import wer, cer

from datasets import load_dataset, Dataset, Audio, load_dataset_builder, get_dataset_config_names
from transformers import AutoProcessor, SeamlessM4Tv2Model

warnings.filterwarnings("ignore")

PROJECT_NAME = "LinguoMT Configurable African-English Speech Translation Experiments"
MODEL_ID = "facebook/seamless-m4t-v2-large"
DATASET_ID = "google/fleurs"
ENGLISH_CONFIG = "en_us"

TARGET_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Runtime control
# ============================================================
# Full-run mode: debugging was successful, so run all supported language pairs and experiments.
DEBUG_MODE = True

# In debug mode, audio inference can still be slow. Set True to test only data + text pipeline.
SKIP_AUDIO_DEBUG = True

# True = run all enabled and supported languages. False = use SELECTED_LANGUAGE_PAIRS only.
RUN_FULL_GRID = False

# Turn components on/off.
RUN_TEXT_EVALUATION = True
RUN_AUDIO_EVALUATION = True
RUN_SAMPLE_CHECK = True
RUN_DATA_EXPLORATION = True  # Generate dataset properties, statistics, plots, and audio-quality tables before inference.

# ============================================================
# Cross-baseline comparability controls
# ============================================================
# These labels make the output schema comparable across notebooks.
# Keep the same values/columns when creating a notebook for another model or dataset.
BASELINE_MODEL_NAME = "SeamlessM4T-v2 Large"
BASELINE_DATASET_NAME = "FLEURS"
BASELINE_PIPELINE_TYPE = "end_to_end_speech_translation"
EXPERIMENT_FAMILY = f"{BASELINE_DATASET_NAME}__{BASELINE_MODEL_NAME}".replace(" ", "_").replace("/", "_")

# ============================================================
# Audio optimization strategy controls
# ============================================================
# Turn individual strategies on/off here. Keep the strategy names stable across notebooks
# so results from different models and datasets can be compared directly in the paper.
RUN_AUDIO_STRATEGY_BASELINE_DIRECT = True
RUN_AUDIO_STRATEGY_NORMALIZED = True
RUN_AUDIO_STRATEGY_TRIMMED = True
RUN_AUDIO_STRATEGY_CHUNK_BASED = True

# If True, expensive audio strategies run only for Experiment_1.
# Set False to run every strategy in every experiment.
RUN_EXPENSIVE_AUDIO_STRATEGIES_ONLY_IN_EXP1 = False

# Chunking parameters for the chunk-based optimization strategy.
CHUNK_SECONDS = 6
CHUNK_OVERLAP_SECONDS = 1

AUDIO_OPTIMIZATION_STRATEGIES = [
    {
        "strategy_key": "baseline_direct",
        "method": "Direct audio → English",
        "category": "Direct Speech Translation",
        "enabled": RUN_AUDIO_STRATEGY_BASELINE_DIRECT,
        "normalize": False,
        "trim": False,
        "chunk": False,
        "expensive": False,
    },
    {
        "strategy_key": "normalized_audio",
        "method": "Normalized audio → English",
        "category": "Audio Normalization",
        "enabled": RUN_AUDIO_STRATEGY_NORMALIZED,
        "normalize": True,
        "trim": False,
        "chunk": False,
        "expensive": False,
    },
    {
        "strategy_key": "trimmed_audio",
        "method": "Trimmed audio → English",
        "category": "Silence Trimming",
        "enabled": RUN_AUDIO_STRATEGY_TRIMMED,
        "normalize": True,
        "trim": True,
        "chunk": False,
        "expensive": True,
    },
    {
        "strategy_key": "chunk_based_audio",
        "method": "Chunk-based audio → English",
        "category": "Audio Segmentation",
        "enabled": RUN_AUDIO_STRATEGY_CHUNK_BASED,
        "normalize": True,
        "trim": False,
        "chunk": True,
        "expensive": True,
    },
]


# Choose directions. Options:
# "african_to_english", "english_to_african"
DIRECTIONS_TO_RUN = ["african_to_english", "english_to_african"]

# Used only when RUN_FULL_GRID=False.
# Change this list to test one or more specific language pairs.
SELECTED_LANGUAGE_PAIRS = ["igbo", "yoruba", "swahili"]

RUN_FULL_AUDIO_OPTIMIZATION_ONLY_IN_EXP1 = False

BASE_OUTPUT_FOLDER_NAME = "LinguoMT_FLEURS_SeamlessM4T_Experiments_Results"
BASE_OUTPUT_FOLDER_NAME_DEBUG = "LinguoMT_FLEURS_SeamlessM4T_Experiments_Results_Debugging"

# ============================================================
# Language configuration
# ============================================================
# seamless_code must be a valid SeamlessM4T language code.
# dataset_supported says whether FLEURS has the corresponding config.
# model_supported says whether SeamlessM4T-v2 can use the language code.
ALL_LANGUAGE_CONFIGS = [
    {
        "language": "igbo",
        "fleurs_config": "ig_ng",
        "source_lang_code": "ibo",
        "display_name": "Igbo",
        "output_prefix": "igbo_english",
        "dataset_supported": True,
        "model_supported": True,
        "enabled": True,
    },
    {
        "language": "yoruba",
        "fleurs_config": "yo_ng",
        "source_lang_code": "yor",
        "display_name": "Yoruba",
        "output_prefix": "yoruba_english",
        "dataset_supported": True,
        "model_supported": True,
        "enabled": True,
    },
    {
        "language": "swahili",
        "fleurs_config": "sw_ke",
        "source_lang_code": "swh",
        "display_name": "Swahili",
        "output_prefix": "swahili_english",
        "dataset_supported": True,
        "model_supported": True,
        "enabled": True,
    },
    {
        "language": "wolof",
        "fleurs_config": "wo_sn",
        "source_lang_code": "wol",
        "display_name": "Wolof",
        "output_prefix": "wolof_english",
        "dataset_supported": True,
        "model_supported": False,
        "enabled": False,
    },
    {
        "language": "hausa",
        "fleurs_config": "ha_ng",
        "source_lang_code": None,
        "display_name": "Hausa",
        "output_prefix": "hausa_english",
        "dataset_supported": True,
        "model_supported": False,  # SeamlessM4T-v2 does not natively support Hausa.
        "enabled": False,
    },
]

# ============================================================
# Experiment configuration
# ============================================================
SPLIT_FOR_EXPERIMENT = "test"
MIN_DURATION = 1.0
MAX_DURATION = 20.0

# ============================================================
# Data exploration configuration
# ============================================================
# Keep this moderate so the EDA is useful but still fast in Colab.
EDA_SPLITS_TO_ANALYZE = ["train", "validation", "test"]
EDA_SAMPLE_SIZE_PER_LANGUAGE_SPLIT = 200
EDA_MAX_SCAN_ROWS = 1000
EDA_AUDIO_EXAMPLES_PER_LANGUAGE = 2

if DEBUG_MODE:
    BASE_OUTPUT_FOLDER_NAME = BASE_OUTPUT_FOLDER_NAME_DEBUG
    EXPERIMENT_CONFIGS = [
        {
            "experiment": "debug",
            "max_text_dev": 8,
            "max_audio_dev": 3,
            "max_scan_rows": 100,
        }
    ]
else:
    EXPERIMENT_CONFIGS = [
        {"experiment": "Experiment_1", "max_text_dev": 50, "max_audio_dev": 10, "max_scan_rows": 1000},
        {"experiment": "Experiment_2", "max_text_dev": 100, "max_audio_dev": 30, "max_scan_rows": 1500},
        {"experiment": "Experiment_3", "max_text_dev": 200, "max_audio_dev": 50, "max_scan_rows": 2000},
    ]

if DEBUG_MODE:
    EDA_SAMPLE_SIZE_PER_LANGUAGE_SPLIT = 25
    EDA_MAX_SCAN_ROWS = 100
    EDA_AUDIO_EXAMPLES_PER_LANGUAGE = 1

# Apply language selection.
if RUN_FULL_GRID:
    LANGUAGE_CONFIGS = [cfg for cfg in ALL_LANGUAGE_CONFIGS if cfg.get("enabled", True)]
else:
    selected = set(SELECTED_LANGUAGE_PAIRS)
    LANGUAGE_CONFIGS = [cfg for cfg in ALL_LANGUAGE_CONFIGS if cfg["language"] in selected]

# Final safety filter: dataset and model must both support the language.
SUPPORTED_LANGUAGE_CONFIGS = []
SKIPPED_LANGUAGE_CONFIGS = []
for cfg in LANGUAGE_CONFIGS:
    reasons = []
    if not cfg.get("dataset_supported", False):
        reasons.append("not available in selected dataset")
    if not cfg.get("model_supported", False):
        reasons.append("not supported by SeamlessM4T-v2")
    if cfg.get("source_lang_code") is None:
        reasons.append("missing SeamlessM4T language code")

    if reasons:
        skipped = dict(cfg)
        skipped["skip_reason"] = "; ".join(reasons)
        SKIPPED_LANGUAGE_CONFIGS.append(skipped)
    else:
        SUPPORTED_LANGUAGE_CONFIGS.append(cfg)

LANGUAGE_CONFIGS = SUPPORTED_LANGUAGE_CONFIGS
LANGUAGE_LOOKUP = {cfg["language"]: cfg for cfg in LANGUAGE_CONFIGS}

MAX_TEXT_DEV_GLOBAL = max(cfg["max_text_dev"] for cfg in EXPERIMENT_CONFIGS)
MAX_AUDIO_DEV_GLOBAL = max(cfg["max_audio_dev"] for cfg in EXPERIMENT_CONFIGS)
MAX_SCAN_ROWS_GLOBAL = max(cfg["max_scan_rows"] for cfg in EXPERIMENT_CONFIGS)

BASE_OUTPUT_DIR = Path("./" + BASE_OUTPUT_FOLDER_NAME)
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_SUMMARY = {
    "project": PROJECT_NAME,
    "model_id": MODEL_ID,
    "dataset_id": DATASET_ID,
    "split": SPLIT_FOR_EXPERIMENT,
    "device": DEVICE,
    "debug_mode": DEBUG_MODE,
    "skip_audio_debug": SKIP_AUDIO_DEBUG,
    "run_full_grid": RUN_FULL_GRID,
    "run_text_evaluation": RUN_TEXT_EVALUATION,
    "run_audio_evaluation": RUN_AUDIO_EVALUATION,
    "run_data_exploration": RUN_DATA_EXPLORATION,
    "baseline_model_name": BASELINE_MODEL_NAME,
    "baseline_dataset_name": BASELINE_DATASET_NAME,
    "baseline_pipeline_type": BASELINE_PIPELINE_TYPE,
    "audio_optimization_strategies": AUDIO_OPTIMIZATION_STRATEGIES,
    "directions_to_run": DIRECTIONS_TO_RUN,
    "selected_language_pairs": SELECTED_LANGUAGE_PAIRS,
    "active_languages": [cfg["display_name"] for cfg in LANGUAGE_CONFIGS],
    "skipped_languages": [{"language": cfg["display_name"], "reason": cfg["skip_reason"]} for cfg in SKIPPED_LANGUAGE_CONFIGS],
    "experiments": EXPERIMENT_CONFIGS,
}

with open(BASE_OUTPUT_DIR / "00_runtime_config.json", "w", encoding="utf-8") as f:
    json.dump(CONFIG_SUMMARY, f, indent=2, ensure_ascii=False)

print("Project:", PROJECT_NAME)
print("Model:", MODEL_ID)
print("Dataset:", DATASET_ID)
print("Device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("DEBUG_MODE:", DEBUG_MODE)
print("SKIP_AUDIO_DEBUG:", SKIP_AUDIO_DEBUG)
print("RUN_FULL_GRID:", RUN_FULL_GRID)
print("RUN_TEXT_EVALUATION:", RUN_TEXT_EVALUATION)
print("RUN_AUDIO_EVALUATION:", RUN_AUDIO_EVALUATION)
print("RUN_DATA_EXPLORATION:", RUN_DATA_EXPLORATION)
print("DIRECTIONS_TO_RUN:", DIRECTIONS_TO_RUN)
print("Active languages:", [cfg["display_name"] for cfg in LANGUAGE_CONFIGS])
print("Skipped languages:", [(cfg["display_name"], cfg["skip_reason"]) for cfg in SKIPPED_LANGUAGE_CONFIGS])
print("Experiments:", EXPERIMENT_CONFIGS)
print("Output dir:", BASE_OUTPUT_DIR)


# %% [markdown] id="KwcdkUV8Kf_g"
# ### 1.3.2.1 Experiment Setup Metadata for Paper Reporting
#
# This subsection records the reproducibility information required for the experimental setup section of the paper. It does **not** change the experiment or model behavior. It only writes metadata tables and JSON files to the output folder.
#
# Expected outputs:
#
# - `00_runtime_config.json`: exact runtime configuration used by the notebook.
# - `00a_runtime_hardware_metadata.csv/json`: GPU, CUDA, Python, PyTorch, and sampling-rate information.
# - `00b_dataset_usage_plan_debug_vs_full.csv`: sample sizes used in `DEBUG_MODE=True` and the full run (`DEBUG_MODE=False`).
# - `00c_language_support_matrix.csv`: language support across FLEURS and SeamlessM4T-v2.
#
# How to interpret this section for paper writing:
#
# - Use the hardware table to report the computational environment.
# - Use the debug/full plan to explain that debugging uses a small subset, while the full experiment uses larger sample-size configurations.
# - Use the support matrix to justify why languages available in FLEURS may still be disabled when SeamlessM4T-v2 does not natively support them.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 864} id="Vg9lEZfZKf_g" outputId="0b300881-197e-4f02-ea2e-e5efb909192e"
# ============================================================
# Paper setup metadata: hardware, debug/full data plan, support matrix
# ============================================================
# This cell only creates metadata files for reporting. It does not run model inference.

def get_runtime_hardware_metadata():
    """Collect runtime and hardware information for the paper's experiment setup section."""
    metadata = {
        "project": PROJECT_NAME,
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "english_reference_config": ENGLISH_CONFIG,
        "device": DEVICE,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "python_version": os.sys.version.split()[0],
        "target_sampling_rate": TARGET_SR,
        "seed": SEED,
        "debug_mode_current_run": DEBUG_MODE,
        "run_full_grid_current_run": RUN_FULL_GRID,
    }
    if torch.cuda.is_available():
        metadata.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        })
    else:
        metadata.update({
            "gpu_name": "CPU only",
            "gpu_count": 0,
            "cuda_version": None,
            "gpu_memory_total_gb": None,
        })
    return metadata


def build_dataset_usage_plan():
    """Contrast the quick debugging subset with the full experiment settings."""
    rows = [
        {
            "run_mode": "debug_true",
            "experiment": "debug",
            "max_text_dev": 4,
            "max_audio_dev": 1,
            "max_scan_rows": 200,
            "eda_sample_size_per_language_split": 25,
            "purpose": "Fast smoke test to verify loading, alignment, model calls, metrics, and output writing.",
        },
        {
            "run_mode": "debug_false_full_experiment",
            "experiment": "Experiment_1",
            "max_text_dev": 50,
            "max_audio_dev": 10,
            "max_scan_rows": 1000,
            "eda_sample_size_per_language_split": 200,
            "purpose": "Small full-run baseline for fast comparison across languages and optimization strategies.",
        },
        {
            "run_mode": "debug_false_full_experiment",
            "experiment": "Experiment_2",
            "max_text_dev": 100,
            "max_audio_dev": 30,
            "max_scan_rows": 3000,
            "eda_sample_size_per_language_split": 200,
            "purpose": "Medium full-run baseline for more stable metric estimates.",
        },
        {
            "run_mode": "debug_false_full_experiment",
            "experiment": "Experiment_3",
            "max_text_dev": 200,
            "max_audio_dev": 50,
            "max_scan_rows": 5000,
            "eda_sample_size_per_language_split": 200,
            "purpose": "Largest configured full-run baseline for paper-level comparison.",
        },
    ]
    for row in rows:
        row["dataset_id"] = DATASET_ID
        row["model_id"] = MODEL_ID
        row["split_for_experiment"] = SPLIT_FOR_EXPERIMENT
        row["directions"] = ", ".join(DIRECTIONS_TO_RUN)
        row["note"] = "This notebook performs evaluation/inference. If adapted for fine-tuning, keep the same debug/full sampling logic for comparability."
    return pd.DataFrame(rows)


def build_language_support_matrix():
    """Summarize dataset and model availability for each candidate African language."""
    rows = []
    for cfg in ALL_LANGUAGE_CONFIGS:
        reasons = []
        if not cfg.get("dataset_supported", False):
            reasons.append("not available in FLEURS configuration list")
        if not cfg.get("model_supported", False):
            reasons.append("not natively supported by SeamlessM4T-v2 in this notebook")
        if cfg.get("source_lang_code") is None:
            reasons.append("missing source language code")
        rows.append({
            "language_key": cfg["language"],
            "display_name": cfg["display_name"],
            "fleurs_config": cfg.get("fleurs_config"),
            "seamlessm4t_source_lang_code": cfg.get("source_lang_code"),
            "dataset_supported": cfg.get("dataset_supported", False),
            "model_supported": cfg.get("model_supported", False),
            "enabled_in_current_notebook": cfg.get("enabled", False) and cfg in LANGUAGE_CONFIGS,
            "skip_reason": "; ".join(reasons),
        })
    return pd.DataFrame(rows)

runtime_hardware_metadata = get_runtime_hardware_metadata()
runtime_hardware_df = pd.DataFrame([runtime_hardware_metadata])
dataset_usage_plan_df = build_dataset_usage_plan()
language_support_matrix_df = build_language_support_matrix()

runtime_hardware_df.to_csv(BASE_OUTPUT_DIR / "00a_runtime_hardware_metadata.csv", index=False)
with open(BASE_OUTPUT_DIR / "00a_runtime_hardware_metadata.json", "w", encoding="utf-8") as f:
    json.dump(runtime_hardware_metadata, f, indent=2, ensure_ascii=False)

dataset_usage_plan_df.to_csv(BASE_OUTPUT_DIR / "00b_dataset_usage_plan_debug_vs_full.csv", index=False)
language_support_matrix_df.to_csv(BASE_OUTPUT_DIR / "00c_language_support_matrix.csv", index=False)

print("Saved paper setup metadata to:", BASE_OUTPUT_DIR)
print(runtime_hardware_df)
print(dataset_usage_plan_df)
print(language_support_matrix_df)


# %% [markdown] id="ed2a8980"
# ### 1.3.3 Prepare Google Drive Backup
#
# This cell mounts Google Drive so the notebook can save a complete backup of the local result folder. This is important because Colab runtimes are temporary.
#
# Expected output: a successful Drive mount message. At the end of the notebook, the local result directory is copied and optionally zipped into Google Drive.

# %% colab={"base_uri": "https://localhost:8080/"} id="265b5dff" outputId="a5c26483-5946-45c5-f479-858980720fc0"
# =========================================================
# Prepare to Save Aggregated Results Locally + Google Drive
# =========================================================

from google.colab import drive
from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd

# Mount Google Drive
drive.mount('/content/drive')

# %% [markdown] id="7004da5b"
# ### 1.3.4 Load SeamlessM4T-v2 Once
#
# This cell loads the model and processor once, then reuses them for all languages and experiments. Loading once avoids repeated downloads and reduces runtime.
#
# Expected output: confirmation that SeamlessM4T-v2 has loaded successfully. If this cell fails because of GPU memory, reduce the number of enabled languages or restart Colab with a larger GPU.

# %% colab={"base_uri": "https://localhost:8080/", "height": 458, "referenced_widgets": ["417603b36c4a4358b7d17571af7fc0f8", "c11fd2461970470bbee4a86013161024", "f72858c7c9794e58bbabb37f4b317369", "bd420cd9384b43289a30aacdd193e612", "3c7d21db77b8405dac3d70be31e4f837", "4eba09ba68c548b39f6985f0a246184f", "ebdb31fc139e4910b61c59c122943acf", "1706e81cd8a64f7aa0f5021d2470bfc5", "55ad5e5a35aa4e6d8ba4839f5e1cfc97", "8fc81a94db7d4beeb088a5aae1e5384f", "7d0a463073f24c0ab792a33662706623", "a3197d555dc64e2aa02f6fa36718423c", "3b9ffdc1b7e145d8aaa3a9fcfa506fde", "d4aa494ac05b4ae78a617c341eefe6e1", "66fbd09599024a56808447d5ad1a1b52", "622005d9d2d249bbba31da7e881367e1", "1d5f809bf95844a9a094f093e65acd6f", "7ed4f375a98e4b908cd07b400aea269b", "4b9466fe6b7c4c01bee2923283d1377a", "4499998f599c4ed7840bc29307e23317", "2d40455dcf1246f888dc9abb765de551", "5f7ad3db72a34a53b35faa9d62912cc9", "847a2d82cd8b40598c1e4002cca9d41f", "5aeb6ce147884fa5bb92221d59516fc3", "8aaeb1d7cde0411b973d5a4523a4325c", "37e8a9916a8f4005ae534e26c89016a8", "c252306ffee748529254645c8573ecae", "8e7aba8c6cd044748e101a3597345293", "57e61e2fc42d4e21987c560e3ceebff9", "20028d51685d46fea902d80946cd519c", "67f5db8059d6440fa9f19ebdae868796", "d041acda9e1641159d4899b0c174d32f", "751e1f49f2bf4e9dafd60d3191e1ffc2", "9843b1c582584323bfe34a49ab6abd6b", "33e271964c0a4a958be83af3fa35f2b5", "5a904df6428242d4a3676b122eed11ed", "0f939ef4358a4e67a3e1c52e6c2ec390", "03dbe5dc7a7b4b0dafec555c65fe4f61", "65263b9bb6b34671ac853d38a478cbec", "8f0d1307ecc14154a46c5784cc2df35c", "1d96cc1892ca4718b7bf47b9c7cb72c6", "79ccd64690224eb59b89df1f52e43cd6", "49b76ecaa45a440bb4cd561736e94a0b", "659c1ba94327492782359e8648184ef0", "f74fda27e74f40b3b570e72f404ec66e", "bf21fb79aec14f6f822cfe05e050405d", "2b5ca64edf8f4d388086428c8a6b0ab3", "999aa32e1ce246a289b98cacb4b8eb2d", "9d8a7a43a5b248c8b1c3276b6959b81b", "891f0e608f484505ba0a80f1d580d79f", "cf76ac497f3e47e38ea271f2e4be1519", "b43af984c3984a7eaf9bbbba001674b3", "6729360f3f9445b28f340cc45868c5a0", "6d68cb3219174cce9cb4e70be59abf1b", "526e43b6e3ef4344a54b468863856a98", "49736b2a797e4b42a2d6b7c482ace0d2", "ad68112d83a54bfd97df371dc3d88aa4", "60942580a0174160aacf240ed931a153", "93777e6cef9e4dac9d3bf3297e34e81a", "28881417f8614031abb5f391ba08fbd0", "e8ad6cb97c854d578615e1a8752bd3e5", "2364e5aaa66a40538b0a8114dc8f9e36", "715cfd3762b949dcaa525e17ae7ed977", "d1f456e6638641d099a85e7b9060ddd1", "2f1fa9df207740a78084bfd929bafd16", "05e4b238f7d64943a8f0f72565c5e4c2", "4ac37dd9e044454b8daaa572d976551a", "d14c633177d94637a71a1f4b7db7f334", "ef2d378aa397492c81e4e0487bef82aa", "28c112caac074570aa7c53147dc45d90", "5303fbb36b1c49839a804b54b14e3289", "3a8de21097934a96afe57e68d05f5413", "7871432b1f1e4f11b38ae0bf8a4555bf", "b6bed50f797f4c9db8caf44cc1efb7be", "3b0ce5bf16754ad7b89e32e3612cb01d", "970dd5090e4a495ba5a1f621101cc3e5", "098e14b01a4b4719b6e08cb21ef7024c", "6acf1b998acd40519f46b80d313a7e03", "7ed8677fad894898920cc79287072e6a", "7fa0ed2793e643ce947d0f1b11eb006a", "975187c78f5149caac2a55d274368c42", "26686128268746619f2bb9c5d30406a8", "efabfb1f1de54bb8893bb3d1144a3a42", "01c2301665ab444fb57de4f6649f0744", "a1e11176011e42b6a704ab97fec6f261", "c16138d24c1c41c2a573959003c59316", "27dcd855fbc248379e1e2587113930df", "bfaf264a31ef49bba5c629dada683ac9", "43cb9cab870346959e5a8616e10a9463", "45ecad9e58fe4f72836405659a606d0e", "384422124adf4f9799ff97fb16e43ebe", "873bbc82691d4bd6b6ed2737bfe4226d", "dac0153d0cf14e29b0ddde057715abd0", "5d027271c2544c698e4f5a1238e09dca", "0fcb4578f5c74872873da2be4a19e892", "6b7c1263798d483d868baa8cf3b14a1a", "7eedf6c7cfd4494c882a247e9dc60ece", "ba91b7554bd0486da54db43b055df5f6", "371a2847f63f48d78676a34649ba925b", "73b1e50c7c7a436597b0b3283dab1d15", "30d3095bfaeb4e4089eafe1f73ae16ea", "61ee43f46cd8430cbdaca510396559f7", "28cc1b8155b742fcbace2af9a67fdafc", "7118437294bb42a387420eb3a25fd591", "673b6beda1f143909041a2c25c3f231a", "4f6e48916df44c26bd1cf6721fccfbc3", "27b3d43feff5416f8d41984067c88f77", "ddf0508e2bf244cc866636a1648897bc", "a181628056794fcfb5550b27fff0b8f0", "2a97c9444a73449486723c6da70f2e99", "9c435db5f10f4954ae81cf1832f1d2cc", "8295fece29bb4ecbbaefa2188c55d9cd", "e7578b3e7b2e4b84a6f4f0a465b651fa", "0a2d5ee5c41e43a486b7c1f4f1a086e6", "9d9161d15c384a2884177a804e41256c", "a3dad30202a74388920c64d802d289dc", "b91f3823a2b54de0ac3422e306ea835f", "52a2b14ca4f14639b4e62857fd0d36b4", "9c5c3e8d90644656a5b580dc66ff5e98", "41773b5284914516b8814d7b75194324", "755f5c237f8444558fcd90924488dda7"]} id="e8e27f8c" outputId="f7988295-f9bb-4094-a656-9a00277cb67e"
if not LANGUAGE_CONFIGS:
    raise RuntimeError("No active supported language pairs. Check SELECTED_LANGUAGE_PAIRS, RUN_FULL_GRID, and model_supported flags.")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = SeamlessM4Tv2Model.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print("Loaded model on", DEVICE)


# %% [markdown] id="wr4JsQuOKf_h"
# ### 1.3.5 Model and Dataset Statistics for Paper Reporting
#
# This subsection extracts model metadata and FLEURS dataset metadata after the model is loaded. These files help the team write a reproducible experimental setup section and compare this baseline with future notebooks using other model/dataset pairs.
#
# Expected outputs:
#
# - `00d_model_metadata.json`: model identifier, parameter count, architecture information, and generation settings.
# - `00e_model_supported_language_codes.csv`: language codes detected from the SeamlessM4T processor/tokenizer when available.
# - `00f_fleurs_dataset_configs.csv`: available FLEURS language configurations.
# - `00g_fleurs_selected_language_split_statistics.csv`: split sizes for the active African languages and English reference configuration.
# - `00h_experiment_setup_summary_for_paper.csv`: compact table combining model, dataset, hardware, language support, and configured sample sizes.
#
# How to interpret this section:
#
# - The model table describes the baseline system.
# - The dataset table describes the speech-to-text translation data source.
# - The selected-language split table documents the data available for the exact languages used in this notebook.
# - The setup summary can be adapted directly into the methodology section.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["65efc2aec8a04f9cba85c8c35315c784", "63e57ee0ebea4605ac848f054e35a538", "824a0dfe5d674127a529c4ab763f06b5", "1220a911098f4677b55c9bf04dd1aa34", "bd122b24600e40f3bdf8ee2cb8b65505", "63e0ef3ae9c143d68d39d86013e5278c", "e3149ae31add4f0d99ee2034f402ea89", "01105b50934e4614a6c1e14cd461f81a", "754166c6462b4835a1b997f3f97d4add", "e9fd30069a6a4e2fb951593e78716cbe", "bc2e2f19b42745f281ab3c1dfa1ccf19"]} id="upPI4qlNKf_h" outputId="f8e4fbd5-7139-4fc1-efd5-628e1cf4b694"
# ============================================================
# Model and dataset statistics for paper reporting
# ============================================================
# This cell inspects model/dataset metadata. It does not change model behavior.

def count_model_parameters(model_obj):
    """Return total/trainable parameter counts for the loaded model."""
    total = sum(p.numel() for p in model_obj.parameters())
    trainable = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "total_parameters_millions": round(total / 1_000_000, 2),
        "trainable_parameters": int(trainable),
        "trainable_parameters_millions": round(trainable / 1_000_000, 2),
    }


def extract_model_language_codes(processor_obj):
    """Try to extract supported language codes from the SeamlessM4T processor/tokenizer.

    Transformers versions expose language codes under different attributes.
    This defensive function keeps the notebook robust in Colab.
    """
    candidate_attrs = ["lang_code_to_id", "language_code_to_id", "supported_language_codes", "supported_languages"]
    tokenizer = getattr(processor_obj, "tokenizer", None)
    found = None
    source = None
    for obj_name, obj in [("processor", processor_obj), ("tokenizer", tokenizer)]:
        if obj is None:
            continue
        for attr in candidate_attrs:
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                if isinstance(value, dict):
                    found = sorted(list(value.keys()))
                elif isinstance(value, (list, tuple, set)):
                    found = sorted(list(value))
                if found:
                    source = f"{obj_name}.{attr}"
                    break
        if found:
            break
    if not found and tokenizer is not None:
        vocab = getattr(tokenizer, "get_vocab", lambda: {})()
        candidates = []
        for token in vocab.keys():
            clean = str(token).replace("__", "").replace("<", "").replace(">", "")
            if len(clean) == 3 and clean.isalpha():
                candidates.append(clean)
        found = sorted(set(candidates))
        source = "tokenizer_vocab_fallback"
    return pd.DataFrame([{"language_code": code, "detected_from": source or "not_detected"} for code in (found or [])])


def collect_model_metadata(model_obj, processor_obj):
    """Collect model metadata for the paper's experimental setup section."""
    param_info = count_model_parameters(model_obj)
    config = getattr(model_obj, "config", None)
    generation_config = getattr(model_obj, "generation_config", None)
    metadata = {
        "model_id": MODEL_ID,
        "baseline_model_name": BASELINE_MODEL_NAME,
        "pipeline_type": BASELINE_PIPELINE_TYPE,
        "model_class": model_obj.__class__.__name__,
        "processor_class": processor_obj.__class__.__name__,
        "torch_dtype": str(next(model_obj.parameters()).dtype),
        "device": DEVICE,
        "architectures": getattr(config, "architectures", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "decoder_start_token_id": getattr(config, "decoder_start_token_id", None),
        "forced_bos_token_id": getattr(generation_config, "forced_bos_token_id", None),
    }
    metadata.update(param_info)
    return metadata


def collect_fleurs_config_metadata():
    """Collect the list of FLEURS configurations available through Hugging Face Datasets."""
    try:
        configs = get_dataset_config_names(DATASET_ID)
    except Exception as e:
        print("Could not list FLEURS configs:", e)
        configs = []
    return pd.DataFrame({"dataset_id": DATASET_ID, "config_name": configs})


def collect_selected_fleurs_split_statistics():
    """Collect split-level metadata for active languages and the English reference config.

    This uses dataset-builder metadata where possible, so it is faster than scanning all audio files.
    """
    configs_to_check = [(cfg["display_name"], cfg["fleurs_config"], cfg["language"]) for cfg in LANGUAGE_CONFIGS]
    configs_to_check.append(("English reference", ENGLISH_CONFIG, "english"))
    rows = []
    for display_name, fleurs_config, language_key in configs_to_check:
        try:
            builder = load_dataset_builder(DATASET_ID, fleurs_config)
            split_info = builder.info.splits
            for split_name, split_meta in split_info.items():
                rows.append({
                    "dataset_id": DATASET_ID,
                    "language_key": language_key,
                    "display_name": display_name,
                    "fleurs_config": fleurs_config,
                    "split": split_name,
                    "num_examples": getattr(split_meta, "num_examples", None),
                    "num_bytes": getattr(split_meta, "num_bytes", None),
                })
        except Exception as e:
            rows.append({
                "dataset_id": DATASET_ID,
                "language_key": language_key,
                "display_name": display_name,
                "fleurs_config": fleurs_config,
                "split": "metadata_error",
                "num_examples": None,
                "num_bytes": None,
                "error": str(e),
            })
    return pd.DataFrame(rows)

model_metadata = collect_model_metadata(model, processor)
model_language_codes_df = extract_model_language_codes(processor)
fleurs_configs_df = collect_fleurs_config_metadata()
fleurs_selected_split_stats_df = collect_selected_fleurs_split_statistics()

with open(BASE_OUTPUT_DIR / "00d_model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(model_metadata, f, indent=2, ensure_ascii=False)
model_language_codes_df.to_csv(BASE_OUTPUT_DIR / "00e_model_supported_language_codes.csv", index=False)
fleurs_configs_df.to_csv(BASE_OUTPUT_DIR / "00f_fleurs_dataset_configs.csv", index=False)
fleurs_selected_split_stats_df.to_csv(BASE_OUTPUT_DIR / "00g_fleurs_selected_language_split_statistics.csv", index=False)

setup_summary_rows = []
for cfg in ALL_LANGUAGE_CONFIGS:
    setup_summary_rows.append({
        "project": PROJECT_NAME,
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "baseline_model": BASELINE_MODEL_NAME,
        "baseline_dataset": BASELINE_DATASET_NAME,
        "pipeline_type": BASELINE_PIPELINE_TYPE,
        "language": cfg["display_name"],
        "language_key": cfg["language"],
        "fleurs_config": cfg.get("fleurs_config"),
        "seamlessm4t_code": cfg.get("source_lang_code"),
        "dataset_supported": cfg.get("dataset_supported"),
        "model_supported": cfg.get("model_supported"),
        "enabled_for_current_run": cfg in LANGUAGE_CONFIGS,
        "debug_mode_current_run": DEBUG_MODE,
        "full_grid_current_run": RUN_FULL_GRID,
        "device": DEVICE,
        "gpu_name": runtime_hardware_metadata.get("gpu_name"),
        "gpu_memory_total_gb": runtime_hardware_metadata.get("gpu_memory_total_gb"),
        "target_sampling_rate": TARGET_SR,
        "min_duration_seconds": MIN_DURATION,
        "max_duration_seconds": MAX_DURATION,
        "directions_to_run": ", ".join(DIRECTIONS_TO_RUN),
        "optimization_strategies": ", ".join([s["strategy_key"] for s in AUDIO_OPTIMIZATION_STRATEGIES if s.get("enabled")]),
        "debug_dataset_samples": "text_dev=4; audio_dev=1; max_scan_rows=200",
        "full_dataset_samples": "Experiment_1 text=50 audio=10; Experiment_2 text=100 audio=30; Experiment_3 text=200 audio=50",
    })
setup_summary_for_paper_df = pd.DataFrame(setup_summary_rows)
setup_summary_for_paper_df.to_csv(BASE_OUTPUT_DIR / "00h_experiment_setup_summary_for_paper.csv", index=False)

print("Model metadata and FLEURS dataset statistics saved to:", BASE_OUTPUT_DIR)
print(pd.DataFrame([model_metadata]))
print(model_language_codes_df.head(20))
print(fleurs_configs_df.head(20))
print(fleurs_selected_split_stats_df)
print(setup_summary_for_paper_df)


# %% [markdown] id="22d30558"
# # 2 Data Preparation
#
# This section prepares text and audio examples in a uniform format so that later evaluation functions can operate independently of the dataset source.
#
# For FLEURS, language data is organized by language configuration. The notebook aligns African-language examples with English references where possible and standardizes the resulting examples into fields such as `source_text`, `target_text`, `audio`, `duration_sec`, and `text_id`.
#
# Expected output: prepared datasets for text translation and audio translation, plus sample-check CSV files for manual validation.

# %% [markdown] id="8d2f4610"
# ## 2.1 Data Acquisition
#
# This stage loads the selected dataset splits and language configurations. For FLEURS, each language is loaded through its language-specific config, such as Igbo, Yoruba, Swahili, Wolof, or Hausa where supported by the dataset and model.
#
# Expected output: dataset objects or streamed examples for the enabled languages. If a language has no data or is not supported, the language configuration should be disabled before the full run.

# %% [markdown] id="714845fe"
# ## 2.2 Initial Sanity Check
#
# The sanity check runs a few short text translations before the full experiment. This confirms that the model accepts the selected language codes and produces a plausible output.
#
# Expected output: short input sentences and model predictions. These are not formal evaluation results; they are a quick check for obvious problems such as wrong language codes, empty outputs, or unsupported directions.

# %% [markdown] id="ed697ac3"
# ## 2.3 Data Loading Helpers
#
# These helper functions normalize text, extract dataset fields, resample audio, normalize waveforms, trim edge silence, compute energy, and compute silence ratio.
#
# Interpretation note: these functions are part of the preprocessing pipeline. Keep them consistent across baseline notebooks if you want fair comparisons between models and datasets.

# %% id="8d15003e"
def get_text(item):
    """Return the best available transcript text from a FLEURS item."""
    return str(
        item.get("transcription")
        or item.get("raw_transcription")
        or item.get("sentence")
        or ""
    ).strip()


def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resample_audio(audio, sr, target_sr=TARGET_SR):
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)


def normalize_audio_waveform(audio):
    audio = np.asarray(audio, dtype=np.float32).flatten()
    audio = np.nan_to_num(audio)
    if len(audio) == 0:
        return audio
    audio = audio - np.mean(audio)
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs
    return audio.astype(np.float32)


def trim_silence(audio, threshold=0.01):
    """Trim silence safely. If trimming makes audio too short, keep original audio."""
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == 0:
        return audio

    min_audio_samples = int(MIN_DURATION * TARGET_SR)
    non_silent_indices = np.where(np.abs(audio) > threshold)[0]

    if len(non_silent_indices) == 0:
        return audio

    trimmed = audio[non_silent_indices[0]: non_silent_indices[-1] + 1]

    # Important: avoid SeamlessM4T spectrogram crash on ultra-short clips.
    if len(trimmed) < min_audio_samples:
        return audio.astype(np.float32)

    return trimmed.astype(np.float32)


def ensure_min_audio_length(audio, min_seconds=MIN_DURATION, sr=TARGET_SR):
    audio = np.asarray(audio, dtype=np.float32).flatten()
    audio = np.nan_to_num(audio)
    min_len = int(min_seconds * sr)
    if len(audio) == 0:
        return np.zeros(min_len, dtype=np.float32)
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))
    return audio.astype(np.float32)


def audio_duration(audio, sr=TARGET_SR):
    audio = np.asarray(audio, dtype=np.float32).flatten()
    return len(audio) / float(sr)


def load_fleurs_split(config, split=SPLIT_FOR_EXPERIMENT, streaming=True):
    return load_dataset(DATASET_ID, config, split=split, streaming=streaming)


def direction_is_enabled(direction_key):
    return direction_key in DIRECTIONS_TO_RUN


def prepare_bidirectional_pairs(language_key, split=SPLIT_FOR_EXPERIMENT, max_samples=50, max_scan_rows=5000):
    """Create aligned text/audio pairs from FLEURS using row-order alignment.

    Returns rows for selected directions:
    - african_to_english
    - english_to_african
    """
    cfg = LANGUAGE_LOOKUP.get(language_key)
    if cfg is None:
        return []

    afr_stream = load_fleurs_split(cfg["fleurs_config"], split=split, streaming=True)
    eng_stream = load_fleurs_split(ENGLISH_CONFIG, split=split, streaming=True)

    rows = []
    base_sample_count = 0

    for idx, (afr_item, eng_item) in enumerate(zip(afr_stream, eng_stream)):
        if idx >= max_scan_rows:
            break

        afr_text = normalize_text(get_text(afr_item))
        eng_text = normalize_text(get_text(eng_item))

        if not afr_text or not eng_text:
            continue

        if direction_is_enabled("african_to_english"):
            rows.append({
                "sample_index": idx,
                "language": language_key,
                "language_display": cfg["display_name"],
                "direction_key": "african_to_english",
                "direction": f"{cfg['display_name']}→English",
                "source_lang_code": cfg["source_lang_code"],
                "target_lang_code": "eng",
                "source_text": afr_text,
                "target_text": eng_text,
                "source_audio": afr_item["audio"],
                "target_audio": eng_item["audio"],
            })

        if direction_is_enabled("english_to_african"):
            rows.append({
                "sample_index": idx,
                "language": language_key,
                "language_display": cfg["display_name"],
                "direction_key": "english_to_african",
                "direction": f"English→{cfg['display_name']}",
                "source_lang_code": "eng",
                "target_lang_code": cfg["source_lang_code"],
                "source_text": eng_text,
                "target_text": afr_text,
                "source_audio": eng_item["audio"],
                "target_audio": afr_item["audio"],
            })

        base_sample_count += 1
        if base_sample_count >= max_samples:
            break

    return rows


def save_dataframe(df, filename):
    path = BASE_OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print("Saved:", path)
    return path



# %% [markdown] id="531026f6"
# ## 2.4 Bidirectional Sample Check
#
# This section creates a small CSV of short bidirectional sample sentences for every enabled language pair. The goal is human inspection, not automatic scoring.
#
# Expected output: `01_fleurs_bidirectional_sample_check.csv`. The file contains African-language text, English text, direction labels, and model predictions where available.
#
# How to interpret: use this file to quickly check whether the model is producing the correct language and broadly correct meaning before running expensive audio experiments. These examples can also support the qualitative discussion in the paper.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="9b106553" outputId="6b218973-ddef-437e-c8b8-dfc0f41d837a"
if RUN_SAMPLE_CHECK:
    # ------------------------------------------------------------------
    # Human-checkable sample data for EVERY African↔English pair.
    # These are intentionally SHORT examples so they can be checked quickly
    # in Google Translate or another reference translator before running
    # the full FLEURS/SeamlessM4T experiment.
    #
    # Note: these rows are sanity-check examples for the CSV
    # 01_fleurs_bidirectional_sample_check.csv. The actual experiment still
    # uses FLEURS audio/transcripts through prepare_bidirectional_pairs().
    # ------------------------------------------------------------------

    SHORT_BIDIRECTIONAL_CHECK_SAMPLES = {
        "igbo": [
            {
                "african_text": "Ndewo, kedu ka ị mere taa?",
                "english_text": "Hello, how are you today?"
            },
            {
                "african_text": "Aga m ụlọ akwụkwọ echi.",
                "english_text": "I will go to school tomorrow."
            },
            {
                "african_text": "Daalụ maka enyemaka gị.",
                "english_text": "Thank you for your help."
            },
        ],
        "yoruba": [
            {
                "african_text": "Báwo ni ọjọ́ rẹ ṣe rí lónìí?",
                "english_text": "How is your day today?"
            },
            {
                "african_text": "Mo fẹ́ lọ sí ilé ẹ̀kọ́ lọ́la.",
                "english_text": "I want to go to school tomorrow."
            },
            {
                "african_text": "Ẹ ṣé fún iranlọwọ yín.",
                "english_text": "Thank you for your help."
            },
        ],
        "swahili": [
            {
                "african_text": "Habari yako leo asubuhi?",
                "english_text": "How are you this morning?"
            },
            {
                "african_text": "Nitaenda shuleni kesho.",
                "english_text": "I will go to school tomorrow."
            },
            {
                "african_text": "Asante kwa msaada wako.",
                "english_text": "Thank you for your help."
            },
        ],
        "wolof": [
            {
                "african_text": "Na nga def ci suba si?",
                "english_text": "How are you this morning?"
            },
            {
                "african_text": "Dinaa dem jangaji ëllëg.",
                "english_text": "I will go to school tomorrow."
            },
            {
                "african_text": "Jërëjëf ci sa ndimbal.",
                "english_text": "Thank you for your help."
            },
        ],
        "hausa": [
            {
                "african_text": "Sannu, yaya kake yau?",
                "english_text": "Hello, how are you today?"
            },
            {
                "african_text": "Zan je makaranta gobe.",
                "english_text": "I will go to school tomorrow."
            },
            {
                "african_text": "Na gode da taimakon ka.",
                "english_text": "Thank you for your help."
            },
        ],
    }

    sample_rows = []

    # Use ALL_LANGUAGE_CONFIGS deliberately, not LANGUAGE_CONFIGS, so the CSV
    # shows all African languages and both directions even in debug mode.
    for cfg in ALL_LANGUAGE_CONFIGS:
        language_key = cfg["language"]
        examples = SHORT_BIDIRECTIONAL_CHECK_SAMPLES.get(language_key, [])

        for sample_index, ex in enumerate(examples):
            african_text = ex["african_text"]
            english_text = ex["english_text"]

            sample_rows.append({
                "sample_index": sample_index,
                "language": cfg["display_name"],
                "language_key": language_key,
                "direction": f"{cfg['display_name']}→English",
                "direction_key": "african_to_english",
                "source_lang_code": cfg.get("source_lang_code"),
                "target_lang_code": "eng",
                "source_text": african_text,
                "target_reference": english_text,
                "dataset_id": DATASET_ID,
                "fleurs_config": cfg.get("fleurs_config"),
                "dataset_supported": cfg.get("dataset_supported", False),
                "model_supported": cfg.get("model_supported", False),
                "enabled_in_full_experiment": cfg.get("enabled", False),
                "sample_type": "short_manual_sanity_check",
                "notes": "Short reference example for checking translation direction before running FLEURS audio experiments.",
            })

            sample_rows.append({
                "sample_index": sample_index,
                "language": cfg["display_name"],
                "language_key": language_key,
                "direction": f"English→{cfg['display_name']}",
                "direction_key": "english_to_african",
                "source_lang_code": "eng",
                "target_lang_code": cfg.get("source_lang_code"),
                "source_text": english_text,
                "target_reference": african_text,
                "dataset_id": DATASET_ID,
                "fleurs_config": cfg.get("fleurs_config"),
                "dataset_supported": cfg.get("dataset_supported", False),
                "model_supported": cfg.get("model_supported", False),
                "enabled_in_full_experiment": cfg.get("enabled", False),
                "sample_type": "short_manual_sanity_check",
                "notes": "Short reference example for checking translation direction before running FLEURS audio experiments.",
            })

    sample_check_df = pd.DataFrame(sample_rows)
    save_dataframe(sample_check_df, "01_fleurs_bidirectional_sample_check.csv")
    print(sample_check_df)
else:
    sample_check_df = pd.DataFrame()
    print("Sample check skipped.")


# %% [markdown] id="e3b61690"
# ## 2.5 Preparing Transcript-Based Translation Data
#
# This section prepares text-only translation pairs. Text-only translation isolates the translation component from the speech understanding component.
#
# Expected output: paired source and English text examples for each enabled language.
#
# How to interpret: strong transcript translation but weak direct audio translation suggests that the model can translate the language when text is available, but struggles with the acoustic signal.

# %% [markdown] id="80628444"
# # 3 Exploratory Data Analysis
#
# This section generates dataset properties, descriptive statistics, compact paper-ready summaries, and visualizations before model evaluation.
#
# Expected outputs include:
#
# - Text length distributions.
# - Audio duration distributions.
# - Audio duration buckets.
# - Source/reference word-count summaries.
# - Audio energy distributions.
# - Silence-ratio distributions.
# - Per-language CSV summaries.
#
# How to interpret: these plots explain the difficulty of the dataset. Longer audio, high silence ratios, low energy, and large variation in sentence length can all affect speech translation quality. Use this section to justify preprocessing and optimization choices in the paper.

# %% [markdown] id="4101b86b"
# This section generates dataset properties, descriptive statistics, compact paper-ready summaries, and visualizations before model inference.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["3fbc583130c643959f8625cc2ca3a85e", "c6f4873798264dd095b0badbdfaa3b27", "7dcc15f6521f4ae789c1f32284a478fe", "92bc64586fc74deca6b727383010347e", "0fc75869e2f741d1bdb657d1bf58c74f", "845f29f4e3ef4788afceb4456492b13c", "21e4d659da9146dfb2816b9390d3048e", "e61fc5c47e914db69d6c31a61aff0c6c", "f8ade6a6b6614790af142a98eddd9b19", "46239103b50141dab347daf082951c12", "ee533e9c268c4e799a806ba86f9b0633", "5e1730d1ff124a15bb2a18f63de60119", "f4c7e098b699407b95e62f4f710a4225", "b5ae9eac52fc4161a03ee112d472d951", "9812ec31ff374f00a9c0b1f0d13539f7", "734b37c784b5439f837cf6ebd80e9cbd", "92313a76ee1744dd8b279c61332c8ca4", "121477e140b74c8cbe0e70bf6e4f650a", "acf2c7d8be9343e7bfe16b9209b79a5b", "5afbb40e230a4b098810202c4d80cb09", "c00072495b9e49a2b0d0888c050d0261", "12953afe23384506b186fb3caad3fd4b", "79af333ba91c4421a225dfc18adc01c5", "31eda748526d4c0a91c126b21e872fc4", "31f443d0d46148c0a570814315d76d91", "ffcf944faddd47e9bf68d426eb192c07", "6caf3259400f4d83b873d9288c068bbb", "11692bb594244d2495402d93d5e03fd3", "b6c6fd08313b4c47ac1e62ca825b5a20", "b470241ed23d441898dc7ea701b2b3f3", "d95d2c73a01b4255834a4eaa21c50f0b", "73e5c698e8924856a7ced0aa9f5eec18", "6bde8d5f480a440eb584c9bc301c0746", "67bc7bc4117f440bb94e066e7b98fe01", "dadfe3fedc1d44e590cc84abb14fcca2", "a42e634b37024613a3930df32130ebc5", "3dffbfb772a543efaaea2a3276956815", "ca34ef2120d14747aba841d4cc7a007a", "509bdcf202954057befc5a7987c23235", "d803de3eb7864e07885eedb3ee5aca8b", "2fe3fb1746fd4b05a4f0e56b724679d8", "4fd1999ee8c74af3b39568fcbda26ee3", "8469fa78928242ff9dde45769d1737fb", "1ccb2ce7d5bf4c83b9d4a9fb8d3cab5e", "f3421d999174444b82bf239bffd00d3c", "4800fe2a6d87415a898d619125cdd7cc", "74a4c08b4d9448f49fce13e1af9e684f", "5489887e82ec462a888b09258807fe1d", "231fa97c217746408efb51d7c88bf99b", "9c7eb970903340e4a859df78a5bc2fd2", "fedf05fe6850461989afd323bf704dcd", "12a0b6478ee4400781a22e8324f0aa8e", "d320d60155714d81a5c1916ad71b5f6c", "62631e5c42ea4baa88fa4c17ddfb94a1", "f181560d4edc40a5994209237e7c5fa9", "37a06d23e34a4ca99c2bf9f10de699e1", "b0a6e18e0b3b45c8b1f93dcb01a6374a", "ea8fdddd64f3415f986925f88dcba178", "40a9a88459e14bff92bd69a228fe1503", "462ecf11dea749f9853b5da3da3c2b80", "6600f456c00c40fc9b2190fe1dfe8fd7", "6aa1411040ce4189a72a1be8423057c7", "3568e1dce4514efcb6cc8ac3c9ab21fa", "cd72fc2cae8c45a2a6ec116e02d018f1", "efeea301c5b948eaab2d74ab31d132d9", "fac6b183eaa645afa1099ad6571cfd0a", "be232e3ab52545d9b73c5d5f874e1d26", "2b10d03455724670b7b41f389b5322d1", "f8874955ac6246a58c3a442b9b101a0d", "2ca822be558145e7946b98224afdf5c8", "ace455e379e04e44b983ee967aba6278", "b6a592786af04f0f83b300868ce70c2f", "b08e829038304829b862fd013107c1a2", "d91a5b0c65194f19b587beafb570bf3c", "bb8382de859a4e678ab48c9715e29bba", "2b4e6e217a674ed182c2e4f36768964f", "0b33ab17884b480ebf3a82da996edb54", "4948e0cb289942609c5ee7343b8ed1c1", "7700b3fbfc1c4232bfebc5119f6d0fa5", "7ff03ee7b8824251a250ef9c1912f89a", "4016a905831c4f948d49d0eeeb3cd15a", "6e6a6250f77d41d8a3285ef02918afd3", "8b92e0d18ecd4b0796f1a52c79f5c211", "2099e49292bf4527b686fc5b323815d8", "925e85b15868472c8534b2961dbf3662", "7e4d6aeafc1b407ba36bb0f93ee52ffa", "1dbf09b5e7f24139bc87f7a8e9f559be", "180df45a8de840a79e93796b129f34fc", "e74c2621d89544af9383f0ecd2333adf", "df06b2d56e204b2093f9f5e42b4ae105", "1806270f270042d5bac8ce8d72af48c7", "ec224e38fcb243a8adb03943a0bc53ee", "38ef36393743485a800ac7d9da2d4ad5", "930f3b4805ee42e38bea2677feb3880f", "b0e4085fd5fe4ba8a33df9eff1d2c085", "07b4f81992954e65b1b94179d6d9ddf7", "0e9d3c4a275b4f28959a99a979652129", "ff3847cda7604c34afb0e76e901b6eb1", "41aedaccf795425995e25bad9d69a9d2", "909c6bb5193f4abdafa7e2342f8c3aa9", "d0f99965df05484aac9ccd470b8a9b67", "88ac7add71394bb3af74bacb77c10398", "da462205814e4f83b7468540e2e61545", "047f6fd0643a4d2899d8b81e584fb99b", "5779230fa23049eb988fab197b75ae3d", "d8433d0ab7f24fccac7aec06aec9759f", "b651e2745658413590c60655d5c06b37", "69ab68bb0cc246a6a5af458998b4f663", "d2e225e054b142e38427e9d347225036", "949bd4f392d646c4a8f01a16f1f88cae", "55436ac39ab749a8ac21795540c23138", "77ce2a5889c94a1680239a9213d1708d", "0fb9082cde9a4830aaf514d80253cd62", "16a14eea7d4c4a9eaa48dd1eaff1baf8", "e7d2ff2bbe2e47d2b681dd90f48f9160", "02027f184a124a1fb6127490449f9d01", "d84ab0c5de184b76a78a68e8d5b64f44", "a883fde8c07f4e4dac22eb59d07d6ee6", "19f2bfb686ae46588e446035e7333aaa", "593048288dbc4bb382a0ebcfaaecb071", "32229544324e4d92ae9a2b7425ee7e27", "675c70a8e37844aa8a05cf5c1c4e9487", "8f26bb6df7a640b488b4c2cd6b2eaea3", "72a832d1c66b40ce8909beb331086113", "1ee54e4d12ca4c3180516fea3da08ee7", "c7dbca9d17894e0c97680b1b5fd71c6f", "70b619de11b74ed4a7eb863b88dab67d", "f1ebf1e55fd2439ca914719b607fcf97", "a8409c8a532d4625bcf713f24ed26bc2", "97de564ce36942ccaa48243b94adc55f", "f1b9b6f43c4a4622b333ce7d9a4c6242", "2a6c6feb2a114da7b145c95d84fa5e2d", "6da8adde6a7b484ea3ed838a1324c3ca", "e26828ac61cd4982abb15ca98adda17b", "f0f95c4e406a4745883547ae934f9412", "00d5206f219240419623139c0bff1add", "19f3860b4f234535b9b96ced36b12aed", "e6d35b1028c749089b87ffbc893cf6cb", "d2647e179a3a496b9a96e2d99c218a50", "9dcf7d5a18034d4b987b4e02eb3de7aa", "1ae83a48b3844f9596998596c1d1c8c3", "93ab6d73c0e14c21b51d9a32a116ca1d", "01f42794fbfd4dea9437a9fbdea045fc", "e6547d5a03514bf8827b1f5c3f1eca6c", "30acc5a8d2974d39b9489dfeb4bc4eee", "31321e2a674249a1a2b9d8ae88ec4cb0", "a01172bb5bd84115b37eca28bfff34a6", "99c34db861a74737873495b54683531a", "039567504b6d4bc98868a57f218ab0db", "0a663d8a4425472eab5d669451616458", "412ef21e166c4fe28cc283fc2c69589e", "4b5aecb05c1f4675bdab3ab824f5febb", "46424127145748df902daff4b6622ff2", "fcd9e67b5405498d9398a977913effc6", "0d8ef9dadba74ef8aa65be0573bfebfd", "831c17b6665845c88475c20ff170ebf8", "39d420eb57ca4b82a36df5b8d5c6143d", "4e22e878ecd641bc80815e621e9601df", "6c1d795ee81c44aea0bf36eacbdaacf2", "051b4613ac1d4b0397bdfc941abc96e2", "61be712d18df4ee3a5bc3752edff36a6", "a1b68695950549f59da24d3881307437", "1536ff5f84364c2ca6b7a27a7dd4e1cb", "c7be327d0c38448b8342e0063790e444", "944afb8eb35e4863bad74365fea82195", "2d3c09c28978488e9d402da7d0071187", "12632cf349544d35b2d22b61917b8121", "a84314e66a5c479e8f93c4562bccb178", "c693a40e9fc244d3b7ddec29315ca5a2", "5b129ed5145843f59288f7179fe190b7", "85ae35280fce49999a5c68c59be36d65", "30cee9f7711c4182a3f59d63fe3bae54", "bb1e4db541ac469c9c37f4c6ca207c01", "2b63eab80160456ca08cad0a0078fa31", "08fa8fe8b0d04d5e9339336a0238e8d5", "131c18fbfd5a4313af0becbbddd1fc5e", "3d3baca8443e4c35a7a87bec8ba4acd6", "b54359590a3d4713bdf429e6ebf11a93", "15cde633bf5b4166b27474094c8fe35c", "eb21662a9ff44d98bee541b1df68b1e2", "1ee13eb683624c1995bba66696a9becb", "6ae8119c4e7d415cb5b2abece0d82d83", "c4dc61c9cc6849febd43cd1315c18557", "6cbe2836e67a46bcb45f86e0670fb7bb", "4a9899fee30d431cb93794280057fbe6", "725e5e41d7db471d9dfdbdbc1fabc83a", "eaf4d1c9625247499507c10781d1c5ee", "cbd7c4455e03435c8003718318502bcc", "feb69b774ced4adc9651f68e7aa129af", "83029054bf184fa5b36c7d4dbd2bfa12", "62043a4bc704433ca961090120cb519d", "3bc2940e57854dc28895dd197653ecd4", "99a63f17ce684284b3092fe8e5104987", "83a8c5bc47fb45c5be5b3f1a2b596fa5", "eadcda2b5fb249d29135c46000b72e46", "cb67729a4bc94bcc98fe158f6401d745", "47817377f61d45aa8f4884cffea2babd", "42ef1bd5ca4648e69d4431f38b67ace4"]} id="9a626d74" outputId="68cb70b5-2e43-4987-a3f2-8874034f2621"
def safe_mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved figure:", path)


def compute_audio_quality_features(audio_array, sr):
    audio = np.asarray(audio_array, dtype=np.float32).flatten()
    audio = np.nan_to_num(audio)

    if len(audio) == 0:
        return {
            "duration_sec": 0.0,
            "num_samples": 0,
            "mean_abs_energy": 0.0,
            "rms_energy": 0.0,
            "peak_amplitude": 0.0,
            "silence_ratio_001": 1.0,
            "clipping_ratio_099": 0.0,
            "zero_crossing_rate": 0.0,
            "dynamic_range": 0.0,
        }

    duration_sec = len(audio) / float(sr)
    mean_abs_energy = float(np.mean(np.abs(audio)))
    rms_energy = float(np.sqrt(np.mean(audio ** 2)))
    peak_amplitude = float(np.max(np.abs(audio)))
    silence_ratio = float(np.mean(np.abs(audio) < 0.01))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(audio)[0]))
    dynamic_range = float(np.percentile(audio, 95) - np.percentile(audio, 5))

    return {
        "duration_sec": duration_sec,
        "num_samples": int(len(audio)),
        "mean_abs_energy": mean_abs_energy,
        "rms_energy": rms_energy,
        "peak_amplitude": peak_amplitude,
        "silence_ratio_001": silence_ratio,
        "clipping_ratio_099": clipping_ratio,
        "zero_crossing_rate": zero_crossing_rate,
        "dynamic_range": dynamic_range,
    }


def collect_fleurs_eda_rows(language_cfg, split, max_samples=200, max_scan_rows=1000):
    """Collect aligned African-English FLEURS examples by row order for EDA only."""
    afr_stream = load_fleurs_split(language_cfg["fleurs_config"], split=split, streaming=True)
    eng_stream = load_fleurs_split(ENGLISH_CONFIG, split=split, streaming=True)

    rows = []
    for idx, (afr_item, eng_item) in enumerate(zip(afr_stream, eng_stream)):
        if idx >= max_scan_rows or len(rows) >= max_samples:
            break

        afr_text = normalize_text(get_text(afr_item))
        eng_text = normalize_text(get_text(eng_item))
        if not afr_text or not eng_text:
            continue

        afr_audio = afr_item["audio"]
        eng_audio = eng_item["audio"]
        afr_array = resample_audio(afr_audio["array"], afr_audio["sampling_rate"], TARGET_SR)
        eng_array = resample_audio(eng_audio["array"], eng_audio["sampling_rate"], TARGET_SR)

        afr_quality = compute_audio_quality_features(afr_array, TARGET_SR)
        eng_quality = compute_audio_quality_features(eng_array, TARGET_SR)

        rows.append({
            "sample_index": idx,
            "split": split,
            "language": language_cfg["display_name"],
            "language_key": language_cfg["language"],
            "fleurs_config": language_cfg["fleurs_config"],
            "source_lang_code": language_cfg["source_lang_code"],
            "african_text": afr_text,
            "english_text": eng_text,
            "african_words": len(afr_text.split()),
            "english_words": len(eng_text.split()),
            "word_length_difference": abs(len(afr_text.split()) - len(eng_text.split())),
            **{f"african_{k}": v for k, v in afr_quality.items()},
            **{f"english_{k}": v for k, v in eng_quality.items()},
        })
    return rows


def plot_language_eda(language_df, out_dir, language_name):
    if language_df.empty:
        print(f"No EDA rows available for {language_name}; skipping plots.")
        return

    # Duration distributions
    plt.figure(figsize=(8, 5))
    plt.hist(language_df["african_duration_sec"].dropna(), bins=20, alpha=0.65, label=language_name)
    plt.hist(language_df["english_duration_sec"].dropna(), bins=20, alpha=0.45, label="English")
    plt.xlabel("Audio duration in seconds")
    plt.ylabel("Number of samples")
    plt.title(f"{language_name} ↔ English: Audio Duration Distribution")
    plt.legend()
    save_plot(out_dir / "eda_audio_duration_distribution.png")

    # Text length distributions
    plt.figure(figsize=(8, 5))
    plt.hist(language_df["african_words"].dropna(), bins=20, alpha=0.65, label=language_name)
    plt.hist(language_df["english_words"].dropna(), bins=20, alpha=0.45, label="English")
    plt.xlabel("Number of words")
    plt.ylabel("Number of samples")
    plt.title(f"{language_name} ↔ English: Transcript Length Distribution")
    plt.legend()
    save_plot(out_dir / "eda_text_length_distribution.png")

    # Energy distribution
    plt.figure(figsize=(8, 5))
    plt.hist(language_df["african_rms_energy"].dropna(), bins=20, alpha=0.65, label=language_name)
    plt.hist(language_df["english_rms_energy"].dropna(), bins=20, alpha=0.45, label="English")
    plt.xlabel("RMS energy")
    plt.ylabel("Number of samples")
    plt.title(f"{language_name} ↔ English: RMS Energy Distribution")
    plt.legend()
    save_plot(out_dir / "eda_rms_energy_distribution.png")

    # Silence ratio distribution
    plt.figure(figsize=(8, 5))
    plt.hist(language_df["african_silence_ratio_001"].dropna(), bins=20, alpha=0.65, label=language_name)
    plt.hist(language_df["english_silence_ratio_001"].dropna(), bins=20, alpha=0.45, label="English")
    plt.xlabel("Silence ratio |amplitude| < 0.01")
    plt.ylabel("Number of samples")
    plt.title(f"{language_name} ↔ English: Silence Ratio Distribution")
    plt.legend()
    save_plot(out_dir / "eda_silence_ratio_distribution.png")

    # Duration vs quality scatter
    plt.figure(figsize=(8, 5))
    plt.scatter(language_df["african_duration_sec"], language_df["african_rms_energy"], alpha=0.65, label=language_name)
    plt.scatter(language_df["english_duration_sec"], language_df["english_rms_energy"], alpha=0.45, label="English")
    plt.xlabel("Audio duration in seconds")
    plt.ylabel("RMS energy")
    plt.title(f"{language_name} ↔ English: Duration vs RMS Energy")
    plt.legend()
    save_plot(out_dir / "eda_duration_vs_energy.png")

    # Duration buckets
    bucket_df = language_df.copy()
    bucket_df["african_duration_bucket"] = pd.cut(
        bucket_df["african_duration_sec"],
        bins=[0, 3, 6, 10, 15, 20, 60],
        labels=["0-3s", "3-6s", "6-10s", "10-15s", "15-20s", "20s+"],
        include_lowest=True,
    )
    bucket_counts = bucket_df["african_duration_bucket"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(bucket_counts.index.astype(str), bucket_counts.values)
    plt.xlabel("Duration bucket")
    plt.ylabel("Number of samples")
    plt.title(f"{language_name}: African-Language Audio Duration Buckets")
    save_plot(out_dir / "eda_duration_bucket_distribution.png")


def save_waveform_examples(language_cfg, split, out_dir, max_examples=2):
    if max_examples <= 0:
        return
    afr_stream = load_fleurs_split(language_cfg["fleurs_config"], split=split, streaming=True)
    for idx, item in enumerate(afr_stream):
        if idx >= max_examples:
            break
        audio_obj = item["audio"]
        audio_array = resample_audio(audio_obj["array"], audio_obj["sampling_rate"], TARGET_SR)
        duration = audio_duration(audio_array, TARGET_SR)
        times = np.arange(len(audio_array)) / TARGET_SR

        plt.figure(figsize=(10, 3))
        plt.plot(times, audio_array)
        plt.xlabel("Time in seconds")
        plt.ylabel("Amplitude")
        plt.title(f"{language_cfg['display_name']} waveform example {idx} ({duration:.2f}s)")
        save_plot(out_dir / f"waveform_example_{idx}.png")


def run_data_exploration():
    eda_root = safe_mkdir(BASE_OUTPUT_DIR / "data_exploration")
    all_rows = []
    metadata_rows = []

    print("Running FLEURS dataset exploration and audio-quality analysis...")
    for cfg in LANGUAGE_CONFIGS:
        language_out_dir = safe_mkdir(eda_root / cfg["language"])
        print("=" * 100)
        print(f"Exploring {cfg['display_name']} ↔ English")
        print("=" * 100)

        # Store dataset properties for paper/methodology reporting.
        try:
            preview_dataset = load_dataset(DATASET_ID, cfg["fleurs_config"], split=SPLIT_FOR_EXPERIMENT, streaming=False)
            dataset_columns = list(preview_dataset.column_names)
            preview_count = len(preview_dataset)
        except Exception as e:
            dataset_columns = []
            preview_count = None
            print(f"Could not load non-streaming preview for {cfg['display_name']}: {e}")

        metadata_rows.append({
            "language": cfg["display_name"],
            "language_key": cfg["language"],
            "fleurs_config": cfg["fleurs_config"],
            "source_lang_code": cfg["source_lang_code"],
            "dataset_id": DATASET_ID,
            "english_config": ENGLISH_CONFIG,
            "analyzed_split": SPLIT_FOR_EXPERIMENT,
            "available_columns": ", ".join(dataset_columns),
            "rows_in_analyzed_split_if_loaded_nonstreaming": preview_count,
            "eda_sample_size_per_split": EDA_SAMPLE_SIZE_PER_LANGUAGE_SPLIT,
            "eda_max_scan_rows": EDA_MAX_SCAN_ROWS,
        })

        language_rows = []
        for split in EDA_SPLITS_TO_ANALYZE:
            try:
                split_rows = collect_fleurs_eda_rows(
                    cfg,
                    split=split,
                    max_samples=EDA_SAMPLE_SIZE_PER_LANGUAGE_SPLIT,
                    max_scan_rows=EDA_MAX_SCAN_ROWS,
                )
                language_rows.extend(split_rows)
                print(f"{cfg['display_name']} {split}: collected {len(split_rows)} rows")
            except Exception as e:
                print(f"EDA collection failed for {cfg['display_name']} split={split}: {e}")

        language_df = pd.DataFrame(language_rows)
        if not language_df.empty:
            language_df.to_csv(language_out_dir / "eda_aligned_samples_with_audio_quality.csv", index=False)
            summary_cols = [
                "african_words", "english_words", "word_length_difference",
                "african_duration_sec", "english_duration_sec",
                "african_rms_energy", "english_rms_energy",
                "african_peak_amplitude", "english_peak_amplitude",
                "african_silence_ratio_001", "english_silence_ratio_001",
                "african_clipping_ratio_099", "english_clipping_ratio_099",
                "african_zero_crossing_rate", "english_zero_crossing_rate",
                "african_dynamic_range", "english_dynamic_range",
            ]
            language_summary = language_df[summary_cols].describe().T.reset_index().rename(columns={"index": "metric"})
            language_summary.insert(0, "language", cfg["display_name"])
            language_summary.to_csv(language_out_dir / "eda_statistical_summary.csv", index=False)

            plot_language_eda(language_df, language_out_dir, cfg["display_name"])
            save_waveform_examples(cfg, SPLIT_FOR_EXPERIMENT, language_out_dir, max_examples=EDA_AUDIO_EXAMPLES_PER_LANGUAGE)
            all_rows.extend(language_df.to_dict("records"))
        else:
            print(f"No EDA rows collected for {cfg['display_name']}.")

    eda_samples_df = pd.DataFrame(all_rows)
    eda_metadata_df = pd.DataFrame(metadata_rows)

    if not eda_samples_df.empty:
        eda_samples_df.to_csv(BASE_OUTPUT_DIR / "01a_fleurs_eda_samples_audio_quality.csv", index=False)

        numeric_cols = eda_samples_df.select_dtypes(include=[np.number]).columns.tolist()
        grouped_summary = (
            eda_samples_df
            .groupby(["language", "split"])[numeric_cols]
            .agg(["count", "mean", "std", "min", "median", "max"])
        )
        grouped_summary.columns = ["_".join(col).strip() for col in grouped_summary.columns.values]
        grouped_summary = grouped_summary.reset_index()
        grouped_summary.to_csv(BASE_OUTPUT_DIR / "01b_fleurs_eda_grouped_summary.csv", index=False)

        # Paper-ready compact table.
        compact_summary = eda_samples_df.groupby("language").agg(
            samples=("sample_index", "count"),
            avg_african_duration_sec=("african_duration_sec", "mean"),
            avg_english_duration_sec=("english_duration_sec", "mean"),
            avg_african_words=("african_words", "mean"),
            avg_english_words=("english_words", "mean"),
            avg_african_rms_energy=("african_rms_energy", "mean"),
            avg_african_silence_ratio=("african_silence_ratio_001", "mean"),
            avg_african_clipping_ratio=("african_clipping_ratio_099", "mean"),
            avg_african_zero_crossing_rate=("african_zero_crossing_rate", "mean"),
        ).reset_index()
        compact_summary.to_csv(BASE_OUTPUT_DIR / "01c_fleurs_paper_ready_data_quality_summary.csv", index=False)

        # Cross-language overview plots.
        plt.figure(figsize=(10, 5))
        compact_summary_sorted = compact_summary.sort_values("avg_african_duration_sec")
        plt.bar(compact_summary_sorted["language"], compact_summary_sorted["avg_african_duration_sec"])
        plt.ylabel("Average duration in seconds")
        plt.title("Average African-Language Audio Duration by Language")
        save_plot(BASE_OUTPUT_DIR / "01d_average_duration_by_language.png")

        plt.figure(figsize=(10, 5))
        plt.bar(compact_summary["language"], compact_summary["avg_african_silence_ratio"])
        plt.ylabel("Average silence ratio")
        plt.title("Average Silence Ratio by Language")
        save_plot(BASE_OUTPUT_DIR / "01e_average_silence_ratio_by_language.png")

        plt.figure(figsize=(10, 5))
        plt.bar(compact_summary["language"], compact_summary["avg_african_rms_energy"])
        plt.ylabel("Average RMS energy")
        plt.title("Average RMS Energy by Language")
        save_plot(BASE_OUTPUT_DIR / "01f_average_rms_energy_by_language.png")
    else:
        grouped_summary = pd.DataFrame()
        compact_summary = pd.DataFrame()
        print("No EDA sample rows were collected.")

    eda_metadata_df.to_csv(BASE_OUTPUT_DIR / "01g_fleurs_dataset_properties.csv", index=False)

    print("EDA outputs saved under:", BASE_OUTPUT_DIR)
    return eda_metadata_df, eda_samples_df, grouped_summary, compact_summary


if RUN_DATA_EXPLORATION:
    eda_metadata_df, eda_audio_quality_df, eda_grouped_summary_df, eda_compact_summary_df = run_data_exploration()
    print(eda_compact_summary_df)
else:
    print("Data exploration skipped because RUN_DATA_EXPLORATION=False.")
    eda_metadata_df = pd.DataFrame()
    eda_audio_quality_df = pd.DataFrame()
    eda_grouped_summary_df = pd.DataFrame()
    eda_compact_summary_df = pd.DataFrame()


# %% [markdown] id="62cf6e3c"
# # 4 Modeling and Evaluation
#
# This section defines reusable functions for model inference and scoring. The goal is to ensure that all methods are evaluated with the same metrics and output format.
#
# Expected outputs: BLEU/ChrF-style metric records and prediction tables for each method.
#
# How to interpret: compare methods within the same language and experiment first, then compare across languages. Cross-language comparisons should consider dataset difficulty, audio duration, and audio quality from the EDA section.

# %% [markdown] id="f67c7e37"
# ## 4.1 Translation Helpers
#
# These functions call the model for text translation and audio translation. For a different model baseline, this is one of the main sections to modify.
#
# Expected behavior: the functions should return plain text predictions and avoid changing the output schema used by later cells.
#
# Paper note: mention decoding settings such as beam size, max tokens, and whether speech generation is disabled, because these choices affect comparability.

# %% id="2181f4c9"
def translate_text_direct(text, source_lang_code, target_lang_code):
    """Translate one text input with SeamlessM4T-v2.

    Used for transcript-to-English and English-to-African back-translation.
    This isolates the text translation capability from the speech/audio component.
    """
    text = str(text).strip()
    if not text:
        return ""

    inputs = processor(
        text=text,
        src_lang=source_lang_code,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            tgt_lang=target_lang_code,
            generate_speech=False,
        )

    return processor.decode(
        generated_tokens[0].tolist()[0],
        skip_special_tokens=True,
    )


def translate_audio_direct(audio_object_or_array, source_lang_code, target_lang_code, normalize_audio=True, improved=True):
    """Translate one audio input with SeamlessM4T-v2.

    This is the core speech-to-text translation call. Optional trimming and
    normalization are controlled by the optimization strategy wrapper.
    """
    if isinstance(audio_object_or_array, dict):
        sr = audio_object_or_array.get("sampling_rate", TARGET_SR)
        audio = audio_object_or_array.get("array")
        audio = resample_audio(audio, sr, TARGET_SR)
    else:
        audio = np.asarray(audio_object_or_array, dtype=np.float32).flatten()

    if improved:
        audio = trim_silence(audio)

    if normalize_audio:
        audio = normalize_audio_waveform(audio)

    audio = ensure_min_audio_length(audio, MIN_DURATION, TARGET_SR)

    inputs = processor(
        audio=audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            tgt_lang=target_lang_code,
            generate_speech=False,
        )

    return processor.decode(
        generated_tokens[0].tolist()[0],
        skip_special_tokens=True,
    )



# %% [markdown] id="0da0be99"
# ## 4.2 Audio Optimization Strategy Helpers
#
# This section defines switchable audio preprocessing strategies. These strategies make it possible to test whether model performance changes when the input signal is normalized, trimmed, segmented, or filtered by duration.
#
# Expected strategies:
#
# - Direct audio: raw baseline.
# - Normalized audio: amplitude-normalized input.
# - Trimmed audio: leading/trailing low-energy segments removed.
# - Chunk-based audio: long audio split into shorter segments.
# - Short/long audio analysis: duration-based error analysis.
# - Gold-transcript cascade: upper-bound translation using the clean transcript.
#
# How to interpret: a strategy that improves scores suggests that the model is sensitive to that input property. A strategy that hurts scores may be removing useful information or introducing segmentation artifacts.

# %% id="68206dea"
# ============================================================
# Audio optimization helpers
# ============================================================

def split_audio_into_chunks(audio, chunk_seconds=CHUNK_SECONDS, overlap_seconds=CHUNK_OVERLAP_SECONDS, target_sr=TARGET_SR):
    """Split a waveform into overlapping chunks for long-audio robustness experiments."""
    audio = np.asarray(audio, dtype=np.float32).flatten()
    chunk_size = int(chunk_seconds * target_sr)
    overlap_size = int(overlap_seconds * target_sr)
    step_size = max(1, chunk_size - overlap_size)

    chunks = []
    for start in range(0, len(audio), step_size):
        end = start + chunk_size
        chunk = audio[start:end]
        if len(chunk) < int(MIN_DURATION * target_sr):
            continue
        chunks.append(chunk.astype(np.float32))
        if end >= len(audio):
            break
    return chunks


def prepare_audio_for_strategy(audio_object_or_array, strategy):
    """Apply the selected optimization strategy while keeping the evaluation schema stable."""
    if isinstance(audio_object_or_array, dict):
        sr = audio_object_or_array.get("sampling_rate", TARGET_SR)
        audio = audio_object_or_array.get("array")
        audio = resample_audio(audio, sr, TARGET_SR)
    else:
        audio = np.asarray(audio_object_or_array, dtype=np.float32).flatten()

    if strategy.get("trim", False):
        audio = trim_silence(audio)

    if strategy.get("normalize", False):
        audio = normalize_audio_waveform(audio)

    audio = ensure_min_audio_length(audio, MIN_DURATION, TARGET_SR)
    return audio.astype(np.float32)


def translate_audio_with_strategy(audio_object_or_array, source_lang_code, target_lang_code, strategy):
    """Translate one audio sample using a named optimization strategy."""
    audio = prepare_audio_for_strategy(audio_object_or_array, strategy)

    if strategy.get("chunk", False):
        chunks = split_audio_into_chunks(audio)
        if not chunks:
            chunks = [audio]

        chunk_predictions = []
        for chunk in chunks:
            pred = translate_audio_direct(
                chunk,
                source_lang_code,
                target_lang_code,
                normalize_audio=False,
                improved=False,
            )
            pred = normalize_text(pred)
            if pred:
                chunk_predictions.append(pred)

        return " ".join(chunk_predictions), len(chunks)

    pred = translate_audio_direct(
        audio,
        source_lang_code,
        target_lang_code,
        normalize_audio=False,
        improved=False,
    )
    return pred, 1


def active_audio_strategies_for_experiment(experiment_name):
    """Return enabled strategies for the current experiment."""
    active = []
    for strategy in AUDIO_OPTIMIZATION_STRATEGIES:
        if not strategy.get("enabled", False):
            continue
        if (
            RUN_EXPENSIVE_AUDIO_STRATEGIES_ONLY_IN_EXP1
            and strategy.get("expensive", False)
            and experiment_name not in ["Experiment_1", "debug"]
        ):
            continue
        active.append(strategy)
    return active



# %% [markdown] id="mv_ba-K0Kf_j"
# ## 4.2.1 Scientific Method Documentation for the Team
#
# This notebook uses the same evaluation logic that should be reused for future model/dataset baselines. Keep the method names and output columns stable so results remain comparable across papers and tables.
#
# | Method / Strategy | What it tests | Expected output | How to interpret it |
# |---|---|---|---|
# | `Transcript → English` | Text translation using the gold source transcript. | BLEU/ChrF and prediction CSV. | High scores mean the text translation component works; low scores suggest translation limitations independent of audio. |
# | `English → Source` | Reverse-direction generation from English to the African language. | BLEU/ChrF and prediction CSV. | Useful for bidirectional analysis; often harder for low-resource languages. |
# | `Direct audio → English` | End-to-end speech-to-text translation from raw audio. | BLEU/ChrF and audio prediction CSV. | This is the core speech translation baseline. If it is much lower than transcript translation, audio understanding is likely the bottleneck. |
# | `Normalized audio → English` | Audio is mean-centered and amplitude-normalized before inference. | BLEU/ChrF under normalized mode. | Improvement suggests the model is sensitive to volume/amplitude differences. Degradation means normalization may distort useful signal. |
# | `Trimmed audio → English` | Leading/trailing silence is removed before inference. | BLEU/ChrF under trimmed mode. | Improvement suggests edge silence hurts inference. Degradation suggests trimming removed useful speech or silence is not the main issue. |
# | `Chunk-based audio → English` | Longer utterances are split into overlapping segments. | BLEU/ChrF and number of chunks. | Improvement suggests long-audio handling is a problem. Degradation may indicate chunk boundary errors or repeated/hallucinated text. |
# | `EDA/audio quality analysis` | Dataset and signal properties: duration, energy, silence ratio, transcript length. | CSV statistics and plots. | Use this to explain model behavior and justify preprocessing choices in the paper. |
# | `Qualitative outputs` | Manual inspection table with prediction/reference/source. | CSV files with manual annotation columns. | Use this for error categories such as wrong language, omission, hallucination, mistranslation, or truncation. |
#
# For paper writing, compare methods within the same language and experiment size first, then compare across languages and across notebooks. The key scientific interpretation is whether gains come from the model, the dataset, or the audio optimization strategy.
#

# %% [markdown] id="4a0d812a"
# ## 4.3 Metrics
#
# This section computes automatic evaluation metrics in a consistent format. BLEU is useful for lexical overlap, while character-level metrics are useful for morphologically rich or orthographically variable languages.
#
# Expected output: metric objects or numeric metric values stored in CSV files.
#
# Interpretation warning: automatic metrics are imperfect for low-resource African languages. Use them together with qualitative examples, not as the only evidence.

# %% id="cf729714"
def compute_metrics(predictions, references):
    """Compute BLEU and ChrF for a list of predictions and references.

    BLEU is useful for word/phrase overlap, while ChrF is often more tolerant
    for morphologically rich and orthographically diverse languages.
    """
    clean_preds = [str(p).strip() for p in predictions]
    clean_refs = [str(r).strip() for r in references]

    if not clean_preds or not clean_refs:
        return {"BLEU": 0.0, "ChrF": 0.0}

    bleu = sacrebleu.corpus_bleu(clean_preds, [clean_refs]).score
    chrf = sacrebleu.corpus_chrf(clean_preds, [clean_refs]).score

    return {
        "BLEU": float(bleu),
        "ChrF": float(chrf),
    }



# %% [markdown] id="926825eb"
# ### 4.4 Evaluating Source Transcript → English and English → Source
#
# This section evaluates text-only translation in both directions.
#
# Expected output:
#
# - African transcript → English predictions.
# - English → African-language predictions.
# - Per-example prediction CSV files.
#
# How to interpret: this evaluation estimates the model's translation ability without acoustic noise. If this score is low, the model likely struggles with the language even before speech is introduced. If this score is high but speech translation is low, the bottleneck is likely audio understanding.

# %% id="a4c96836" colab={"base_uri": "https://localhost:8080/"} outputId="9d12cc96-e5e0-4ff7-a0fd-fbb8fc22189e"
text_results = []
text_predictions_rows = []

if RUN_TEXT_EVALUATION:
    for exp_cfg in EXPERIMENT_CONFIGS:
        experiment_name = exp_cfg["experiment"]
        print("\nRunning text evaluation:", experiment_name)

        for cfg in LANGUAGE_CONFIGS:
            rows = prepare_bidirectional_pairs(
                cfg["language"],
                split=SPLIT_FOR_EXPERIMENT,
                max_samples=exp_cfg["max_text_dev"],
                max_scan_rows=exp_cfg["max_scan_rows"],
            )

            direction_groups = {}
            for row in rows:
                direction_groups.setdefault(row["direction"], []).append(row)

            for direction, group_rows in direction_groups.items():
                preds, refs = [], []
                start = time.time()

                for row in group_rows:
                    try:
                        pred = translate_text_direct(
                            row["source_text"],
                            row["source_lang_code"],
                            row["target_lang_code"],
                        )
                    except Exception as e:
                        pred = ""
                        print("Text translation error:", experiment_name, direction, row["sample_index"], e)

                    preds.append(pred)
                    refs.append(row["target_text"])

                    text_predictions_rows.append({
                        "baseline_model": BASELINE_MODEL_NAME,
                        "dataset": BASELINE_DATASET_NAME,
                        "experiment_family": EXPERIMENT_FAMILY,
                        "experiment": experiment_name,
                        "sample_index": row["sample_index"],
                        "language": row["language_display"],
                        "direction": direction,
                        "direction_key": row["direction_key"],
                        "mode": "text_to_text",
                        "method": "Transcript ↔ Text Translation",
                        "category": "Text Translation",
                        "source_text": row["source_text"],
                        "reference": row["target_text"],
                        "prediction": pred,
                    })

                metrics = compute_metrics(preds, refs)
                elapsed = time.time() - start

                text_results.append({
                    "baseline_model": BASELINE_MODEL_NAME,
                    "dataset": BASELINE_DATASET_NAME,
                    "experiment_family": EXPERIMENT_FAMILY,
                    "experiment": experiment_name,
                    "mode": "text_to_text",
                    "method": "Transcript ↔ Text Translation",
                    "category": "Text Translation",
                    "language": cfg["display_name"],
                    "direction": direction,
                    "num_samples": len(group_rows),
                    "BLEU": metrics["BLEU"],
                    "ChrF": metrics["ChrF"],
                    "runtime_seconds": elapsed,
                })
else:
    print("Text evaluation skipped.")

text_results_df = pd.DataFrame(text_results)
text_predictions_df = pd.DataFrame(text_predictions_rows)

save_dataframe(text_results_df, "02_text_bidirectional_metrics.csv")
save_dataframe(text_predictions_df, "03_text_bidirectional_predictions.csv")

print(text_results_df)


# %% [markdown] id="a1dbb46a"
# # 5 Audio Data Preparation for Official Speech-to-English Evaluation
#
# This section prepares audio examples for the official speech-to-English evaluation direction. It applies duration filtering and stores the aligned audio, source transcript, target English reference, and duration metadata.
#
# Expected output: `audio_dev_pairs.csv` and a prepared audio dataset for each language and experiment.
#
# How to interpret: the number of retained audio examples matters. If too many examples are filtered out, report the filtering criteria and consider relaxing duration limits.

# %% [markdown] id="ac6fcf8b"
# # 6 Audio Data Exploration
#
# This section analyzes audio properties before translation. It produces descriptive statistics and visualizations that help explain model behavior.
#
# Expected outputs:
#
# - Audio duration histogram and boxplot.
# - Duration bucket distribution.
# - Audio energy distribution.
# - Silence ratio distribution.
# - Waveform summaries where enabled.
# - Audio quality CSV tables.
#
# How to interpret: low energy can indicate quiet recordings; high silence ratio can indicate long pauses; long duration can increase hallucination or truncation risk. These findings should be discussed when explaining why some languages or methods perform worse.

# %% [markdown] id="33948d68"
# # 7 Modeling and Evaluation: Source Audio → English
#
# This section evaluates the official speech translation task: African-language audio → English text.
#
# Expected outputs:
#
# - Direct audio prediction CSV.
# - Normalized audio prediction CSV.
# - Trimmed audio prediction CSV if enabled.
# - Chunk-based prediction CSV if enabled.
# - Method-level metric rows.
#
# How to interpret: this is the most important evaluation for speech-to-text translation. Compare it against transcript translation and gold cascade results to identify whether the model's weakness is audio understanding or translation.

# %% id="4790370f"
audio_results = []
audio_predictions_rows = []

should_run_audio = RUN_AUDIO_EVALUATION and not (DEBUG_MODE and SKIP_AUDIO_DEBUG)

if should_run_audio:
    for exp_cfg in EXPERIMENT_CONFIGS:
        experiment_name = exp_cfg["experiment"]
        active_strategies = active_audio_strategies_for_experiment(experiment_name)
        print("\nRunning audio evaluation:", experiment_name)
        print("Active audio strategies:", [s["strategy_key"] for s in active_strategies])

        for cfg in LANGUAGE_CONFIGS:
            rows = prepare_bidirectional_pairs(
                cfg["language"],
                split=SPLIT_FOR_EXPERIMENT,
                max_samples=exp_cfg["max_audio_dev"],
                max_scan_rows=exp_cfg["max_scan_rows"],
            )

            direction_groups = {}
            for row in rows:
                direction_groups.setdefault(row["direction"], []).append(row)

            for strategy in active_strategies:
                strategy_key = strategy["strategy_key"]
                method = strategy["method"]
                category = strategy["category"]

                for direction, group_rows in direction_groups.items():
                    preds, refs = [], []
                    durations = []
                    chunk_counts = []
                    start = time.time()

                    for row in group_rows:
                        source_audio = row["source_audio"]
                        audio_array = resample_audio(source_audio["array"], source_audio["sampling_rate"], TARGET_SR)
                        duration = audio_duration(audio_array, TARGET_SR)
                        durations.append(duration)

                        if duration < MIN_DURATION or duration > MAX_DURATION:
                            # Keep row visible, but do not waste runtime on unsuitable clips.
                            pred = ""
                            num_chunks = 0
                            error_message = f"skipped_duration_{duration:.2f}s"
                        else:
                            error_message = ""
                            try:
                                pred, num_chunks = translate_audio_with_strategy(
                                    source_audio,
                                    row["source_lang_code"],
                                    row["target_lang_code"],
                                    strategy,
                                )
                            except Exception as e:
                                pred = ""
                                num_chunks = 0
                                error_message = str(e)
                                print("Audio translation error:", experiment_name, strategy_key, direction, row["sample_index"], e)

                        chunk_counts.append(num_chunks)
                        preds.append(pred)
                        refs.append(row["target_text"])

                        audio_predictions_rows.append({
                            "baseline_model": BASELINE_MODEL_NAME,
                            "dataset": BASELINE_DATASET_NAME,
                            "experiment_family": EXPERIMENT_FAMILY,
                            "experiment": experiment_name,
                            "sample_index": row["sample_index"],
                            "language": row["language_display"],
                            "direction": direction,
                            "direction_key": row["direction_key"],
                            "mode": strategy_key,
                            "method": method,
                            "category": category,
                            "source_text_transcription": row["source_text"],
                            "reference": row["target_text"],
                            "prediction": pred,
                            "duration_seconds": duration,
                            "num_chunks": num_chunks,
                            "error_message": error_message,
                        })

                    metrics = compute_metrics(preds, refs)
                    elapsed = time.time() - start

                    audio_results.append({
                        "baseline_model": BASELINE_MODEL_NAME,
                        "dataset": BASELINE_DATASET_NAME,
                        "experiment_family": EXPERIMENT_FAMILY,
                        "experiment": experiment_name,
                        "mode": strategy_key,
                        "method": method,
                        "category": category,
                        "language": cfg["display_name"],
                        "direction": direction,
                        "num_samples": len(group_rows),
                        "BLEU": metrics["BLEU"],
                        "ChrF": metrics["ChrF"],
                        "avg_duration_seconds": float(np.mean(durations)) if durations else 0.0,
                        "avg_num_chunks": float(np.mean(chunk_counts)) if chunk_counts else 0.0,
                        "runtime_seconds": elapsed,
                    })
else:
    print("Audio evaluation skipped. RUN_AUDIO_EVALUATION=", RUN_AUDIO_EVALUATION, "DEBUG_MODE=", DEBUG_MODE, "SKIP_AUDIO_DEBUG=", SKIP_AUDIO_DEBUG)

audio_results_df = pd.DataFrame(audio_results)
audio_predictions_df = pd.DataFrame(audio_predictions_rows)

save_dataframe(audio_results_df, "04_audio_bidirectional_metrics.csv")
save_dataframe(audio_predictions_df, "05_audio_bidirectional_predictions.csv")

print(audio_results_df)


# %% [markdown] id="643d07cf"
# # 8 Comparative Analysis
#
# This section compares methods within each language and experiment. It creates tables and plots that can be reused in the paper.
#
# Expected outputs:
#
# - All-method comparison CSV files.
# - Bar charts comparing BLEU/ChrF across methods.
# - Aggregate result tables across languages and experiments.
#
# How to interpret: the best method should be reported together with the method category. For example, if gold cascade greatly outperforms direct audio, the model likely benefits from better ASR or cleaner transcripts.

# %% id="32c880be"
frames = []
if not text_results_df.empty:
    frames.append(text_results_df)
if not audio_results_df.empty:
    frames.append(audio_results_df)

aggregate_results_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
save_dataframe(aggregate_results_df, "06_aggregate_bidirectional_metrics.csv")
print(aggregate_results_df)


# %% [markdown] id="19331ba4"
# ## 8.1 Visualization of Comparable Metrics
#
# This section creates paper-ready metric visualizations. The goal is not only to show which method wins, but also to show how optimization strategies change performance.
#
# Expected plots:
#
# - Method comparison per language.
# - Cross-language score comparison.
# - Best-method summary.
#
# How to interpret: large differences between languages may reflect language coverage, dataset quality, orthography, audio quality, or model pretraining coverage.

# %% id="2056fcec"
if not aggregate_results_df.empty:
    plt.figure(figsize=(12, 5))
    labels = (
        aggregate_results_df["experiment"].astype(str)
        + " | " + aggregate_results_df["mode"].astype(str)
        + " | " + aggregate_results_df["direction"].astype(str)
    )
    plt.bar(range(len(labels)), aggregate_results_df["ChrF"])
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("ChrF")
    plt.title("Bidirectional Translation Quality by Experiment, Mode, and Direction")
    plt.tight_layout()
    fig_path = BASE_OUTPUT_DIR / "07_chrf_bidirectional_bar_chart.png"
    plt.savefig(fig_path, dpi=200)
    plt.show()
    print("Saved:", fig_path)
else:
    print("No aggregate results available for visualization.")


# %% [markdown] id="9438680c"
# # 9 Error Analysis
#
# This section prepares qualitative output tables for manual inspection. The tables include source text, reference translation, prediction, duration, and blank annotation columns.
#
# Expected manual error categories include:
#
# - Hallucination.
# - Omission.
# - Wrong language output.
# - Repetition.
# - Named-entity error.
# - Semantic drift.
# - Truncation.
#
# How to interpret: use these examples to support the discussion section of the paper. Qualitative examples are especially important when automatic metrics disagree or when a model produces fluent but incorrect outputs.

# %% [markdown] id="40656153"
# Qualitative output tables are created with blank manual columns so you can annotate hallucination, omission, wrong-language output, mistranslation, and truncation.

# %% [markdown] id="0d93eb6d"
# # 10 Monitoring
#
# This section records runtime-level and experiment-level monitoring information. Monitoring helps explain missing results, failed samples, and performance bottlenecks.
#
# Expected outputs include sample counts, average duration, average energy, average silence ratio, BLEU/ChrF per method, and output directory paths.
#
# How to interpret: monitoring tables help ensure that differences between methods are not caused by different sample counts or skipped audio evaluations.

# %% [markdown] id="d4c4dab6"
# Runtime, sample counts, average duration, BLEU, ChrF, and strategy-level outputs are saved in aggregate CSV files.

# %% [markdown] id="42b66b54"
# # 11 Save Results
#
# This section saves all per-language, per-experiment, and aggregate outputs. It also backs up the complete result directory to Google Drive.
#
# Expected outputs:
#
# - Per-experiment result folders.
# - Aggregated CSV files.
# - Qualitative sample CSV files.
# - Monitoring summaries.
# - Audio-quality and duration-analysis tables.
# - Google Drive backup and optional ZIP archive.
#
# Paper note: use the aggregated CSV files for tables and figures, and use qualitative CSV files for examples in the error analysis section.

# %% [markdown] id="78e08f0a"
# Final local and Google Drive exports are handled in the aggregate results section.

# %% [markdown] id="26c5c8ab"
# # 12 Advanced Audio Translation Improvement Experiments
#
# This section contains the audio optimization and ablation strategies. These strategies are central to the research contribution because they test how preprocessing affects speech translation quality.
#
# Expected methods include direct audio, normalized audio, trimmed audio, chunk-based audio, duration-based analysis, and gold-transcript cascade.
#
# How to interpret: these experiments help answer whether improvements can be achieved by preprocessing alone or whether the main limitation is the model architecture or language coverage.

# %% [markdown] id="e4ddfa96"
# ## 12.1 Analysis 1 — Audio Duration and Segmentation Analysis
#
# This analysis compares model behavior on shorter and longer audio examples. Long recordings may cause hallucination, truncation, or attention degradation.
#
# Expected output: short-audio and long-audio result tables and scores.
#
# How to interpret: if short audio performs better, duration is likely a key source of error. If long audio performs similarly, the model may be robust to duration but still affected by other factors such as noise or language coverage.

# %% [markdown] id="55333125"
# ## 12.2 Acoustic Signal and Audio Quality Analysis
#
# This analysis computes acoustic properties such as energy and silence ratio. These values help explain why some examples are difficult for the model.
#
# Expected output: audio-quality CSV files and histograms.
#
# How to interpret: high silence ratio may indicate long pauses or poor segmentation; low energy may indicate quiet recordings. These properties can be correlated with translation quality in the discussion.

# %% [markdown] id="147f85b0"
# ## 12.3 Waveform Comparison: Original vs Normalized Audio
#
# This analysis compares raw audio with amplitude-normalized audio. Normalization may help when recordings vary greatly in loudness.
#
# Expected output: normalized-audio evaluation results and, where enabled, waveform comparison plots.
#
# How to interpret: improvement after normalization suggests amplitude variability affects model inference. No improvement suggests the model is already robust to loudness or that errors come from other causes.

# %% [markdown] id="3edfb417"
# ## 12.4 Silence Trimming Before Translation
#
# This analysis removes low-energy leading and trailing audio segments before translation. The trimming function is protected so that audio is not trimmed into an unusably short clip.
#
# Expected output: trimmed-audio predictions and scores.
#
# How to interpret: improvement suggests edge silence harms inference. Degradation suggests trimming may remove useful speech cues or that silence is not the main problem.

# %% [markdown] id="5ec10d0d"
# ## 12.5 Chunk-Based Audio Segmentation
#
# This analysis splits longer recordings into shorter overlapping chunks and translates them separately.
#
# Expected output: chunk-based predictions, number of chunks, and scores.
#
# How to interpret: improvement suggests the model struggles with long continuous audio. Degradation suggests chunking introduces context loss, repeated translations, or boundary errors.

# %% [markdown] id="02d37ebf"
# ## 12.6 Cascade Translation Strategy
#
# This analysis uses the gold source transcript as if it came from a perfect ASR system, then translates it into English. It estimates the upper-bound performance of a cascaded ASR + MT system.
#
# Expected output: gold-cascade prediction table and score.
#
# How to interpret: if gold cascade strongly outperforms direct audio, the main bottleneck is speech recognition or speech representation. If gold cascade is also weak, the translation component is likely weak for that language.

# %% [markdown] id="9e119fbf"
# ## 12.7 Qualitative Evaluation of the Cascade/System Outputs
#
# This section creates qualitative examples for manual analysis. It includes length analysis so the team can identify over-generation, under-generation, and truncation.
#
# Expected output: qualitative CSV files with example predictions and annotation columns.
#
# How to interpret: select representative examples for the paper, especially cases where automatic metrics are low but outputs are partially meaningful, or cases where outputs are fluent but wrong.

# %% id="f7f1eb56"
def build_qualitative_table(predictions_df, mode_name, examples_per_direction=5):
    if predictions_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["experiment", "direction"] if "experiment" in predictions_df.columns else ["direction"]
    for keys, group in predictions_df.groupby(group_cols):
        subset = group.head(examples_per_direction)
        for _, row in subset.iterrows():
            rows.append({
                "experiment": row.get("experiment", ""),
                "mode": row.get("mode", mode_name),
                "method": row.get("method", ""),
                "category": row.get("category", ""),
                "baseline_model": row.get("baseline_model", BASELINE_MODEL_NAME if "BASELINE_MODEL_NAME" in globals() else ""),
                "dataset": row.get("dataset", BASELINE_DATASET_NAME if "BASELINE_DATASET_NAME" in globals() else ""),
                "language": row.get("language", ""),
                "direction": row.get("direction", ""),
                "sample_index": row.get("sample_index", ""),
                "source": row.get("source_text", row.get("source_text_transcription", "")),
                "reference": row.get("reference", ""),
                "prediction": row.get("prediction", ""),
                "automatic_note": row.get("error_message", ""),
                "manual_error_category": "",  # Fill manually: correct / mistranslation / wrong language / omission / hallucination / ASR issue
                "manual_severity": "",        # Fill manually: low / medium / high
                "manual_comment": "",         # Fill manually during qualitative analysis
            })
    return pd.DataFrame(rows)

text_qualitative_df = build_qualitative_table(text_predictions_df, "text_to_text", examples_per_direction=5)
audio_qualitative_df = build_qualitative_table(audio_predictions_df, "speech_to_text_translation", examples_per_direction=5)

qualitative_frames = []
if not text_qualitative_df.empty:
    qualitative_frames.append(text_qualitative_df)
if not audio_qualitative_df.empty:
    qualitative_frames.append(audio_qualitative_df)

qualitative_all_df = pd.concat(qualitative_frames, ignore_index=True) if qualitative_frames else pd.DataFrame()

save_dataframe(text_qualitative_df, "08_text_qualitative_outputs_to_fill.csv")
save_dataframe(audio_qualitative_df, "09_audio_qualitative_outputs_to_fill.csv")
save_dataframe(qualitative_all_df, "10_all_qualitative_outputs_to_fill.csv")

print(qualitative_all_df)


# %% [markdown] id="3fa69cf1"
# # Multi-Language Experiment Runner
#
# This section executes the full experiment grid across all enabled languages and sample-size configurations. It is the main execution block of the notebook.
#
# Expected output: one result folder per language and experiment, containing prediction CSVs, audio-quality tables, monitoring summaries, and plots.
#
# Before running this section in full mode, verify that the sample check and one debug experiment work. For fair comparison, use the same language list and sample sizes across all model/dataset notebooks.

# %% [markdown] id="412df6ea"
# # Aggregate Cross-Language Results
#
# This section combines results from all languages, methods, and experiments into paper-ready aggregate tables.
#
# Expected outputs:
#
# - `all_language_all_experiment_bleu_results.csv`
# - `all_language_monitoring_summary.csv`
# - `all_language_audio_quality.csv`
# - `all_language_duration_analysis.csv`
# - qualitative sample CSV files
#
# How to interpret: use aggregate tables to compare model/dataset pairs across notebooks. Keep the same column schema for all baselines.

# %% [markdown] id="5b074f11"
# ## Aggregated Result Table for Paper Writing
#
# This table is the primary result table for paper writing. It should contain method, category, language, experiment, sample size, metric values, and output paths.
#
# How to interpret: report results by method category rather than only by raw score. For example, compare direct speech translation, normalized speech translation, chunk-based speech translation, and gold cascade separately.

# %% id="5a28c0a3"
# =========================================================
# Local Aggregate Folder
# =========================================================

aggregate_dir = BASE_OUTPUT_DIR / "aggregated_results"
aggregate_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# Create DataFrames
# =========================================================

# The notebook already creates these result DataFrames during the experiment.
# This cell collects them again into one clean aggregated_results folder.
all_results_df = aggregate_results_df.copy() if "aggregate_results_df" in globals() else pd.DataFrame()

text_predictions_export_df = text_predictions_df.copy() if "text_predictions_df" in globals() else pd.DataFrame()
audio_predictions_export_df = audio_predictions_df.copy() if "audio_predictions_df" in globals() else pd.DataFrame()
qualitative_all_export_df = qualitative_all_df.copy() if "qualitative_all_df" in globals() else pd.DataFrame()

sample_check_export_df = sample_check_df.copy() if "sample_check_df" in globals() else pd.DataFrame()

# EDA and audio-quality analysis outputs created in Section 4.1.
monitoring_df = aggregate_results_df.copy() if "aggregate_results_df" in globals() else pd.DataFrame()
quality_all_df = eda_audio_quality_df.copy() if "eda_audio_quality_df" in globals() else pd.DataFrame()
duration_all_df = eda_audio_quality_df[[c for c in ["language", "split", "sample_index", "african_duration_sec", "english_duration_sec", "african_words", "english_words", "word_length_difference"] if c in eda_audio_quality_df.columns]].copy() if "eda_audio_quality_df" in globals() else pd.DataFrame()
eda_metadata_export_df = eda_metadata_df.copy() if "eda_metadata_df" in globals() else pd.DataFrame()
eda_grouped_summary_export_df = eda_grouped_summary_df.copy() if "eda_grouped_summary_df" in globals() else pd.DataFrame()
eda_compact_summary_export_df = eda_compact_summary_df.copy() if "eda_compact_summary_df" in globals() else pd.DataFrame()

# =========================================================
# Save CSV Files Locally
# =========================================================

all_results_df.to_csv(
    aggregate_dir / "all_language_all_experiment_metrics.csv",
    index=False
)

text_predictions_export_df.to_csv(
    aggregate_dir / "all_language_text_predictions.csv",
    index=False
)

audio_predictions_export_df.to_csv(
    aggregate_dir / "all_language_audio_predictions.csv",
    index=False
)

qualitative_all_export_df.to_csv(
    aggregate_dir / "all_language_qualitative_outputs_to_fill.csv",
    index=False
)

sample_check_export_df.to_csv(
    aggregate_dir / "01_fleurs_bidirectional_sample_check.csv",
    index=False
)

monitoring_df.to_csv(
    aggregate_dir / "all_language_monitoring_summary.csv",
    index=False
)

quality_all_df.to_csv(
    aggregate_dir / "all_language_audio_quality.csv",
    index=False
)

duration_all_df.to_csv(
    aggregate_dir / "all_language_duration_analysis.csv",
    index=False
)

eda_metadata_export_df.to_csv(
    aggregate_dir / "all_language_dataset_properties.csv",
    index=False
)

eda_grouped_summary_export_df.to_csv(
    aggregate_dir / "all_language_eda_grouped_summary.csv",
    index=False
)

eda_compact_summary_export_df.to_csv(
    aggregate_dir / "all_language_paper_ready_data_quality_summary.csv",
    index=False
)

print("Saved aggregated results locally to:", aggregate_dir)

# =========================================================
# Save to Google Drive
# =========================================================

# =========================================================
# Create Timestamped Backup Folder
# =========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

DRIVE_BACKUP_ROOT = Path(
    "/content/drive/MyDrive/" + BASE_OUTPUT_FOLDER_NAME
)

DRIVE_BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

DRIVE_BACKUP_DIR = DRIVE_BACKUP_ROOT / f"experiment_{timestamp}"

# =========================================================
# Copy Entire Output Directory
# =========================================================

if DRIVE_BACKUP_DIR.exists():
    shutil.rmtree(DRIVE_BACKUP_DIR)

shutil.copytree(
    BASE_OUTPUT_DIR,
    DRIVE_BACKUP_DIR
)

print("=================================================")
print("Saved complete experiment backup to Google Drive")
print(DRIVE_BACKUP_DIR)
print("=================================================")

# =========================================================
# Optional: Create ZIP Backup Too
# =========================================================

zip_path = shutil.make_archive(
    str(DRIVE_BACKUP_DIR),
    'zip',
    root_dir=BASE_OUTPUT_DIR
)

print("ZIP backup created:")
print(zip_path)

# %% [markdown] id="01fb7c65"
# ## Visualize All BLEU Results
#
# This section visualizes aggregate BLEU results across languages, experiments, and strategies.
#
# Expected output: cross-language bar charts or comparable figures.
#
# How to interpret: use the figures to identify the strongest method per language and whether the same optimization strategy generalizes across languages.

# %% [markdown] id="88c4f327"
# ## Best Method per Language and Experiment
#
# This section identifies the best-scoring method for each language and experiment.
#
# Expected output: summary table showing the best method, category, score, language, and experiment.
#
# How to interpret: if the best method is usually gold cascade, then better ASR is needed. If the best method is normalized or trimmed audio, then preprocessing is a useful improvement strategy.

# %% [markdown] id="4d5fa273"
# # Final Interpretation
#
# This section provides a checklist for turning notebook outputs into paper discussion points.
#
# The team should answer:
#
# 1. Which language is easiest and hardest for the model?
# 2. Does transcript translation outperform direct audio translation?
# 3. Does audio normalization improve performance?
# 4. Does silence trimming help or hurt?
# 5. Does chunking improve long-audio examples?
# 6. Is the gold cascade much stronger than direct audio?
# 7. Are errors mostly hallucinations, omissions, wrong-language outputs, or truncations?
# 8. Are results consistent across experiment sizes?
#
# Use this section to write the findings, limitations, and future-work parts of the paper.

# %% id="873159a7"
print("Main interpretation checklist:")
print("1. Start in DEBUG_MODE=True with one language pair before running the full grid.")
print("2. Compare African→English vs English→African performance.")
print("3. Compare text-to-text vs speech-to-text translation.")
print("4. Identify languages where reverse translation is weaker.")
print("5. Use the EDA/audio-quality tables to discuss duration, silence, energy, clipping, and transcript length.")
print("6. Check whether low scores are caused by audio duration, speech recognition, or translation issues.")
print("7. Populate manual_error_category, manual_severity, and manual_comment in the qualitative CSV files.")
print("8. Report unsupported languages explicitly. Hausa is available in FLEURS but not natively supported by SeamlessM4T-v2.")
print("9. When the debug run works, set DEBUG_MODE=False and RUN_FULL_GRID=True for the full experiment.")

