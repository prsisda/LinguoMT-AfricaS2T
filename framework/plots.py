from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

_PLT_READY = False
try:
    import matplotlib
    import matplotlib.pyplot as plt
    _PLT_READY = True
except Exception:
    pass


def _save(path: Path, in_colab: bool = False) -> None:
    if not _PLT_READY:
        return
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if in_colab:
        plt.show()
    plt.close()
    print(f"  Plot: {path.name}")


def setup_backend(in_colab: bool) -> None:
    if not _PLT_READY:
        return
    if not in_colab:
        matplotlib.use("Agg")


# ── EDA plots ─────────────────────────────────────────────────────────────────

def plot_duration_distribution(eda_df: pd.DataFrame, out_dir: Path, in_colab: bool = False) -> None:
    if not _PLT_READY or eda_df.empty or "duration_sec" not in eda_df.columns:
        return
    plt.figure(figsize=(9, 4))
    for lang, grp in eda_df.groupby("language"):
        plt.hist(grp["duration_sec"].dropna(), bins=20, alpha=0.6, label=lang)
    plt.xlabel("Duration (s)"); plt.ylabel("Count")
    plt.title("Audio Duration Distribution by Language"); plt.legend()
    _save(out_dir / "duration_distribution.png", in_colab)


def plot_silence_ratio(eda_df: pd.DataFrame, out_dir: Path, in_colab: bool = False) -> None:
    if not _PLT_READY or eda_df.empty or "silence_ratio" not in eda_df.columns:
        return
    plt.figure(figsize=(9, 4))
    for lang, grp in eda_df.groupby("language"):
        plt.hist(grp["silence_ratio"].dropna(), bins=20, alpha=0.6, label=lang)
    plt.xlabel("Silence Ratio"); plt.ylabel("Count")
    plt.title("Silence Ratio Distribution by Language"); plt.legend()
    _save(out_dir / "silence_ratio_distribution.png", in_colab)


def plot_energy_distribution(eda_df: pd.DataFrame, out_dir: Path, in_colab: bool = False) -> None:
    if not _PLT_READY or eda_df.empty or "rms_energy" not in eda_df.columns:
        return
    plt.figure(figsize=(9, 4))
    for lang, grp in eda_df.groupby("language"):
        plt.hist(grp["rms_energy"].dropna(), bins=20, alpha=0.6, label=lang)
    plt.xlabel("RMS Energy"); plt.ylabel("Count")
    plt.title("Audio Energy Distribution by Language"); plt.legend()
    _save(out_dir / "energy_distribution.png", in_colab)


def plot_waveform_comparison(audio_raw: np.ndarray, lang_name: str, sr: int, out_dir: Path, in_colab: bool = False) -> None:
    if not _PLT_READY:
        return
    from .audio import normalize_waveform
    original   = np.asarray(audio_raw, dtype=np.float32).flatten()
    normalized = normalize_waveform(original.copy())
    t          = np.arange(len(original)) / float(sr)
    fig, axes  = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(t, original,   color="steelblue",  linewidth=0.6)
    axes[0].set_title(f"{lang_name} — Original Waveform"); axes[0].set_ylabel("Amplitude")
    axes[1].plot(t, normalized, color="darkorange", linewidth=0.6)
    axes[1].set_title(f"{lang_name} — Normalized Waveform")
    axes[1].set_ylabel("Amplitude"); axes[1].set_xlabel("Time (s)")
    _save(out_dir / f"waveform_{lang_name.lower().replace(' ', '_')}.png", in_colab)


# ── Results plots ──────────────────────────────────────────────────────────────

def plot_bleu_by_language_direction(df: pd.DataFrame, out_dir: Path, title: str = "", in_colab: bool = False) -> None:
    if not _PLT_READY or df.empty or "BLEU" not in df.columns or "language" not in df.columns:
        return
    try:
        piv = df.pivot_table(index="language", columns="direction_label", values="BLEU", aggfunc="mean")
        ax  = piv.plot(kind="bar", colormap="tab10", figsize=(10, 5), rot=30)
        ax.set_title(title or "BLEU by Language × Direction")
        ax.set_ylabel("BLEU"); ax.set_xlabel("")
        _save(out_dir / "bleu_language_direction.png", in_colab)
    except Exception:
        pass


def plot_strategy_comparison(audio_df: pd.DataFrame, out_dir: Path, in_colab: bool = False) -> None:
    if not _PLT_READY or audio_df.empty or "strategy_label" not in audio_df.columns:
        return
    try:
        piv = audio_df.pivot_table(index="strategy_label", columns="language", values="ChrF", aggfunc="mean")
        ax  = piv.plot(kind="bar", colormap="Set2", figsize=(10, 5), rot=30)
        ax.set_title("Audio Strategy ChrF by Language")
        ax.set_ylabel("ChrF"); ax.set_xlabel("Strategy")
        _save(out_dir / "strategy_chrf_by_language.png", in_colab)
    except Exception:
        pass


def plot_direction_comparison(df: pd.DataFrame, out_dir: Path, in_colab: bool = False) -> None:
    if not _PLT_READY or df.empty or "direction_label" not in df.columns or "BLEU" not in df.columns:
        return
    try:
        piv = df.pivot_table(index="language", columns="direction_label", values="BLEU", aggfunc="mean")
        ax  = piv.plot(kind="bar", colormap="Paired", figsize=(10, 5), rot=30)
        ax.set_title("Translation BLEU: Direction Comparison")
        ax.set_ylabel("BLEU"); ax.set_xlabel("")
        _save(out_dir / "direction_comparison_bleu.png", in_colab)
    except Exception:
        pass


def plot_experiment_progression(aggregate_df: pd.DataFrame, out_dir: Path, in_colab: bool = False) -> None:
    if not _PLT_READY or aggregate_df.empty or "experiment" not in aggregate_df.columns:
        return
    try:
        grp = aggregate_df.groupby("experiment")[["BLEU", "ChrF"]].mean().reset_index()
        if len(grp) < 2:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(grp["experiment"], grp["BLEU"], marker="o", label="BLEU")
        plt.plot(grp["experiment"], grp["ChrF"], marker="s", label="ChrF")
        plt.title("Score Progression by Experiment Size")
        plt.ylabel("Score"); plt.xlabel("Experiment"); plt.legend()
        _save(out_dir / "experiment_progression.png", in_colab)
    except Exception:
        pass


def plot_asr_quality(asr_df: pd.DataFrame, out_dir: Path, in_colab: bool = False) -> None:
    if not _PLT_READY or asr_df.empty or "WER" not in asr_df.columns:
        return
    try:
        piv = asr_df.pivot_table(index="language", columns="experiment", values="WER", aggfunc="mean")
        ax  = piv.plot(kind="bar", figsize=(9, 5), rot=30)
        ax.set_title("ASR WER by Language × Experiment")
        ax.set_ylabel("WER (lower = better)"); ax.set_xlabel("")
        _save(out_dir / "asr_wer_by_language.png", in_colab)
    except Exception:
        pass
