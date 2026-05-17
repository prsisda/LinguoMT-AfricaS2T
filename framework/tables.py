from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _bleu(df: pd.DataFrame, **filters) -> float:
    if df is None or df.empty:
        return float("nan")
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in filters.items():
        if col in df.columns:
            mask &= df[col] == val
    sub = df[mask]
    return float(sub["BLEU"].iloc[0]) if not sub.empty and "BLEU" in sub.columns else float("nan")


def build_text_table(text_results: pd.DataFrame) -> pd.DataFrame:
    """T1 — text translation results."""
    if text_results.empty:
        return pd.DataFrame()
    cols = ["experiment", "language", "direction_label", "BLEU", "ChrF",
            "num_samples", "avg_pred_len", "avg_ref_len", "pred_ref_ratio"]
    return text_results[[c for c in cols if c in text_results.columns]].copy()


def build_audio_table(audio_results: pd.DataFrame) -> pd.DataFrame:
    """T2 — audio strategy results."""
    if audio_results.empty:
        return pd.DataFrame()
    cols = ["experiment", "language", "direction_label", "strategy_key", "strategy_label",
            "BLEU", "ChrF", "num_samples", "avg_duration", "runtime_s"]
    return audio_results[[c for c in cols if c in audio_results.columns]].copy()


def build_strategy_pivot(audio_results: pd.DataFrame) -> pd.DataFrame:
    """T3 — strategy × language BLEU pivot (latest experiment)."""
    if audio_results.empty or "strategy_label" not in audio_results.columns:
        return pd.DataFrame()
    try:
        latest = audio_results["experiment"].max()
        sub    = audio_results[audio_results["experiment"] == latest]
        return sub.pivot_table(
            index="language", columns="strategy_label", values="BLEU", aggfunc="mean"
        ).reset_index()
    except Exception:
        return pd.DataFrame()


def build_direction_pivot(aggregate: pd.DataFrame) -> pd.DataFrame:
    """T4 — direction × language BLEU pivot (latest experiment)."""
    if aggregate.empty or "direction_label" not in aggregate.columns:
        return pd.DataFrame()
    try:
        latest = aggregate["experiment"].max()
        sub    = aggregate[aggregate["experiment"] == latest]
        return sub.pivot_table(
            index="language", columns="direction_label", values="BLEU", aggfunc="mean"
        ).reset_index()
    except Exception:
        return pd.DataFrame()


def build_asr_table(asr_results: pd.DataFrame) -> pd.DataFrame:
    """T5 — ASR evaluation results."""
    if asr_results.empty:
        return pd.DataFrame()
    cols = ["experiment", "language", "WER", "CER", "num_samples"]
    return asr_results[[c for c in cols if c in asr_results.columns]].copy()


def build_summary_table(
    text_results: pd.DataFrame,
    audio_results: pd.DataFrame,
    audio_pred_df: pd.DataFrame,
    asr_results: pd.DataFrame,
    experiment_configs: list[dict],
    language_configs: list[dict],
    model_name: str,
    dataset_name: str,
    eda_sample_size: int,
    compute_fn,           # compute_translation_metrics callable
    short_thresh: float = 5.0,
    long_thresh: float  = 12.0,
) -> pd.DataFrame:
    """T6 — paper-ready summary (one row per experiment × language)."""
    rows = []
    for exp in experiment_configs:
        exp_name = exp["experiment"]
        for cfg in language_configs:
            lang = cfg["display"]
            a2e  = f"{lang}→English"
            e2a  = f"English→{lang}"

            # Duration-stratified audio BLEU
            short_bleu = long_bleu = float("nan")
            if not audio_pred_df.empty:
                sub = audio_pred_df[
                    (audio_pred_df.get("experiment", "") == exp_name) &
                    (audio_pred_df.get("strategy_key", "") == "baseline_direct") &
                    (audio_pred_df.get("direction_label", "") == a2e) &
                    (audio_pred_df.get("language", "") == lang)
                ] if all(c in audio_pred_df.columns for c in ["experiment", "strategy_key", "direction_label", "language"]) else pd.DataFrame()
                s = sub[sub["duration_s"] < short_thresh]  if "duration_s" in sub.columns else pd.DataFrame()
                l = sub[sub["duration_s"] >= long_thresh]  if "duration_s" in sub.columns else pd.DataFrame()
                if len(s) >= 1:
                    short_bleu = compute_fn(s["prediction"].tolist(), s["reference"].tolist())["BLEU"]
                if len(l) >= 1:
                    long_bleu  = compute_fn(l["prediction"].tolist(), l["reference"].tolist())["BLEU"]

            rows.append({
                "method":                   model_name,
                "language":                 lang,
                "experiment":               dataset_name,
                "experiment_config":        exp_name,
                "max_text_train_samples":   eda_sample_size,
                "max_text_dev_samples":     exp.get("max_text_dev", ""),
                "max_audio_dev_samples":    exp.get("max_audio_dev", ""),
                "Chunk-based audio → English": _bleu(audio_results, experiment=exp_name, strategy_key="chunk_based_audio", direction_label=a2e, language=lang),
                "Direct audio → English":      _bleu(audio_results, experiment=exp_name, strategy_key="baseline_direct",   direction_label=a2e, language=lang),
                "English → Source":            _bleu(text_results,  experiment=exp_name, direction_label=e2a, language=lang),
                "Gold ASR cascade":            _bleu(asr_results if not asr_results.empty else pd.DataFrame(),
                                                     experiment=exp_name, direction_label=a2e, language=lang) if not asr_results.empty else float("nan"),
                "Long audio → English":        long_bleu,
                "Normalized audio → English":  _bleu(audio_results, experiment=exp_name, strategy_key="normalized_audio",  direction_label=a2e, language=lang),
                "Short audio → English":       short_bleu,
                "Transcript → English":        _bleu(text_results,  experiment=exp_name, direction_label=a2e, language=lang),
                "Trimmed audio → English":     _bleu(audio_results, experiment=exp_name, strategy_key="trimmed_audio",     direction_label=a2e, language=lang),
            })
    return pd.DataFrame(rows)


def build_cross_language_comparisons(
    text_results: pd.DataFrame,
    audio_results: pd.DataFrame,
) -> pd.DataFrame:
    """T8 — BLEU pivoted by language × direction and mode (within one run)."""
    frames = []
    if not text_results.empty and "BLEU" in text_results.columns:
        frames.append(text_results.assign(mode="text"))
    if not audio_results.empty and "BLEU" in audio_results.columns:
        if "strategy_key" in audio_results.columns:
            base = audio_results[audio_results["strategy_key"] == "baseline_direct"]
        else:
            base = audio_results
        if not base.empty:
            frames.append(base.assign(mode="audio_baseline"))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    try:
        return combined.pivot_table(
            index=["mode", "direction_label"],
            columns="language",
            values="BLEU",
            aggfunc="mean",
        ).reset_index()
    except Exception:
        return pd.DataFrame()


def build_cross_model_comparisons(outputs_root: Path) -> pd.DataFrame:
    """T9 — BLEU aggregated across all run dirs found under outputs_root."""
    rows = []
    root = Path(outputs_root)
    for metric_file in root.glob("*/metrics/text_metrics.csv"):
        try:
            df = pd.read_csv(metric_file)
            if "model_name" in df.columns:
                rows.append(df)
        except Exception:
            pass
    for metric_file in root.glob("*/metrics/audio_metrics.csv"):
        try:
            df = pd.read_csv(metric_file)
            if "model_name" not in df.columns:
                continue
            if "strategy_key" in df.columns:
                df = df[df["strategy_key"] == "baseline_direct"]
            rows.append(df)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    try:
        return combined.pivot_table(
            index=["language", "direction_label"],
            columns="model_name",
            values="BLEU",
            aggfunc="mean",
        ).reset_index()
    except Exception:
        return pd.DataFrame()


def build_qualitative_table(pred_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [c for c in ["experiment", "language", "direction_label"] if c in pred_df.columns]
    for _, grp in pred_df.groupby(group_cols):
        for _, r in grp.head(n).iterrows():
            rows.append({
                "experiment":   r.get("experiment", ""),
                "language":     r.get("language", ""),
                "direction":    r.get("direction_label", ""),
                "strategy":     r.get("strategy_key", r.get("mode", "")),
                "source":       r.get("source_text", ""),
                "reference":    r.get("reference", ""),
                "prediction":   r.get("prediction", ""),
                "error":        r.get("error_message", ""),
                "manual_category": "",
                "manual_comment":  "",
            })
    return pd.DataFrame(rows)
