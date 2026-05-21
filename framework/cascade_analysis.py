from __future__ import annotations

"""
Cascade vs E2E analysis utilities for Paper 4 (LinguoMT-Cascade).

Functions:
  run_oracle_cascade     — run MT on gold source transcripts (text MT ceiling)
  run_error_propagation  — corrupt ASR transcripts at WER rates; measure BLEU drop
  run_breakeven_analysis — linear regression WER→BLEU; find break-even WER
  run_latency_profile    — measure median/P95 latency (ms) and peak VRAM (MB)
"""

import random
import time
import traceback
from typing import Callable

import numpy as np
import pandas as pd

from .metrics import compute_translation_metrics
from .audio import extract_audio_array


# ---------------------------------------------------------------------------
# Helper: pick the best translation lang code from a lang_cfg dict
# ---------------------------------------------------------------------------

def _pick_nllb_or_seamless(lang_cfg: dict) -> str | None:
    """Try nllb_code first, then seamless_code (for oracle cascade calls)."""
    return lang_cfg.get("nllb_code") or lang_cfg.get("seamless_code")


# ---------------------------------------------------------------------------
# 1. Oracle cascade: MT on gold source transcripts
# ---------------------------------------------------------------------------

def run_oracle_cascade(
    translate_text_fn: Callable,   # callable(text, src_lang, tgt_lang) -> str
    dev_cache,                     # DatasetCache
    lang_cfgs: list[dict],
    caps,                          # ModelCapabilities
    dirs,                          # OutputDirs | None
    monitor,                       # StepMonitor | None
) -> pd.DataFrame:
    """Run MT on gold source transcripts to produce a text MT ceiling.

    Computes BLEU and ChrF against gold English references.
    Writes oracle_cascade_metrics.csv with columns:
        language, language_key, BLEU, ChrF, num_samples
    """
    english_code = caps.english_code
    rows: list[dict] = []

    for lang_cfg in lang_cfgs:
        lk = lang_cfg["language_key"]
        display = lang_cfg.get("display", lk)
        src_lang_code = _pick_nllb_or_seamless(lang_cfg)

        print(f"[OracleCascade] {display}  src_lang={src_lang_code}")
        if monitor:
            monitor.step(f"OracleCascade {display}", f"src_lang={src_lang_code}")

        if not src_lang_code:
            print(f"  [OracleCascade] {display}: no nllb_code or seamless_code, skipping.")
            continue

        try:
            dev_pairs = dev_cache.get_pairs(lk, 200)
            if not dev_pairs:
                print(f"  [OracleCascade] {display}: no dev pairs, skipping.")
                continue

            preds, refs = [], []
            for pair in dev_pairs:
                src_text = pair.get("src_text", "")
                eng_text = pair.get("eng_text", "")
                try:
                    pred = translate_text_fn(src_text, src_lang_code, english_code)
                except Exception:
                    pred = ""
                preds.append(str(pred).strip())
                refs.append(eng_text)

            mt_metrics = compute_translation_metrics(preds, refs)
            row = {
                "language":     display,
                "language_key": lk,
                "BLEU":         mt_metrics.get("BLEU", float("nan")),
                "ChrF":         mt_metrics.get("ChrF", float("nan")),
                "num_samples":  mt_metrics.get("num_samples", len(preds)),
            }
            rows.append(row)
            print(f"  [OracleCascade] {display}: BLEU={row['BLEU']:.2f}  ChrF={row['ChrF']:.2f}  n={row['num_samples']}")

        except Exception:
            print(f"  [OracleCascade] ERROR {display}:\n{traceback.format_exc()}")

    df = pd.DataFrame(rows, columns=["language", "language_key", "BLEU", "ChrF", "num_samples"])

    if dirs is not None:
        out_path = dirs.metrics / "oracle_cascade_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"[OracleCascade] Written: {out_path}")
        if monitor:
            monitor.step("OracleCascade CSV written", str(out_path))

    return df


# ---------------------------------------------------------------------------
# 2. Error propagation: corrupt transcripts at varying WER rates
# ---------------------------------------------------------------------------

def run_error_propagation(
    translate_text_fn: Callable,   # callable(text, src_lang, tgt_lang) -> str
    dev_cache,                     # DatasetCache
    lang_cfgs: list[dict],
    caps,                          # ModelCapabilities
    dirs,                          # OutputDirs | None
    monitor,                       # StepMonitor | None
    wer_rates: list[float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Corrupt source transcripts at different WER rates; measure BLEU degradation.

    For each WER rate, words are randomly replaced with '<unk>' tokens.
    Writes error_propagation_metrics.csv with columns:
        language, language_key, wer_rate, BLEU, ChrF
    """
    if wer_rates is None:
        wer_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    english_code = caps.english_code
    rows: list[dict] = []

    for lang_cfg in lang_cfgs:
        lk = lang_cfg["language_key"]
        display = lang_cfg.get("display", lk)
        src_lang_code = _pick_nllb_or_seamless(lang_cfg)

        print(f"[ErrorProp] {display}")
        if monitor:
            monitor.step(f"ErrorProp {display}", f"wer_rates={wer_rates}")

        if not src_lang_code:
            print(f"  [ErrorProp] {display}: no lang code, skipping.")
            continue

        try:
            dev_pairs = dev_cache.get_pairs(lk, 200)
            if not dev_pairs:
                print(f"  [ErrorProp] {display}: no dev pairs, skipping.")
                continue

            # Gold transcripts (src_text) and their reference English translations
            gold_transcripts = [p.get("src_text", "") for p in dev_pairs]
            eng_refs = [p.get("eng_text", "") for p in dev_pairs]

            for wer_rate in wer_rates:
                random.seed(seed)

                corrupted: list[str] = []
                for transcript in gold_transcripts:
                    words = transcript.split()
                    if not words:
                        corrupted.append(transcript)
                        continue
                    n_corrupt = int(round(wer_rate * len(words)))
                    if n_corrupt == 0:
                        corrupted.append(transcript)
                        continue
                    # Choose random positions to corrupt (without replacement)
                    positions = random.sample(range(len(words)), min(n_corrupt, len(words)))
                    corrupted_words = words[:]
                    for pos in positions:
                        corrupted_words[pos] = "<unk>"
                    corrupted.append(" ".join(corrupted_words))

                preds = []
                for corr_text in corrupted:
                    try:
                        pred = translate_text_fn(corr_text, src_lang_code, english_code)
                    except Exception:
                        pred = ""
                    preds.append(str(pred).strip())

                mt_metrics = compute_translation_metrics(preds, eng_refs)
                row = {
                    "language":     display,
                    "language_key": lk,
                    "wer_rate":     wer_rate,
                    "BLEU":         mt_metrics.get("BLEU", float("nan")),
                    "ChrF":         mt_metrics.get("ChrF", float("nan")),
                }
                rows.append(row)
                print(f"  [ErrorProp] {display}  wer_rate={wer_rate:.1f}: "
                      f"BLEU={row['BLEU']:.2f}  ChrF={row['ChrF']:.2f}")

        except Exception:
            print(f"  [ErrorProp] ERROR {display}:\n{traceback.format_exc()}")

    df = pd.DataFrame(rows, columns=["language", "language_key", "wer_rate", "BLEU", "ChrF"])

    if dirs is not None:
        out_path = dirs.metrics / "error_propagation_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"[ErrorProp] Written: {out_path}")
        if monitor:
            monitor.step("ErrorProp CSV written", str(out_path))

    return df


# ---------------------------------------------------------------------------
# 3. Break-even analysis: where does cascade BLEU cross E2E BLEU?
# ---------------------------------------------------------------------------

def run_breakeven_analysis(
    error_prop_df: pd.DataFrame,
    e2e_bleu_by_lang: dict,        # language_key → float (E2E BLEU)
    dirs,                          # OutputDirs | None
) -> pd.DataFrame:
    """Fit linear regression BLEU ~ wer_rate per language; find break-even WER.

    Break-even WER is where the regression line for cascade BLEU crosses the
    E2E BLEU value.  If the line never crosses, break-even is set to None.

    Writes breakeven_metrics.csv with columns:
        language, language_key, breakeven_wer, e2e_bleu, cascade_bleu_at_0wer,
        regression_slope, regression_intercept
    """
    rows: list[dict] = []

    for lk, group in error_prop_df.groupby("language_key"):
        display = group["language"].iloc[0] if "language" in group.columns else lk
        e2e_bleu = e2e_bleu_by_lang.get(lk)

        group_sorted = group.sort_values("wer_rate")
        wer_vals = group_sorted["wer_rate"].values.astype(float)
        bleu_vals = group_sorted["BLEU"].values.astype(float)

        # Filter out NaN
        valid_mask = ~(np.isnan(wer_vals) | np.isnan(bleu_vals))
        wer_vals = wer_vals[valid_mask]
        bleu_vals = bleu_vals[valid_mask]

        slope = intercept = float("nan")
        cascade_bleu_at_0 = float("nan")
        breakeven_wer = None

        if len(wer_vals) >= 2:
            try:
                coeffs = np.polyfit(wer_vals, bleu_vals, 1)
                slope = float(coeffs[0])
                intercept = float(coeffs[1])
                cascade_bleu_at_0 = intercept  # BLEU at wer_rate=0

                if e2e_bleu is not None and not np.isnan(slope) and slope != 0.0:
                    # Solve: slope * wer + intercept = e2e_bleu  →  wer = (e2e_bleu - intercept) / slope
                    candidate = (e2e_bleu - intercept) / slope
                    # Only valid if it falls in [0, 1]
                    if 0.0 <= candidate <= 1.0:
                        breakeven_wer = round(float(candidate), 4)

            except Exception:
                pass

        row = {
            "language":               display,
            "language_key":           lk,
            "breakeven_wer":          breakeven_wer,
            "e2e_bleu":               e2e_bleu,
            "cascade_bleu_at_0wer":   cascade_bleu_at_0,
            "regression_slope":       slope,
            "regression_intercept":   intercept,
        }
        rows.append(row)
        print(f"[Breakeven] {display}: slope={slope:.3f}  intercept={intercept:.3f}  "
              f"breakeven_wer={breakeven_wer}  e2e_bleu={e2e_bleu}")

    df = pd.DataFrame(rows, columns=[
        "language", "language_key", "breakeven_wer", "e2e_bleu",
        "cascade_bleu_at_0wer", "regression_slope", "regression_intercept",
    ])

    if dirs is not None:
        out_path = dirs.metrics / "breakeven_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"[Breakeven] Written: {out_path}")

    return df


# ---------------------------------------------------------------------------
# 4. Latency profile: median / P95 ms and peak VRAM MB
# ---------------------------------------------------------------------------

def run_latency_profile(
    translate_audio_fn: Callable | None,  # callable(audio_arr, src_lang, tgt_lang) -> str  (E2E)
    asr_fn: Callable | None,              # callable(audio_arr, src_lang) -> str
    translate_text_fn: Callable | None,   # callable(text, src_lang, tgt_lang) -> str
    dev_cache,                            # DatasetCache
    lang_cfgs: list[dict],
    caps,                                 # ModelCapabilities
    dirs,                                 # OutputDirs | None
    monitor,                              # StepMonitor | None
    n_warmup: int = 5,
    n_measure: int = 50,
) -> pd.DataFrame:
    """Measure median/P95 latency (ms) and peak VRAM (MB) for cascade vs E2E.

    Uses the first language with available dev pairs and audio data.
    Writes latency_metrics.csv with columns:
        architecture, median_ms, p95_ms, peak_vram_mb, n_samples
    """
    # Optional CUDA
    try:
        import torch
        _cuda_ok = torch.cuda.is_available()
    except Exception:
        torch = None  # type: ignore[assignment]
        _cuda_ok = False

    english_code = caps.english_code

    # Find the first language with pairs
    sample_pairs: list[dict] = []
    sample_lang_cfg: dict | None = None

    for lang_cfg in lang_cfgs:
        lk = lang_cfg["language_key"]
        pairs = dev_cache.get_pairs(lk, n_warmup + n_measure)
        if pairs:
            sample_pairs = pairs
            sample_lang_cfg = lang_cfg
            break

    if not sample_pairs or sample_lang_cfg is None:
        print("[LatencyProfile] No dev pairs found for any language — returning empty DataFrame.")
        return pd.DataFrame(columns=["architecture", "median_ms", "p95_ms", "peak_vram_mb", "n_samples"])

    lk = sample_lang_cfg["language_key"]
    display = sample_lang_cfg.get("display", lk)
    src_lang_code = _pick_nllb_or_seamless(sample_lang_cfg)
    asr_lang_attr = caps.asr_lang_attr
    asr_lang_code = sample_lang_cfg.get(asr_lang_attr) if asr_lang_attr else None

    print(f"[LatencyProfile] Using language: {display}  n_pairs={len(sample_pairs)}")
    if monitor:
        monitor.step("LatencyProfile", f"lang={display}  n={len(sample_pairs)}")

    def _reset_vram():
        if _cuda_ok and torch is not None:
            torch.cuda.reset_peak_memory_stats()

    def _peak_vram_mb() -> float | None:
        if _cuda_ok and torch is not None:
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return None

    rows: list[dict] = []

    # ---- Cascade: ASR → MT ----
    if asr_fn is not None and translate_text_fn is not None and asr_lang_code and src_lang_code:
        print("[LatencyProfile] Profiling cascade (ASR → MT) …")
        timings: list[float] = []
        _reset_vram()
        vram_before = _peak_vram_mb() or 0.0

        for i, pair in enumerate(sample_pairs[: n_warmup + n_measure]):
            audio_obj = pair.get("src_audio")
            try:
                audio_arr = extract_audio_array(audio_obj)
            except Exception:
                continue

            t0 = time.perf_counter()
            try:
                transcript = asr_fn(audio_arr, asr_lang_code)
                translate_text_fn(str(transcript).strip(), src_lang_code, english_code)
            except Exception:
                pass
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if i >= n_warmup:  # discard warm-up
                timings.append(elapsed_ms)

        vram_peak = _peak_vram_mb()
        peak_vram_mb = (vram_peak - vram_before) if vram_peak is not None else None

        if timings:
            arr = np.array(timings)
            rows.append({
                "architecture": "cascade",
                "median_ms":    float(np.median(arr)),
                "p95_ms":       float(np.percentile(arr, 95)),
                "peak_vram_mb": peak_vram_mb,
                "n_samples":    len(timings),
            })
            print(f"  [LatencyProfile] cascade: median={np.median(arr):.1f}ms  "
                  f"p95={np.percentile(arr, 95):.1f}ms  "
                  f"vram={peak_vram_mb:.1f}MB" if peak_vram_mb is not None
                  else f"  [LatencyProfile] cascade: median={np.median(arr):.1f}ms  p95={np.percentile(arr, 95):.1f}ms  vram=N/A")

    # ---- E2E: direct audio → translation ----
    if translate_audio_fn is not None and src_lang_code:
        print("[LatencyProfile] Profiling E2E (audio → translation) …")
        timings_e2e: list[float] = []
        _reset_vram()
        vram_before = _peak_vram_mb() or 0.0

        for i, pair in enumerate(sample_pairs[: n_warmup + n_measure]):
            audio_obj = pair.get("src_audio")
            try:
                audio_arr = extract_audio_array(audio_obj)
            except Exception:
                continue

            t0 = time.perf_counter()
            try:
                translate_audio_fn(audio_arr, src_lang_code, english_code)
            except Exception:
                pass
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if i >= n_warmup:
                timings_e2e.append(elapsed_ms)

        vram_peak = _peak_vram_mb()
        peak_vram_mb = (vram_peak - vram_before) if vram_peak is not None else None

        if timings_e2e:
            arr = np.array(timings_e2e)
            rows.append({
                "architecture": "e2e",
                "median_ms":    float(np.median(arr)),
                "p95_ms":       float(np.percentile(arr, 95)),
                "peak_vram_mb": peak_vram_mb,
                "n_samples":    len(timings_e2e),
            })
            print(f"  [LatencyProfile] e2e: median={np.median(arr):.1f}ms  "
                  f"p95={np.percentile(arr, 95):.1f}ms  "
                  f"vram={peak_vram_mb:.1f}MB" if peak_vram_mb is not None
                  else f"  [LatencyProfile] e2e: median={np.median(arr):.1f}ms  p95={np.percentile(arr, 95):.1f}ms  vram=N/A")

    df = pd.DataFrame(rows, columns=["architecture", "median_ms", "p95_ms", "peak_vram_mb", "n_samples"])

    if dirs is not None:
        out_path = dirs.metrics / "latency_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"[LatencyProfile] Written: {out_path}")
        if monitor:
            monitor.step("LatencyProfile CSV written", str(out_path))

    return df
