from __future__ import annotations

"""
Cross-lingual transfer analysis utilities for Paper 5 (LinguoMT-Transfer).

Functions:
  compute_typological_similarity — URIEL syntax_knn cosine similarity via lang2vec
  run_crosslingual_transfer      — fine-tune on source lang, evaluate on target lang
  run_few_shot_scaling           — multiple-budget fine-tuning learning curve
  run_interaction_regression     — log(samples) x language_family interaction test
"""

import math
import traceback
import warnings
from typing import Callable

import numpy as np
import pandas as pd

from .metrics import compute_translation_metrics, compute_asr_metrics
from .languages import get_model_lang_code
from .audio import extract_audio_array


# ---------------------------------------------------------------------------
# 1. Typological similarity via lang2vec
# ---------------------------------------------------------------------------

def compute_typological_similarity(
    lang_pairs: list[tuple[str, str, str]],  # (lang_a_code, lang_b_code, label)  ISO 639-3
    feature_set: str = "syntax_knn",
) -> pd.DataFrame:
    """Compute URIEL cosine similarity for each (lang_a, lang_b) pair via lang2vec.

    lang_pairs entries are (iso3_a, iso3_b, human_readable_label).
    Writes typological_similarity.csv with columns:
        lang_a, lang_b, label, cosine_similarity, feature_set
    Returns empty DataFrame with a warning when lang2vec is unavailable.
    """
    try:
        import lang2vec.lang2vec as l2v
        _L2V_OK = True
    except Exception:
        _L2V_OK = False
        warnings.warn(
            "[TypologicalSim] lang2vec is not installed — returning empty DataFrame. "
            "Install it with: pip install lang2vec",
            RuntimeWarning,
            stacklevel=2,
        )

    if not _L2V_OK:
        return pd.DataFrame(columns=["lang_a", lang_b_col := "lang_b", "label",
                                     "cosine_similarity", "feature_set"])

    rows: list[dict] = []

    for code_a, code_b, label in lang_pairs:
        print(f"[TypSim] {code_a} — {code_b}  ({label})")
        try:
            vectors = l2v.query(feature_set, [code_a, code_b])
            v_a = np.array(vectors[code_a], dtype=float)
            v_b = np.array(vectors[code_b], dtype=float)

            # Filter positions where both vectors have valid (non-NaN) values
            valid = ~(np.isnan(v_a) | np.isnan(v_b))
            if valid.sum() == 0:
                print(f"  [TypSim] WARNING: all features are NaN for {code_a} or {code_b}. Setting sim=NaN.")
                cosine_sim = float("nan")
            else:
                va_clean = v_a[valid]
                vb_clean = v_b[valid]
                norm_a = np.linalg.norm(va_clean)
                norm_b = np.linalg.norm(vb_clean)
                if norm_a == 0.0 or norm_b == 0.0:
                    cosine_sim = float("nan")
                else:
                    cosine_sim = float(np.dot(va_clean, vb_clean) / (norm_a * norm_b))

            rows.append({
                "lang_a":           code_a,
                "lang_b":           code_b,
                "label":            label,
                "cosine_similarity": cosine_sim,
                "feature_set":      feature_set,
            })
            print(f"  [TypSim] cosine_similarity={cosine_sim:.4f}" if not math.isnan(cosine_sim) else
                  f"  [TypSim] cosine_similarity=NaN")

        except Exception:
            print(f"  [TypSim] ERROR for ({code_a}, {code_b}):\n{traceback.format_exc()}")

    df = pd.DataFrame(rows, columns=["lang_a", "lang_b", "label", "cosine_similarity", "feature_set"])
    return df


# ---------------------------------------------------------------------------
# 2. Cross-lingual transfer: fine-tune on one language, evaluate on others
# ---------------------------------------------------------------------------

def run_crosslingual_transfer(
    train_lang_cfg: dict,
    eval_lang_cfgs: list[dict],
    train_cache,                          # DatasetCache
    dev_cache,                            # DatasetCache
    finetune_fn: Callable,                # callable(pairs, lang_cfg, exp_dict)
    translate_text_fn: Callable | None,   # callable(text, src_lang, tgt_lang) -> str
    asr_fn: Callable | None,              # callable(audio_arr, src_lang) -> str
    caps,                                 # ModelCapabilities
    dirs,                                 # OutputDirs | None
    monitor,                              # StepMonitor | None
    n_train: int = 1000,
) -> pd.DataFrame:
    """Fine-tune on train_lang_cfg, then evaluate on every language in eval_lang_cfgs.

    Text eval: BLEU (source→English, up to 100 dev pairs).
    ASR eval:  WER  (up to 50 dev pairs if asr_fn is provided).

    Writes crosslingual_transfer_metrics.csv with columns:
        train_language, eval_language, train_language_key, eval_language_key,
        task, BLEU, ChrF, WER, CER, n_train_samples
    """
    english_code = caps.english_code
    train_lk = train_lang_cfg["language_key"]
    train_display = train_lang_cfg.get("display", train_lk)

    # --- Fine-tune on source language ---
    print(f"\n[XLTransfer] Fine-tuning on {train_display} (n_train={n_train}) …")
    if monitor:
        monitor.step(f"XLTransfer finetune {train_display}", f"n_train={n_train}")

    try:
        train_pairs = train_cache.get_pairs(train_lk, n_train)
        actual_n_train = len(train_pairs)
        finetune_fn(
            train_pairs,
            train_lang_cfg,
            {"experiment": f"xl_transfer_from_{train_lk}", "max_text_train": n_train},
        )
        print(f"  [XLTransfer] Fine-tuning complete on {actual_n_train} pairs.")
    except Exception:
        actual_n_train = 0
        print(f"  [XLTransfer] ERROR during fine-tuning:\n{traceback.format_exc()}")

    rows: list[dict] = []

    # --- Evaluate on each target language ---
    for eval_lang_cfg in eval_lang_cfgs:
        eval_lk = eval_lang_cfg["language_key"]
        eval_display = eval_lang_cfg.get("display", eval_lk)
        model_lang_code = get_model_lang_code(eval_lang_cfg, caps)

        print(f"  [XLTransfer] Evaluating on {eval_display} …")
        if monitor:
            monitor.step(f"  XLTransfer eval {eval_display}", f"train={train_display}")

        # Text MT
        if translate_text_fn is not None:
            try:
                dev_pairs = dev_cache.get_pairs(eval_lk, 100)
                if dev_pairs:
                    preds, refs = [], []
                    for pair in dev_pairs:
                        src_text = pair.get("src_text", "")
                        eng_text = pair.get("eng_text", "")
                        try:
                            pred = translate_text_fn(src_text, model_lang_code, english_code)
                        except Exception:
                            pred = ""
                        preds.append(str(pred).strip())
                        refs.append(eng_text)
                    mt_metrics = compute_translation_metrics(preds, refs)
                    rows.append({
                        "train_language":     train_display,
                        "eval_language":      eval_display,
                        "train_language_key": train_lk,
                        "eval_language_key":  eval_lk,
                        "task":              "text_mt",
                        "BLEU":              mt_metrics.get("BLEU", float("nan")),
                        "ChrF":              mt_metrics.get("ChrF", float("nan")),
                        "WER":               float("nan"),
                        "CER":               float("nan"),
                        "n_train_samples":   actual_n_train,
                    })
                    print(f"    text_mt: BLEU={mt_metrics.get('BLEU', 0):.2f}  ChrF={mt_metrics.get('ChrF', 0):.2f}")
            except Exception:
                print(f"    [XLTransfer] ERROR text_mt eval {eval_display}:\n{traceback.format_exc()}")

        # ASR
        if asr_fn is not None:
            try:
                asr_lang_attr = caps.asr_lang_attr
                asr_lang_code = eval_lang_cfg.get(asr_lang_attr) if asr_lang_attr else None
                dev_pairs = dev_cache.get_pairs(eval_lk, 50)
                if dev_pairs and asr_lang_code:
                    preds, refs = [], []
                    for pair in dev_pairs:
                        src_audio = pair.get("src_audio")
                        src_text = pair.get("src_text", "")
                        try:
                            audio_arr = extract_audio_array(src_audio)
                            pred = asr_fn(audio_arr, asr_lang_code)
                        except Exception:
                            pred = ""
                        preds.append(str(pred).strip())
                        refs.append(src_text)
                    asr_metrics = compute_asr_metrics(preds, refs)
                    rows.append({
                        "train_language":     train_display,
                        "eval_language":      eval_display,
                        "train_language_key": train_lk,
                        "eval_language_key":  eval_lk,
                        "task":              "asr",
                        "BLEU":              float("nan"),
                        "ChrF":              float("nan"),
                        "WER":               asr_metrics.get("WER", float("nan")),
                        "CER":               asr_metrics.get("CER", float("nan")),
                        "n_train_samples":   actual_n_train,
                    })
                    print(f"    asr: WER={asr_metrics.get('WER', float('nan')):.3f}  "
                          f"CER={asr_metrics.get('CER', float('nan')):.3f}")
            except Exception:
                print(f"    [XLTransfer] ERROR asr eval {eval_display}:\n{traceback.format_exc()}")

    df = pd.DataFrame(rows, columns=[
        "train_language", "eval_language", "train_language_key", "eval_language_key",
        "task", "BLEU", "ChrF", "WER", "CER", "n_train_samples",
    ])

    if dirs is not None:
        out_path = dirs.metrics / "crosslingual_transfer_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"[XLTransfer] Written: {out_path}")
        if monitor:
            monitor.step("XLTransfer CSV written", str(out_path))

    return df


# ---------------------------------------------------------------------------
# 3. Few-shot scaling: same logic as Paper 2 scaling, different output file
# ---------------------------------------------------------------------------

def run_few_shot_scaling(
    budgets: list[int],
    lang_cfgs: list[dict],
    train_cache,
    dev_cache,
    finetune_fn: Callable,
    translate_text_fn: Callable | None,
    asr_fn: Callable | None,
    caps,
    dirs,
    monitor,
) -> pd.DataFrame:
    """Run few-shot scaling (same logic as scaling.py) writing few_shot_scaling_metrics.csv.

    Columns: budget, language, language_key, task, BLEU, ChrF, WER, CER
    """
    english_code = caps.english_code
    rows: list[dict] = []

    for budget in budgets:
        print(f"\n[FewShotScaling] ===== Budget: {budget} =====")
        if monitor:
            monitor.step(f"FewShotScaling budget={budget}", f"{len(lang_cfgs)} languages")

        # Fine-tune all languages at this budget
        for lang_cfg in lang_cfgs:
            lk = lang_cfg["language_key"]
            display = lang_cfg.get("display", lk)
            try:
                train_pairs = train_cache.get_pairs(lk, budget)
                if not train_pairs:
                    print(f"  [FewShot] {display}: no training pairs, skipping.")
                    continue
                print(f"  [FewShot] Fine-tuning {display} on {len(train_pairs)} pairs …")
                if monitor:
                    monitor.step(f"  FewShot finetune {display}", f"budget={budget}")
                finetune_fn(
                    train_pairs,
                    lang_cfg,
                    {"experiment": f"few_shot_{budget}", "max_text_train": budget},
                )
            except Exception:
                print(f"  [FewShot] ERROR fine-tuning {display}:\n{traceback.format_exc()}")

        # Evaluate all languages
        for lang_cfg in lang_cfgs:
            lk = lang_cfg["language_key"]
            display = lang_cfg.get("display", lk)
            model_lang_code = get_model_lang_code(lang_cfg, caps)

            # Text MT
            if translate_text_fn is not None:
                try:
                    dev_pairs = dev_cache.get_pairs(lk, 100)
                    if dev_pairs:
                        preds, refs = [], []
                        for pair in dev_pairs:
                            src_text = pair.get("src_text", "")
                            eng_text = pair.get("eng_text", "")
                            try:
                                pred = translate_text_fn(src_text, model_lang_code, english_code)
                            except Exception:
                                pred = ""
                            preds.append(str(pred).strip())
                            refs.append(eng_text)
                        mt_metrics = compute_translation_metrics(preds, refs)
                        rows.append({
                            "budget":       budget,
                            "language":     display,
                            "language_key": lk,
                            "task":         "text_mt",
                            "BLEU":         mt_metrics.get("BLEU", float("nan")),
                            "ChrF":         mt_metrics.get("ChrF", float("nan")),
                            "WER":          float("nan"),
                            "CER":          float("nan"),
                        })
                        print(f"  [FewShot] {display} text_mt budget={budget}: "
                              f"BLEU={mt_metrics.get('BLEU', 0):.2f}")
                except Exception:
                    print(f"  [FewShot] ERROR text_mt {display}:\n{traceback.format_exc()}")

            # ASR
            if asr_fn is not None:
                try:
                    asr_lang_attr = caps.asr_lang_attr
                    asr_lang_code = lang_cfg.get(asr_lang_attr) if asr_lang_attr else None
                    dev_pairs = dev_cache.get_pairs(lk, 50)
                    if dev_pairs and asr_lang_code:
                        preds, refs = [], []
                        for pair in dev_pairs:
                            src_audio = pair.get("src_audio")
                            src_text = pair.get("src_text", "")
                            try:
                                audio_arr = extract_audio_array(src_audio)
                                pred = asr_fn(audio_arr, asr_lang_code)
                            except Exception:
                                pred = ""
                            preds.append(str(pred).strip())
                            refs.append(src_text)
                        asr_metrics = compute_asr_metrics(preds, refs)
                        rows.append({
                            "budget":       budget,
                            "language":     display,
                            "language_key": lk,
                            "task":         "asr",
                            "BLEU":         float("nan"),
                            "ChrF":         float("nan"),
                            "WER":          asr_metrics.get("WER", float("nan")),
                            "CER":          asr_metrics.get("CER", float("nan")),
                        })
                        print(f"  [FewShot] {display} asr     budget={budget}: "
                              f"WER={asr_metrics.get('WER', float('nan')):.3f}")
                except Exception:
                    print(f"  [FewShot] ERROR asr {display}:\n{traceback.format_exc()}")

        if monitor:
            monitor.step(f"FewShotScaling budget={budget} complete", f"{len(rows)} rows")

    df = pd.DataFrame(rows, columns=["budget", "language", "language_key", "task",
                                     "BLEU", "ChrF", "WER", "CER"])

    if dirs is not None:
        out_path = dirs.metrics / "few_shot_scaling_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"[FewShotScaling] Written: {out_path}")
        if monitor:
            monitor.step("FewShotScaling CSV written", str(out_path))

    return df


# ---------------------------------------------------------------------------
# 4. Interaction regression: log(budget) × language_family
# ---------------------------------------------------------------------------

def run_interaction_regression(
    scaling_df: pd.DataFrame,
    lang_family_map: dict,   # language_key → family name
    dirs,                    # OutputDirs | None
) -> pd.DataFrame:
    """OLS regression: metric ~ log(budget+1) + family + log(budget+1)*family.

    Reports the interaction coefficient and approximate t/p statistics.
    Writes interaction_regression.csv with columns:
        task, metric, interaction_coeff, t_stat, p_value, n_obs, interpretation
    """
    from scipy.stats import t as _t_dist  # type: ignore[import]

    _SCIPY_OK = False
    try:
        from scipy.stats import t as _t_dist  # noqa: F811
        _SCIPY_OK = True
    except Exception:
        pass

    # Approximate p-value from t-statistic via numpy if scipy unavailable
    def _p_from_t(t_val: float, df: int) -> float:
        """Two-tailed p-value approximation using normal distribution as fallback."""
        if _SCIPY_OK:
            return float(2.0 * _t_dist.sf(abs(t_val), df=max(df, 1)))
        # Normal approximation
        z = abs(t_val)
        # Abramowitz & Stegun 26.2.17 approximation
        p1 = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        # two-sided
        p_val = 2.0 * p1 * (1 + z * (0.31938 + z * (-0.35656 + z * (1.78148 + z * (-1.82138 + z * 1.33027)))))
        return max(0.0, min(1.0, p_val))

    task_metric_map = {
        "text_mt": "BLEU",
        "asr":     "WER",
    }

    rows: list[dict] = []

    for task, metric in task_metric_map.items():
        sub = scaling_df[scaling_df["task"] == task].copy()
        if sub.empty:
            print(f"[InteractionReg] No rows for task={task}, skipping.")
            continue

        # Attach family labels
        sub["family"] = sub["language_key"].map(lang_family_map).fillna("unknown")
        sub["log_budget"] = np.log(sub["budget"].astype(float) + 1.0)
        sub = sub.dropna(subset=[metric, "log_budget"])

        if sub.empty or sub["log_budget"].nunique() < 2:
            print(f"[InteractionReg] Insufficient data for task={task}.")
            continue

        # Encode family as dummy variables (drop-first)
        families = sorted(sub["family"].unique())
        ref_family = families[0]
        other_families = families[1:]

        n = len(sub)
        # Design matrix columns:
        #   [intercept, log_budget, fam_2, fam_3, ..., log_budget*fam_2, log_budget*fam_3, ...]
        n_dummy = len(other_families)
        n_cols = 1 + 1 + n_dummy + n_dummy   # intercept + log_budget + dummies + interactions

        X = np.zeros((n, n_cols), dtype=float)
        X[:, 0] = 1.0                                         # intercept
        X[:, 1] = sub["log_budget"].values                    # log_budget

        for j, fam in enumerate(other_families):
            indicator = (sub["family"] == fam).astype(float).values
            X[:, 2 + j] = indicator                           # family dummy
            X[:, 2 + n_dummy + j] = indicator * X[:, 1]      # interaction

        y = sub[metric].values.astype(float)

        # OLS: β = (X'X)^{-1} X'y
        try:
            XtX = X.T @ X
            Xty = X.T @ y
            beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        except Exception:
            print(f"[InteractionReg] OLS failed for task={task}:\n{traceback.format_exc()}")
            continue

        # Residuals and variance
        y_hat = X @ beta
        residuals = y - y_hat
        k = n_cols
        dof = max(n - k, 1)
        sigma2 = float(np.sum(residuals ** 2) / dof)

        # Covariance matrix of β
        try:
            XtX_inv = np.linalg.pinv(XtX)
            var_beta = sigma2 * XtX_inv
        except Exception:
            var_beta = np.full((k, k), float("nan"))

        # The interaction term(s) — report the first one (for the first non-ref family)
        # Column index for first interaction: 2 + n_dummy + 0
        if n_dummy > 0:
            interact_col = 2 + n_dummy
            interact_coeff = float(beta[interact_col])
            se = math.sqrt(max(float(var_beta[interact_col, interact_col]), 0.0))
            t_stat = interact_coeff / se if se > 1e-12 else float("nan")
            p_value = _p_from_t(t_stat, dof) if not math.isnan(t_stat) else float("nan")
        else:
            # Only one family — report log_budget coefficient as a fallback
            interact_coeff = float(beta[1])
            se = math.sqrt(max(float(var_beta[1, 1]), 0.0))
            t_stat = interact_coeff / se if se > 1e-12 else float("nan")
            p_value = _p_from_t(t_stat, dof) if not math.isnan(t_stat) else float("nan")

        interpretation = (
            "significant (p<0.05)" if not math.isnan(p_value) and p_value < 0.05
            else "not significant"
        )

        row = {
            "task":              task,
            "metric":            metric,
            "interaction_coeff": interact_coeff,
            "t_stat":            t_stat,
            "p_value":           p_value,
            "n_obs":             n,
            "interpretation":    interpretation,
        }
        rows.append(row)
        print(f"[InteractionReg] task={task}  metric={metric}: "
              f"coeff={interact_coeff:.4f}  t={t_stat:.3f}  p={p_value:.4f}  → {interpretation}")

    df = pd.DataFrame(rows, columns=[
        "task", "metric", "interaction_coeff", "t_stat", "p_value", "n_obs", "interpretation",
    ])

    if dirs is not None:
        out_path = dirs.metrics / "interaction_regression.csv"
        df.to_csv(out_path, index=False)
        print(f"[InteractionReg] Written: {out_path}")

    return df
