from __future__ import annotations

"""
Data scaling experiment for Paper 2 (LinguoMT-Adapt).

Runs fine-tuning at multiple data budgets sequentially, evaluating after each.
Writes scaling_metrics.csv to dirs.metrics/ with columns:
  budget, language, language_key, task, BLEU, ChrF, WER, CER
"""

import traceback
from typing import Callable, Any

import pandas as pd

from .metrics import compute_translation_metrics, compute_asr_metrics
from .dataset import make_translation_rows
from .languages import get_model_lang_code
from .audio import extract_audio_array


def run_scaling_experiment(
    budgets: list[int],
    train_cache,
    dev_cache,
    lang_cfgs: list[dict],
    finetune_fn: Callable,        # callable(pairs, lang_cfg, exp_dict)
    translate_text_fn: Callable | None,  # callable(text, src_lang, tgt_lang) -> str, or None
    asr_fn: Callable | None,             # callable(audio_arr, src_lang) -> str, or None
    caps,                         # ModelCapabilities
    dirs,                         # OutputDirs
    monitor,                      # StepMonitor
) -> pd.DataFrame:
    """Run fine-tuning at multiple data budgets and evaluate after each.

    Returns a DataFrame with columns:
        budget, language, language_key, task, BLEU, ChrF, WER, CER
    """
    rows: list[dict] = []

    english_code = caps.english_code

    for budget in budgets:
        print(f"\n[Scaling] ===== Budget: {budget} =====")
        if monitor:
            monitor.step(f"Scaling budget={budget}", f"{len(lang_cfgs)} languages")

        # --- Fine-tune all languages at this budget ---
        for lang_cfg in lang_cfgs:
            lk = lang_cfg["language_key"]
            display = lang_cfg.get("display", lk)
            try:
                train_pairs = train_cache.get_pairs(lk, budget)
                if not train_pairs:
                    print(f"  [Scaling] {display}: no training pairs available, skipping.")
                    continue
                print(f"  [Scaling] Fine-tuning {display} on {len(train_pairs)} pairs …")
                if monitor:
                    monitor.step(f"  Finetune {display}", f"budget={budget}, n={len(train_pairs)}")
                finetune_fn(
                    train_pairs,
                    lang_cfg,
                    {"experiment": f"scaling_{budget}", "max_text_train": budget},
                )
            except Exception:
                print(f"  [Scaling] ERROR fine-tuning {display}:\n{traceback.format_exc()}")
                if monitor:
                    monitor.step(f"  Finetune ERROR {display}", traceback.format_exc(limit=3))

        # --- Evaluate all languages at this budget ---
        for lang_cfg in lang_cfgs:
            lk = lang_cfg["language_key"]
            display = lang_cfg.get("display", lk)

            model_lang_code = get_model_lang_code(lang_cfg, caps)

            # ---- Text MT evaluation ----
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
                        print(f"  [Scaling] {display} text_mt  budget={budget}: "
                              f"BLEU={mt_metrics.get('BLEU', 0):.2f}  ChrF={mt_metrics.get('ChrF', 0):.2f}")
                except Exception:
                    print(f"  [Scaling] ERROR text_mt eval {display}:\n{traceback.format_exc()}")

            # ---- ASR evaluation ----
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
                        print(f"  [Scaling] {display} asr      budget={budget}: "
                              f"WER={asr_metrics.get('WER', float('nan')):.3f}  "
                              f"CER={asr_metrics.get('CER', float('nan')):.3f}")
                except Exception:
                    print(f"  [Scaling] ERROR asr eval {display}:\n{traceback.format_exc()}")

        if monitor:
            monitor.step(f"Scaling budget={budget} complete", f"{len(rows)} rows so far")

    df = pd.DataFrame(rows, columns=["budget", "language", "language_key", "task",
                                     "BLEU", "ChrF", "WER", "CER"])

    if dirs is not None:
        out_path = dirs.metrics / "scaling_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"[Scaling] Written: {out_path}")
        if monitor:
            monitor.step("Scaling CSV written", str(out_path))

    return df
