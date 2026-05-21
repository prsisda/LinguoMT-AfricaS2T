"""
ExperimentRunner — model-agnostic experiment orchestrator.

Callers supply callable translation functions; this module runs all applicable
strategies, computes metrics, saves outputs, and generates paper-ready artefacts.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd

from .audio import (
    AUDIO_STRATEGIES, MIN_DURATION_SEC, MAX_DURATION_SEC,
    extract_audio_array, apply_strategy, chunk_audio, audio_duration, compute_audio_quality,
)
from .capabilities import ModelCapabilities
from .dataset import DatasetCache, make_translation_rows
from .interpretations import generate_table_interpretation, generate_summary_interpretation, select_strategies_from_eda
from .languages import get_model_lang_code
from .metrics import compute_translation_metrics, compute_asr_metrics
from .monitoring import StepMonitor
from .output import OutputDirs, save_df
from .plots import (
    plot_duration_distribution, plot_silence_ratio, plot_energy_distribution,
    plot_waveform_comparison, plot_bleu_by_language_direction,
    plot_strategy_comparison, plot_direction_comparison,
    plot_experiment_progression, plot_asr_quality,
)
from .tables import (
    build_text_table, build_audio_table, build_strategy_pivot,
    build_direction_pivot, build_asr_table, build_summary_table, build_qualitative_table,
    build_cross_language_comparisons, build_cross_model_comparisons,
)
from .finetuning import build_finetune_comparison_table
from .sota import (
    load_sota, build_sota_comparison_table, build_gap_table,
    build_system_ranking_table, build_improvement_table,
    generate_sota_interpretation,
)
from .paper_modes import get_paper_mode_config

warnings.filterwarnings("ignore")


@dataclass
class RunConfig:
    model_name:        str
    dataset_name:      str
    experiment_family: str
    model_id:          str
    dataset_id:        str
    split:             str  = "dev"
    train_split:       str  = "train"
    debug_mode:        bool = True
    fast_mode:         bool = False
    skip_audio_debug:  bool = True
    force_rerun:       bool = False
    run_full_grid:     bool = True
    in_colab:          bool = False
    eda_sample_size:   int  = 25
    enable_finetuning: bool = False   # disabled until baseline is stable
    paper_mode:        str  = "benchmark"  # benchmark | adaptation | audio | cascade | transfer
    sota_path:         str  = ""           # path to sota_results.csv or published_baselines.json
    directions:        list[str] = field(default_factory=lambda: ["source_to_english", "english_to_source"])
    scaling_budgets:   list[int] = field(default_factory=list)  # Paper 2: data scaling budgets
    extra:             dict = field(default_factory=dict)

    @property
    def run_audio(self) -> bool:
        return not (self.debug_mode and self.skip_audio_debug)


def default_experiment_configs(
    debug_mode: bool,
    n_runs: int = 3,
    text_samples: list | None = None,
    audio_samples: list | None = None,
) -> list[dict]:
    """Return the list of experiment grid configs.

    Args:
        debug_mode:    When True returns a single tiny config for pipeline checks.
        n_runs:        How many evaluation passes to run (1, 2, or 3).
                       Each pass evaluates on a progressively larger sample.
        text_samples:  Text-pair counts per pass, e.g. [100, 200, 300].
                       Defaults to [100, 200, 300]. Sliced to n_runs entries.
        audio_samples: Audio-pair counts per pass, e.g. [30, 75, 100].
                       Defaults to [30, 75, 100]. Sliced to n_runs entries.
    """
    if debug_mode:
        return [{"experiment": "debug", "max_text_train": 0, "max_text_dev": 8, "max_audio_dev": 3}]

    n_runs       = max(1, min(3, int(n_runs)))
    default_text  = [100, 200, 300]
    default_audio = [ 30,  75, 100]
    text_s  = (text_samples  if text_samples  is not None else default_text) [:n_runs]
    audio_s = (audio_samples if audio_samples is not None else default_audio)[:n_runs]

    train_bases = [500, 600, 700]
    return [
        {
            "experiment":    f"Experiment_{i + 1}",
            "max_text_train": train_bases[i],
            "max_text_dev":   text_s[i],
            "max_audio_dev":  audio_s[i],
        }
        for i in range(n_runs)
    ]


class ExperimentRunner:
    """
    Model-agnostic experiment orchestrator.

    Required callables (pass None for unsupported operations):
      translate_text_fn(text, src_lang, tgt_lang) -> str
      translate_audio_fn(audio_arr, src_lang, tgt_lang) -> (str, int_num_chunks)
      asr_fn(audio_arr, src_lang) -> str
    """

    def __init__(
        self,
        config: RunConfig,
        capabilities: ModelCapabilities,
        data_cache: DatasetCache,
        language_configs: list[dict],
        dirs: OutputDirs,
        monitor: StepMonitor,
        experiment_configs: list[dict],
        translate_text_fn:  Callable | None = None,
        translate_audio_fn: Callable | None = None,
        asr_fn:             Callable | None = None,
        train_cache:        DatasetCache | None = None,
        finetune_fn:        Callable | None = None,
    ) -> None:
        self.cfg          = config
        self.caps         = capabilities
        self.cache        = data_cache
        self.train_cache  = train_cache
        self.lang_cfgs    = language_configs
        self.dirs         = dirs
        self.monitor      = monitor
        self.exp_cfgs     = experiment_configs
        self._t_text      = translate_text_fn
        self._t_audio     = translate_audio_fn
        self._asr         = asr_fn
        self._finetune    = finetune_fn

        # Accumulate results across strategies
        self._text_results:   list[dict] = []
        self._text_preds:     list[dict] = []
        self._audio_results:  list[dict] = []
        self._audio_preds:    list[dict] = []
        self._asr_results:    list[dict] = []
        self._asr_preds:      list[dict] = []
        self._eda_rows:       list[dict] = []
        self._eda_compact:    pd.DataFrame = pd.DataFrame()
        self._strategy_rationale: dict = {}
        # Before/after fine-tuning snapshots
        self._pretrain_text_results: list[dict] = []
        self._pretrain_asr_results:  list[dict] = []

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        if not self.cfg.run_full_grid and len(self.exp_cfgs) > 1:
            self.exp_cfgs = self.exp_cfgs[:1]
        mode_info = get_paper_mode_config(self.cfg.paper_mode)
        self.monitor.step("Run started", f"{self.cfg.experiment_family} | mode={self.cfg.paper_mode}")
        print(f"[LinguoMT] Paper mode: {self.cfg.paper_mode!r} — {mode_info.title}")

        # Baseline evaluation before fine-tuning (for before/after comparison)
        if self.cfg.enable_finetuning and self._finetune is not None and self.train_cache is not None:
            self.monitor.step("Baseline eval (pre-FT)")
            if self.caps.t2tt and self._t_text is not None:
                self._run_text_experiments()
                self._pretrain_text_results = list(self._text_results)
                self._text_results.clear(); self._text_preds.clear()
            if self.caps.asr and self._asr is not None:
                self._run_asr_experiments()
                self._pretrain_asr_results = list(self._asr_results)
                self._asr_results.clear(); self._asr_preds.clear()
            # Reset resume cache so post-FT eval runs fresh
            _clear_eval_cache(self.dirs)

        self._run_finetuning()
        self._run_eda()
        self._strategy_rationale = select_strategies_from_eda(self._eda_compact)
        if self.caps.t2tt and self._t_text is not None:
            self._run_text_experiments()
        if self.cfg.run_audio and self.caps.supports_audio_input and self._t_audio is not None:
            self._run_audio_experiments()
        if self.caps.asr and self._asr is not None:
            self._run_asr_experiments()
        self._run_paper_mode_extras()
        self._build_all_outputs()
        self.monitor.step("Run complete")
        print(f"\nOutputs → {self.dirs.base}")

    # ── fine-tuning ────────────────────────────────────────────────────────────

    def _run_finetuning(self) -> None:
        if not self.cfg.enable_finetuning:
            self.monitor.step("Fine-tuning", "disabled — using pretrained model")
            return
        if self._finetune is None or self.train_cache is None:
            self.monitor.step("Fine-tuning skipped", "finetune_fn or train_cache not provided")
            return
        self.monitor.step("Fine-tuning started")
        for exp in self.exp_cfgs:
            n = exp.get("max_text_train", 0)
            if n == 0:
                self.monitor.step(f"  [{exp['experiment']}] fine-tuning skipped", "max_text_train=0")
                continue
            for lang_cfg in self.lang_cfgs:
                lk    = lang_cfg["language_key"]
                pairs = self.train_cache.get_pairs(lk, n)
                if not pairs:
                    self.monitor.step(f"  [{lang_cfg['display']}] no train pairs", "skipping")
                    continue
                try:
                    # _finetune may be a FineTuner.finetune method or a plain callable
                    self._finetune(pairs, lang_cfg, exp)
                    self.monitor.step(f"  [{lang_cfg['display']}] fine-tuned", f"{len(pairs)} pairs")
                except Exception as e:
                    self.monitor.step(f"  [{lang_cfg['display']}] fine-tune error", str(e)[:120])
        self.monitor.step("Fine-tuning complete")

    # ── paper-mode extras ─────────────────────────────────────────────────────

    def _run_paper_mode_extras(self) -> None:
        """Run additional analyses triggered by PAPER_MODE and optional config."""
        mode = self.cfg.paper_mode

        # Paper 2 — data scaling experiment
        if (
            mode == "adaptation"
            and self.cfg.scaling_budgets
            and self._finetune is not None
            and self.train_cache is not None
        ):
            from .scaling import run_scaling_experiment
            self.monitor.step("Scaling experiment (paper2)")
            try:
                run_scaling_experiment(
                    budgets=self.cfg.scaling_budgets,
                    train_cache=self.train_cache,
                    dev_cache=self.cache,
                    lang_cfgs=self.lang_cfgs,
                    finetune_fn=self._finetune,
                    translate_text_fn=self._t_text,
                    asr_fn=self._asr,
                    caps=self.caps,
                    dirs=self.dirs,
                    monitor=self.monitor,
                )
            except Exception as e:
                self.monitor.step("Scaling experiment error", str(e)[:120])

        # Paper 4 — oracle cascade (text MT ceiling on gold transcripts)
        if mode == "cascade" and self._t_text is not None:
            from .cascade_analysis import run_oracle_cascade
            self.monitor.step("Oracle cascade (paper4)")
            try:
                run_oracle_cascade(
                    translate_text_fn=self._t_text,
                    dev_cache=self.cache,
                    lang_cfgs=self.lang_cfgs,
                    caps=self.caps,
                    dirs=self.dirs,
                    monitor=self.monitor,
                )
            except Exception as e:
                self.monitor.step("Oracle cascade error", str(e)[:120])

    # ── EDA ───────────────────────────────────────────────────────────────────

    def _run_eda(self) -> None:
        self.monitor.step("EDA started")
        for cfg in self.lang_cfgs:
            lk = cfg["language_key"]
            pairs = self.cache.get_pairs(lk, self.cfg.eda_sample_size)
            lang_dir = self.dirs.eda / lk
            lang_dir.mkdir(exist_ok=True)
            for p in pairs:
                if p.get("src_audio") is None:
                    continue
                try:
                    q = compute_audio_quality(p["src_audio"])
                    self._eda_rows.append({
                        "language": cfg["display"], "language_key": lk,
                        "src_text": p["src_text"], "eng_text": p["eng_text"],
                        **q,
                    })
                except Exception:
                    pass
            # Waveform plot
            if pairs and pairs[0].get("src_audio"):
                try:
                    from .audio import extract_audio_array, TARGET_SR
                    arr = extract_audio_array(pairs[0]["src_audio"])
                    plot_waveform_comparison(arr, cfg["display"], TARGET_SR, lang_dir, self.cfg.in_colab)
                except Exception:
                    pass

        eda_df = pd.DataFrame(self._eda_rows)
        if not eda_df.empty:
            save_df(eda_df, "eda_all.csv", self.dirs.eda)
            plot_duration_distribution(eda_df, self.dirs.eda, self.cfg.in_colab)
            plot_silence_ratio(eda_df,   self.dirs.eda, self.cfg.in_colab)
            plot_energy_distribution(eda_df, self.dirs.eda, self.cfg.in_colab)
            self._eda_compact = eda_df.groupby("language").agg(
                duration_sec=("duration_sec", "mean"),
                silence_ratio=("silence_ratio", "mean"),
                rms_energy=("rms_energy", "mean"),
            ).reset_index()
            save_df(self._eda_compact, "eda_compact.csv", self.dirs.eda)
        self.monitor.step("EDA complete", f"{len(self._eda_rows)} rows")

    # ── text evaluation ───────────────────────────────────────────────────────

    def _run_text_experiments(self) -> None:
        pred_path   = self.dirs.predictions / "text_predictions.csv"
        metric_path = self.dirs.metrics     / "text_metrics.csv"
        if not self.cfg.force_rerun and pred_path.exists() and metric_path.exists():
            self.monitor.step("Text eval — resuming from cache", pred_path.name)
            self._text_preds   = pd.read_csv(pred_path).fillna("").to_dict("records")
            self._text_results = pd.read_csv(metric_path).to_dict("records")
            return
        self.monitor.step("Text evaluation started")
        for exp in self.exp_cfgs:
            exp_name = exp["experiment"]
            for cfg in self.lang_cfgs:
                lk          = cfg["language_key"]
                model_code  = get_model_lang_code(cfg, self.caps)
                if not model_code:
                    continue
                pairs = self.cache.get_pairs(lk, exp["max_text_dev"])
                rows  = make_translation_rows(
                    cfg, pairs, self.cfg.directions, model_code, self.caps.english_code
                )
                for dir_key, group in _group_by(rows, "direction").items():
                    preds, refs = [], []
                    t0 = time.time()
                    for r in group:
                        try:
                            p = self._t_text(r["source_text"], r["source_lang"], r["target_lang"])
                        except Exception as e:
                            p = ""
                            print(f"  text error [{cfg['display']}]: {e}")
                        preds.append(p); refs.append(r["target_text"])
                        self._text_preds.append({
                            **_base_row(self.cfg, exp_name, cfg, r),
                            "mode": "text", "source_text": r["source_text"],
                            "reference": r["target_text"], "prediction": p,
                        })
                    m = compute_translation_metrics(preds, refs)
                    self._text_results.append({
                        **_base_meta(self.cfg, exp_name, cfg, r),
                        "mode": "text", **m, "runtime_s": time.time() - t0,
                    })
                    print(f"  [{exp_name}] {cfg['display']} | {r['direction_label']} | BLEU={m['BLEU']:.2f} ChrF={m['ChrF']:.2f}")
        self.monitor.step("Text evaluation complete", f"{len(self._text_results)} rows")

    # ── audio evaluation ──────────────────────────────────────────────────────

    def _run_audio_experiments(self) -> None:
        pred_path   = self.dirs.predictions / "audio_predictions.csv"
        metric_path = self.dirs.metrics     / "audio_metrics.csv"
        if not self.cfg.force_rerun and pred_path.exists() and metric_path.exists():
            self.monitor.step("Audio eval — resuming from cache", pred_path.name)
            self._audio_preds   = pd.read_csv(pred_path).fillna("").to_dict("records")
            self._audio_results = pd.read_csv(metric_path).to_dict("records")
            return
        self.monitor.step("Audio evaluation started")
        for exp in self.exp_cfgs:
            exp_name   = exp["experiment"]
            strategies = [s for s in AUDIO_STRATEGIES if self._audio_strategy_enabled(s, exp_name)]
            for cfg in self.lang_cfgs:
                lk         = cfg["language_key"]
                model_code = get_model_lang_code(cfg, self.caps)
                if not model_code:
                    continue
                pairs = self.cache.get_pairs(lk, exp["max_audio_dev"])
                rows  = make_translation_rows(
                    cfg, pairs, self.cfg.directions, model_code, self.caps.english_code
                )
                for strat in strategies:
                    for dir_key, group in _group_by(rows, "direction").items():
                        preds, refs, durs, nchunks = [], [], [], []
                        t0 = time.time()
                        for r in group:
                            arr = extract_audio_array(r["source_audio"])
                            dur = audio_duration(arr); durs.append(dur)
                            if dur < MIN_DURATION_SEC or dur > MAX_DURATION_SEC:
                                pred, nc, err = "", 0, f"skipped_duration_{dur:.1f}s"
                            else:
                                err = ""
                                try:
                                    if strat["chunk"]:
                                        chunks = chunk_audio(apply_strategy(r["source_audio"], strat))
                                        parts  = []
                                        for ch in chunks:
                                            pt = self._t_audio(ch, r["source_lang"], r["target_lang"])
                                            if isinstance(pt, tuple): pt = pt[0]
                                            if pt.strip(): parts.append(pt.strip())
                                        pred, nc = " ".join(parts), len(chunks)
                                    else:
                                        pred = self._t_audio(apply_strategy(r["source_audio"], strat), r["source_lang"], r["target_lang"])
                                        if isinstance(pred, tuple): pred, nc = pred
                                        else: nc = 1
                                except Exception as e:
                                    pred, nc, err = "", 0, str(e)
                            preds.append(pred); refs.append(r["target_text"])
                            nchunks.append(nc)
                            self._audio_preds.append({
                                **_base_row(self.cfg, exp_name, cfg, r),
                                "strategy_key":   strat["key"],
                                "strategy_label": strat["label"],
                                "source_text":    r["source_text"],
                                "reference":      r["target_text"],
                                "prediction":     pred,
                                "duration_s":     dur,
                                "num_chunks":     nc,
                                "error_message":  err,
                            })
                        m = compute_translation_metrics(preds, refs)
                        self._audio_results.append({
                            **_base_meta(self.cfg, exp_name, cfg, r),
                            "strategy_key":    strat["key"],
                            "strategy_label":  strat["label"],
                            **m,
                            "avg_duration":    float(np.mean(durs)) if durs else 0.0,
                            "avg_chunks":      float(np.mean(nchunks)) if nchunks else 0.0,
                            "runtime_s":       time.time() - t0,
                        })
                        print(f"  [{exp_name}] {cfg['display']} | {strat['key']} | {r['direction_label']} | BLEU={m['BLEU']:.2f}")
        self.monitor.step("Audio evaluation complete", f"{len(self._audio_results)} rows")

    # ── ASR evaluation ────────────────────────────────────────────────────────

    def _run_asr_experiments(self) -> None:
        if self._asr is None:
            return
        pred_path   = self.dirs.predictions / "asr_predictions.csv"
        metric_path = self.dirs.metrics     / "asr_metrics.csv"
        if not self.cfg.force_rerun and pred_path.exists() and metric_path.exists():
            self.monitor.step("ASR eval — resuming from cache", pred_path.name)
            self._asr_preds   = pd.read_csv(pred_path).fillna("").to_dict("records")
            self._asr_results = pd.read_csv(metric_path).to_dict("records")
            return
        self.monitor.step("ASR evaluation started")
        for exp in self.exp_cfgs:
            exp_name = exp["experiment"]
            for cfg in self.lang_cfgs:
                lk         = cfg["language_key"]
                asr_attr   = self.caps.asr_lang_attr
                model_code = (cfg.get(asr_attr) if asr_attr else None) or get_model_lang_code(cfg, self.caps)
                if not model_code:
                    continue
                pairs = self.cache.get_pairs(lk, exp["max_audio_dev"])
                if not pairs:
                    continue
                preds, refs = [], []
                t0 = time.time()
                for p in pairs:
                    if p.get("src_audio") is None:
                        preds.append(""); refs.append(p["src_text"]); continue
                    try:
                        pred = self._asr(extract_audio_array(p["src_audio"]), model_code)
                    except Exception as e:
                        pred = ""; print(f"  ASR error [{cfg['display']}]: {e}")
                    preds.append(pred); refs.append(p["src_text"])
                    self._asr_preds.append({
                        "experiment": exp_name, "language": cfg["display"],
                        "reference": p["src_text"], "prediction": pred,
                    })
                m = compute_asr_metrics(preds, refs)
                self._asr_results.append({
                    "experiment": exp_name, "language": cfg["display"],
                    **m, "runtime_s": time.time() - t0,
                })
                print(f"  [{exp_name}] {cfg['display']} | WER={m.get('WER',float('nan')):.3f}")
        self.monitor.step("ASR evaluation complete", f"{len(self._asr_results)} rows")

    # ── output generation ─────────────────────────────────────────────────────

    def _build_all_outputs(self) -> None:
        self.monitor.step("Building outputs")
        text_df  = pd.DataFrame(self._text_results)
        text_pd  = pd.DataFrame(self._text_preds)
        audio_df = pd.DataFrame(self._audio_results)
        audio_pd = pd.DataFrame(self._audio_preds)
        asr_df   = pd.DataFrame(self._asr_results)
        asr_pd   = pd.DataFrame(self._asr_preds)
        _non_empty = [df for df in [text_df, audio_df] if not df.empty]
        aggregate  = pd.concat(_non_empty, ignore_index=True) if _non_empty else pd.DataFrame()

        # Save raw data
        for df, name in [(text_df, "text_metrics.csv"), (text_pd, "text_predictions.csv"),
                         (audio_df, "audio_metrics.csv"), (audio_pd, "audio_predictions.csv"),
                         (asr_df,   "asr_metrics.csv"),   (asr_pd,   "asr_predictions.csv"),
                         (aggregate, "aggregate_metrics.csv")]:
            if not df.empty:
                save_df(df, name, self.dirs.metrics if "metrics" in name else self.dirs.predictions)

        # Paper tables
        T1 = build_text_table(text_df)
        T2 = build_audio_table(audio_df)
        T3 = build_strategy_pivot(audio_df)
        T4 = build_direction_pivot(aggregate)
        T5 = build_asr_table(asr_df)
        T6 = build_summary_table(
            text_df, audio_df, audio_pd, asr_df,
            self.exp_cfgs, self.lang_cfgs,
            self.cfg.model_name, self.cfg.dataset_name,
            self.cfg.eda_sample_size,
            compute_translation_metrics,
        )
        _non_empty_pd = [p for p in [text_pd, audio_pd] if not p.empty]
        T7 = build_qualitative_table(pd.concat(_non_empty_pd, ignore_index=True) if _non_empty_pd else pd.DataFrame())
        T8 = build_cross_language_comparisons(text_df, audio_df)
        T9 = build_cross_model_comparisons(self.dirs.base.parent)

        for df, name in [(T1, "T1_text_evaluation.csv"),    (T2, "T2_audio_evaluation.csv"),
                         (T3, "T3_strategy_pivot.csv"),     (T4, "T4_direction_pivot.csv"),
                         (T5, "T5_asr_evaluation.csv"),     (T6, "T6_summary_table.csv"),
                         (T7, "T7_qualitative.csv"),        (T8, "T8_cross_language.csv"),
                         (T9, "T9_cross_model.csv")]:
            if not df.empty:
                save_df(df, name, self.dirs.tables)
        save_df(T6, "summary_table.csv", self.dirs.metrics)

        # Fine-tuning before/after comparison tables
        if self.cfg.enable_finetuning:
            T_FT1 = build_finetune_comparison_table(
                self._pretrain_text_results, self._text_results, task="text"
            )
            T_FT2 = build_finetune_comparison_table(
                self._pretrain_asr_results, self._asr_results, task="asr"
            )
            for df, name in [(T_FT1, "T_FT1_text_finetune_comparison.csv"),
                             (T_FT2, "T_FT2_asr_finetune_comparison.csv")]:
                if not df.empty:
                    save_df(df, name, self.dirs.tables)
                    print(df.to_string(index=False))

        # SOTA comparison tables (paper-mode-driven)
        mode_cfg  = get_paper_mode_config(self.cfg.paper_mode)
        sota_df   = load_sota(self.cfg.sota_path) if self.cfg.sota_path else pd.DataFrame()
        _sota_available = not sota_df.empty

        if _sota_available and mode_cfg.include_sota_comparison:
            T_SOTA1 = build_sota_comparison_table(
                text_df, sota_df, metric=mode_cfg.primary_metric,
                dataset_filter=self.cfg.dataset_name,
            )
            if not T_SOTA1.empty:
                save_df(T_SOTA1, "T_SOTA1_vs_published.csv", self.dirs.tables)

        if _sota_available and mode_cfg.include_gap_table:
            T_SOTA2 = build_gap_table(text_df, sota_df, metric=mode_cfg.primary_metric)
            if not T_SOTA2.empty:
                save_df(T_SOTA2, "T_SOTA2_language_gap.csv", self.dirs.tables)

        if _sota_available and mode_cfg.include_system_ranking:
            T_SOTA3 = build_system_ranking_table(
                self.cfg.model_name, text_df, sota_df, metric=mode_cfg.primary_metric
            )
            if not T_SOTA3.empty:
                save_df(T_SOTA3, "T_SOTA3_system_ranking.csv", self.dirs.tables)

        if mode_cfg.include_improvement_table and self._pretrain_text_results:
            T_SOTA4 = build_improvement_table(
                pd.DataFrame(self._pretrain_text_results), text_df,
                metric=mode_cfg.primary_metric,
            )
            if not T_SOTA4.empty:
                save_df(T_SOTA4, "T_SOTA4_improvement.csv", self.dirs.tables)

        if _sota_available and mode_cfg.include_sota_comparison:
            _sota_cmp   = T_SOTA1 if "T_SOTA1" in dir() else pd.DataFrame()
            _gap_table  = T_SOTA2 if "T_SOTA2" in dir() else pd.DataFrame()
            sota_interp = generate_sota_interpretation(
                _sota_cmp, _gap_table, self.cfg.model_name, mode_cfg.primary_metric
            )
            (self.dirs.interpretations / "I_SOTA.md").write_text(sota_interp, encoding="utf-8")

        # Write paper-mode header to summary directory
        _mode_header = (
            f"# {mode_cfg.title}\n\n"
            f"**Paper mode:** {mode_cfg.mode}  |  "
            f"**Model:** {self.cfg.model_name}  |  "
            f"**Dataset:** {self.cfg.dataset_name}\n\n"
            f"{mode_cfg.description}\n"
        )
        (self.dirs.summaries / "paper_mode.md").write_text(_mode_header, encoding="utf-8")

        # Plots
        plot_bleu_by_language_direction(text_df, self.dirs.plots, in_colab=self.cfg.in_colab)
        plot_strategy_comparison(audio_df, self.dirs.plots, in_colab=self.cfg.in_colab)
        plot_direction_comparison(aggregate, self.dirs.plots, in_colab=self.cfg.in_colab)
        plot_experiment_progression(aggregate, self.dirs.plots, in_colab=self.cfg.in_colab)
        plot_asr_quality(asr_df, self.dirs.plots, in_colab=self.cfg.in_colab)

        # Interpretations
        for df, title, fname in [
            (T1, "T1: Text Translation Evaluation",    "I1_text.md"),
            (T2, "T2: Audio Strategy Evaluation",      "I2_audio.md"),
            (T3, "T3: Strategy Comparison",            "I3_strategy_pivot.md"),
            (T4, "T4: Direction Comparison",           "I4_direction_pivot.md"),
            (T5, "T5: ASR Evaluation",                 "I5_asr.md"),
        ]:
            if not df.empty:
                (self.dirs.interpretations / fname).write_text(
                    generate_table_interpretation(df, title), encoding="utf-8"
                )

        full_summary = generate_summary_interpretation(
            T6, text_df, audio_df, asr_df,
            self.cfg.model_name, self.cfg.dataset_name,
            self.lang_cfgs, self.caps,
            self._strategy_rationale, self.cfg.debug_mode,
        )
        (self.dirs.summaries     / "experiment_summary.txt").write_text(full_summary, encoding="utf-8")
        (self.dirs.interpretations / "I0_full_summary.md").write_text(full_summary, encoding="utf-8")
        print(full_summary)

        self.monitor.step("Outputs saved")
        self.monitor.save()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _audio_strategy_enabled(self, strategy: dict, exp_name: str) -> bool:
        return strategy.get("key") in self.caps.enabled_strategies


def _clear_eval_cache(dirs: "OutputDirs") -> None:
    """Remove cached metric/prediction CSVs so post-FT eval runs from scratch."""
    for name in ["text_metrics.csv", "text_predictions.csv",
                 "asr_metrics.csv", "asr_predictions.csv",
                 "audio_metrics.csv", "audio_predictions.csv"]:
        for parent in [dirs.metrics, dirs.predictions]:
            p = parent / name
            if p.exists():
                p.unlink()


# ── module-level helpers ──────────────────────────────────────────────────────

def _group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def _base_row(cfg: RunConfig, exp_name: str, lang_cfg: dict, row: dict) -> dict:
    return {
        "experiment":       exp_name,
        "language":         lang_cfg["display"],
        "language_key":     lang_cfg["language_key"],
        "direction":        row["direction"],
        "direction_label":  row["direction_label"],
        "pair_idx":         row["pair_idx"],
        "model_name":       cfg.model_name,
        "dataset_name":     cfg.dataset_name,
        "experiment_family": cfg.experiment_family,
    }


def _base_meta(cfg: RunConfig, exp_name: str, lang_cfg: dict, row: dict) -> dict:
    return {
        "experiment":       exp_name,
        "language":         lang_cfg["display"],
        "language_key":     lang_cfg["language_key"],
        "direction":        row["direction"],
        "direction_label":  row["direction_label"],
        "model_name":       cfg.model_name,
        "dataset_name":     cfg.dataset_name,
        "experiment_family": cfg.experiment_family,
    }
