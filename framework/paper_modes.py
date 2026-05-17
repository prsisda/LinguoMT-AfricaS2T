"""
Paper mode definitions for the LinguoMT journal publication series.

Each mode reuses the same core evaluation pipeline but controls which tables,
plots, and interpretations are emphasised in the output.

Usage in run scripts:
    PAPER_MODE = "benchmark"   # LinguoMT
    PAPER_MODE = "adaptation"  # LinguoMT-Adapt
    PAPER_MODE = "audio"       # LinguoMT-Audio
    PAPER_MODE = "cascade"     # LinguoMT-Cascade
    PAPER_MODE = "transfer"    # LinguoMT-Transfer
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PaperModeConfig:
    mode: str
    title: str
    description: str
    primary_metric: str = "BLEU"

    # Core tables always generated — these flags ADD paper-specific extras
    include_sota_comparison:   bool = True   # T_SOTA1: our vs published SOTA
    include_gap_table:         bool = True   # T_SOTA2: language SOTA gap
    include_system_ranking:    bool = False  # T_SOTA3: all systems ranked
    include_improvement_table: bool = False  # T_SOTA4: before/after improvement

    # Which core tables to feature in the paper-mode summary
    featured_tables: list[str] = field(default_factory=list)

    # Extra plot types to generate
    extra_plots: list[str] = field(default_factory=list)

    # Interpretation focus keywords (used in summary text headings)
    interpretation_focus: list[str] = field(default_factory=list)

    # Which fine-tuning tables to include in the output section
    include_finetuning_tables: bool = False

    # Cascade-specific: include direct vs cascade comparison
    include_cascade_analysis:  bool = False


PAPER_MODES: dict[str, PaperModeConfig] = {
    "benchmark": PaperModeConfig(
        mode="benchmark",
        title="LinguoMT: Benchmarking Multilingual Speech Translation for Low-Resource African Languages",
        description=(
            "Systematic pretrained-model evaluation across African languages, datasets, "
            "directions, and strategies. Primary goal: establish baselines and compare "
            "against published SOTA."
        ),
        primary_metric="BLEU",
        include_sota_comparison=True,
        include_gap_table=True,
        include_system_ranking=True,
        include_improvement_table=False,
        featured_tables=["T1_text", "T5_asr", "T8_cross_language", "T_SOTA1", "T_SOTA2"],
        extra_plots=["bleu_by_language_direction", "experiment_progression"],
        interpretation_focus=["baseline performance", "cross-language comparison", "SOTA gap"],
        include_finetuning_tables=False,
        include_cascade_analysis=False,
    ),
    "adaptation": PaperModeConfig(
        mode="adaptation",
        title="LinguoMT-Adapt: Parameter-Efficient Fine-Tuning for African Speech Translation",
        description=(
            "LoRA fine-tuning experiments. Primary goal: show how much improvement "
            "fine-tuning brings over pretrained baselines and compare against adapted SOTA."
        ),
        primary_metric="BLEU",
        include_sota_comparison=True,
        include_gap_table=True,
        include_system_ranking=False,
        include_improvement_table=True,
        featured_tables=["T1_text", "T_FT1", "T_FT2", "T_SOTA1", "T_SOTA4"],
        extra_plots=["bleu_by_language_direction", "direction_comparison"],
        interpretation_focus=["fine-tuning improvement", "before/after comparison", "adaptation efficiency"],
        include_finetuning_tables=True,
        include_cascade_analysis=False,
    ),
    "audio": PaperModeConfig(
        mode="audio",
        title="LinguoMT-Audio: Audio Preprocessing and Robustness for African Speech Translation",
        description=(
            "Audio strategy experiments. Primary goal: measure the impact of preprocessing "
            "strategies (normalization, trimming, chunking) on translation quality."
        ),
        primary_metric="BLEU",
        include_sota_comparison=False,
        include_gap_table=False,
        include_system_ranking=False,
        include_improvement_table=False,
        featured_tables=["T2_audio", "T3_strategy_pivot", "T4_direction_pivot", "T5_asr"],
        extra_plots=["strategy_comparison", "duration_distribution", "silence_ratio", "asr_quality"],
        interpretation_focus=["audio strategy comparison", "robustness", "speech quality"],
        include_finetuning_tables=False,
        include_cascade_analysis=False,
    ),
    "cascade": PaperModeConfig(
        mode="cascade",
        title="LinguoMT-Cascade: Cascade vs End-to-End Architectures for African Speech Translation",
        description=(
            "Architecture comparison. Primary goal: analyse when cascade (ASR+MT) "
            "outperforms end-to-end and compare both against published systems."
        ),
        primary_metric="BLEU",
        include_sota_comparison=True,
        include_gap_table=True,
        include_system_ranking=True,
        include_improvement_table=False,
        featured_tables=["T1_text", "T2_audio", "T3_strategy_pivot", "T9_cross_model", "T_SOTA3"],
        extra_plots=["strategy_comparison", "direction_comparison", "experiment_progression"],
        interpretation_focus=["cascade vs end-to-end", "architecture tradeoffs", "error propagation"],
        include_finetuning_tables=False,
        include_cascade_analysis=True,
    ),
    "transfer": PaperModeConfig(
        mode="transfer",
        title="LinguoMT-Transfer: Cross-Lingual Adaptation in African Speech Translation",
        description=(
            "Cross-lingual transfer experiments. Primary goal: evaluate how well "
            "multilingual models transfer to unseen or low-resource African languages."
        ),
        primary_metric="BLEU",
        include_sota_comparison=True,
        include_gap_table=True,
        include_system_ranking=False,
        include_improvement_table=True,
        featured_tables=["T1_text", "T8_cross_language", "T_SOTA2", "T_SOTA4"],
        extra_plots=["bleu_by_language_direction", "direction_comparison"],
        interpretation_focus=["cross-lingual transfer", "language similarity", "zero-shot performance"],
        include_finetuning_tables=False,
        include_cascade_analysis=False,
    ),
}


def get_paper_mode_config(mode: str) -> PaperModeConfig:
    """Return config for the given mode; fall back to 'benchmark' if unknown."""
    return PAPER_MODES.get(mode, PAPER_MODES["benchmark"])


def describe_paper_modes() -> str:
    """Print a human-readable description of all modes."""
    lines = ["LinguoMT Paper Modes", "=" * 40]
    for key, cfg in PAPER_MODES.items():
        lines += [f"\n  {key!r}  —  {cfg.title}", f"  {cfg.description}"]
    return "\n".join(lines)
