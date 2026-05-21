from .environment import detect_environment, install_colab_dependencies, mount_google_drive
from .capabilities import detect_model_capabilities, ModelCapabilities
from .languages import select_supported_languages, AFRICAN_LANGUAGES, get_adapter_type
from .monitoring import StepMonitor
from .output import create_run_dirs, save_config, zip_run_outputs, drive_backup, colab_download
from .dataset import DatasetCache
from .experiments import ExperimentRunner, RunConfig, default_experiment_configs
from .finetuning import (
    FinetuneConfig,
    NLLBTextFineTuner,
    WhisperASRFineTuner,
    SeamlessM4TFineTuner,
    WhisperNLLBCascadeFineTuner,
    build_finetune_comparison_table,
)
from .sota import (
    load_sota,
    load_yaml_references,
    validate_references,
    load_bib_references,   # deprecated alias
    validate_bib,          # deprecated alias
    build_sota_comparison_table,
    build_gap_table,
    build_system_ranking_table,
    build_improvement_table,
    generate_sota_interpretation,
)
from .paper_modes import (
    PaperModeConfig,
    get_paper_mode_config,
    PAPER_MODES,
    describe_paper_modes,
)
from .scaling import run_scaling_experiment
from .cascade_analysis import (
    run_oracle_cascade,
    run_error_propagation,
    run_breakeven_analysis,
    run_latency_profile,
)
from .transfer import (
    compute_typological_similarity,
    run_crosslingual_transfer,
    run_few_shot_scaling,
    run_interaction_regression,
)

__all__ = [
    "detect_environment", "install_colab_dependencies", "mount_google_drive",
    "detect_model_capabilities", "ModelCapabilities",
    "select_supported_languages", "AFRICAN_LANGUAGES", "get_adapter_type",
    "StepMonitor",
    "create_run_dirs", "save_config", "zip_run_outputs", "drive_backup", "colab_download",
    "DatasetCache",
    "ExperimentRunner", "RunConfig", "default_experiment_configs",
    "FinetuneConfig",
    "NLLBTextFineTuner", "WhisperASRFineTuner",
    "SeamlessM4TFineTuner", "WhisperNLLBCascadeFineTuner",
    "build_finetune_comparison_table",
    "load_sota",
    "load_yaml_references", "validate_references",
    "load_bib_references", "validate_bib",  # deprecated aliases
    "build_sota_comparison_table", "build_gap_table",
    "build_system_ranking_table", "build_improvement_table", "generate_sota_interpretation",
    "PaperModeConfig", "get_paper_mode_config", "PAPER_MODES", "describe_paper_modes",
    # Paper-specific analyses
    "run_scaling_experiment",
    "run_oracle_cascade", "run_error_propagation", "run_breakeven_analysis", "run_latency_profile",
    "compute_typological_similarity", "run_crosslingual_transfer",
    "run_few_shot_scaling", "run_interaction_regression",
]
