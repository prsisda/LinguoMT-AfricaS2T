#!/usr/bin/env python3
"""Extract experiment results from Colab output folders into a flat results.csv.

Usage:
    python papers/extract_results.py paper1_benchmark
    python papers/extract_results.py --all

Workflow:
    1. Copy Colab output folders into results/<paper_id>/from_colab/<experiment>/
    2. Run this script — it finds the latest benchmark_full (or relevant mode) folder
       per experiment, reads text_metrics.csv / asr_metrics.csv / audio_metrics.csv,
       maps rows to result keys, and writes results/<paper_id>/results.csv.
    3. Then run:  python papers/fill_results.py <paper_id>

Result key format:
    <experiment_prefix>.<language>.<metric>
    e.g.  fleurs_seamless.yoruba.bleu
          fleurs_whisper.hausa.wer
          ac_seamless.igbo.spbleu
"""

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPERS_DIR  = REPO_ROOT / "papers"

PAPER_IDS = [
    "paper1_benchmark",
    "paper2_adaptation",
    "paper3_audio",
    "paper4_cascade",
    "paper5_transfer",
]

# Map (experiment_dir_name, paper_mode) → result key prefix
EXPERIMENT_PREFIXES = {
    "FLEURS__SeamlessM4Tv2":       "fleurs_seamless",
    "FLEURS__WhisperNLLB":         "fleurs_whisper",
    "AfricanCeltic__SeamlessM4Tv2": "ac_seamless",
    "AfricanCeltic__WhisperNLLB":  "ac_whisper",
}

# Map paper_id → list of experiment folders to scan
PAPER_EXPERIMENTS = {
    "paper1_benchmark": list(EXPERIMENT_PREFIXES.keys()),
    "paper2_adaptation": ["FLEURS__SeamlessM4Tv2", "FLEURS__WhisperNLLB"],
    "paper3_audio":      ["FLEURS__SeamlessM4Tv2", "FLEURS__WhisperNLLB"],
    "paper4_cascade":    ["FLEURS__SeamlessM4Tv2", "FLEURS__WhisperNLLB"],
    "paper5_transfer":   ["FLEURS__SeamlessM4Tv2", "FLEURS__WhisperNLLB"],
}

# Map paper_id → paper_mode string used in output folder names
PAPER_MODE = {
    "paper1_benchmark":  "benchmark",
    "paper2_adaptation": "adaptation",
    "paper3_audio":      "audio",
    "paper4_cascade":    "cascade",
    "paper5_transfer":   "transfer",
}

# Map column name in metrics CSV → metric key suffix
METRIC_COL_MAP = {
    "BLEU":   "bleu",
    "spBLEU": "spbleu",
    "ChrF":   "chrf",
    "WER":    "wer",
    "CER":    "cer",
}

# Map direction label fragments → key suffix (for text metrics that have directions)
DIRECTION_SUFFIX = {
    "source_to_english": "",       # default / main direction — no suffix
    "english_to_source": ".textmt",
}


def find_latest_output(from_colab_dir: Path, experiment: str, mode: str) -> Path | None:
    """Find the most recent output folder for this experiment and mode."""
    exp_dir = from_colab_dir / experiment
    if not exp_dir.exists():
        return None

    # Look inside the experiment folder for an outputs/ subdirectory
    for base in [exp_dir / "outputs", exp_dir]:
        pattern = str(base / f"*{mode}*full*")
        candidates = sorted(glob.glob(pattern), reverse=True)
        if not candidates:
            # Fall back: any output folder
            pattern = str(base / "*full*")
            candidates = sorted(glob.glob(pattern), reverse=True)
        if candidates:
            return Path(candidates[0])

    return None


def load_metrics_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def language_to_key(lang_display: str) -> str:
    """Normalise language display name to key fragment."""
    return lang_display.lower().replace(" ", "_")


def direction_to_suffix(direction: str) -> str:
    for frag, suffix in DIRECTION_SUFFIX.items():
        if frag in direction.lower():
            return suffix
    return ""


def extract_text_metrics(rows: list[dict], prefix: str) -> dict[str, str]:
    """Map text_metrics.csv rows to result keys."""
    results: dict[str, str] = {}
    for row in rows:
        lang   = language_to_key(row.get("language", ""))
        dirsuf = direction_to_suffix(row.get("direction", "") + row.get("direction_label", ""))
        for col, metric_key in METRIC_COL_MAP.items():
            if col in row and row[col] not in ("", None):
                try:
                    val = round(float(row[col]), 2)
                    key = f"{prefix}.{lang}{dirsuf}.{metric_key}"
                    results[key] = str(val)
                except (ValueError, TypeError):
                    pass
    return results


def extract_asr_metrics(rows: list[dict], prefix: str) -> dict[str, str]:
    """Map asr_metrics.csv rows to result keys."""
    results: dict[str, str] = {}
    for row in rows:
        lang = language_to_key(row.get("language", ""))
        for col, metric_key in METRIC_COL_MAP.items():
            if col in row and row[col] not in ("", None):
                try:
                    val = round(float(row[col]), 1)   # WER as percentage
                    key = f"{prefix}.{lang}.{metric_key}"
                    results[key] = str(val)
                except (ValueError, TypeError):
                    pass
    return results


def load_template_keys(paper_id: str) -> list[dict]:
    """Load the results.csv template to get all expected keys."""
    template = RESULTS_DIR / paper_id / "results.csv"
    if not template.exists():
        return []
    with open(template) as f:
        return list(csv.DictReader(f))


def write_results_csv(paper_id: str, extracted: dict[str, str]) -> Path:
    """Merge extracted values into the results.csv template and write it."""
    template_rows = load_template_keys(paper_id)
    out_path = RESULTS_DIR / paper_id / "results.csv"

    # If template exists, update it in-place; otherwise create from scratch
    if template_rows:
        for row in template_rows:
            key = row.get("key", "")
            if key in extracted:
                row["value"] = extracted[key]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=template_rows[0].keys())
            writer.writeheader()
            writer.writerows(template_rows)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value", "experiment", "language", "metric", "notes"])
            for key, value in sorted(extracted.items()):
                writer.writerow([key, value, "", "", "", "auto-extracted"])

    return out_path


def process_paper(paper_id: str) -> None:
    mode         = PAPER_MODE.get(paper_id, "benchmark")
    from_colab   = RESULTS_DIR / paper_id / "from_colab"
    experiments  = PAPER_EXPERIMENTS.get(paper_id, [])

    all_extracted: dict[str, str] = {}
    found_any = False

    for exp in experiments:
        prefix = EXPERIMENT_PREFIXES.get(exp, exp.lower().replace("__", "_"))
        output_dir = find_latest_output(from_colab, exp, mode)

        if not output_dir:
            print(f"  [skip] {exp} — no {mode}_full output found in {from_colab / exp}")
            continue

        found_any = True
        metrics_dir = output_dir / "metrics"
        print(f"  [read] {exp} → {output_dir.name}/metrics/")

        text_rows  = load_metrics_csv(metrics_dir / "text_metrics.csv")
        asr_rows   = load_metrics_csv(metrics_dir / "asr_metrics.csv")
        audio_rows = load_metrics_csv(metrics_dir / "audio_metrics.csv")

        all_extracted.update(extract_text_metrics(text_rows,  prefix))
        all_extracted.update(extract_asr_metrics(asr_rows,    prefix))
        all_extracted.update(extract_text_metrics(audio_rows, prefix))

    if not found_any:
        print(f"  No experiment outputs found. Copy Colab output folders to:")
        print(f"  {from_colab}/<experiment>/outputs/")
        return

    out_path = write_results_csv(paper_id, all_extracted)
    print(f"\n  Written: {out_path.relative_to(REPO_ROOT)}")

    # Report coverage
    all_rows = load_template_keys(paper_id)
    filled   = [r for r in all_rows if r.get("value")]
    print(f"  Coverage: {len(filled)}/{len(all_rows)} keys populated")
    missing  = [r["key"] for r in all_rows if not r.get("value")]
    if missing:
        print(f"  Missing ({len(missing)}):")
        for k in missing:
            print(f"    {k}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paper_id", nargs="?", choices=PAPER_IDS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    targets = PAPER_IDS if (args.all or args.paper_id is None) else [args.paper_id]

    for paper_id in targets:
        print(f"\n{'─'*60}")
        print(f"  Extracting: {paper_id}")
        print(f"{'─'*60}")
        process_paper(paper_id)


if __name__ == "__main__":
    main()
