#!/usr/bin/env python3
"""Fill [RESULT:key] placeholders in paper_outline.md from experiment output CSVs.

Usage:
    python papers/fill_results.py paper1_benchmark
    python papers/fill_results.py paper1_benchmark --results-dir outputs/2025-06-01_full_run/
    python papers/fill_results.py --all

The script looks for experiment outputs in the following order:
1. --results-dir argument (explicit path)
2. The most recent timestamped folder under each experiment's outputs/ dir
3. RESULTS_DIR environment variable

Output: paper_outline.filled.md in the paper directory (original unchanged).
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = Path(__file__).resolve().parent

PAPER_IDS = [
    "paper1_benchmark",
    "paper2_adaptation",
    "paper3_audio",
    "paper4_cascade",
    "paper5_transfer",
]

# Map experiment output directory names to result key prefixes
EXPERIMENT_DIRS = {
    "paper1_benchmark": [
        "FLEURS__SeamlessM4Tv2",
        "FLEURS__WhisperNLLB",
        "AfricanCeltic__SeamlessM4Tv2",
        "AfricanCeltic__WhisperNLLB",
    ],
    "paper2_adaptation": [
        "FLEURS__SeamlessM4Tv2",
        "FLEURS__WhisperNLLB",
    ],
    "paper3_audio": [
        "FLEURS__SeamlessM4Tv2",
        "FLEURS__WhisperNLLB",
    ],
    "paper4_cascade": [
        "FLEURS__SeamlessM4Tv2",
        "FLEURS__WhisperNLLB",
    ],
    "paper5_transfer": [
        "FLEURS__SeamlessM4Tv2",
        "FLEURS__WhisperNLLB",
    ],
}


def find_latest_output(experiment_dir: str, paper_mode: str) -> Path | None:
    """Return the most recent outputs/ subfolder for the given experiment and mode."""
    pattern = REPO_ROOT / experiment_dir / "outputs" / f"*_{paper_mode}*"
    candidates = sorted(glob.glob(str(pattern)), reverse=True)
    if not candidates:
        # Fall back: any outputs folder
        pattern = REPO_ROOT / experiment_dir / "outputs" / "*_full*"
        candidates = sorted(glob.glob(str(pattern)), reverse=True)
    return Path(candidates[0]) if candidates else None


def load_metrics_from_dir(output_dir: Path) -> dict[str, float]:
    """Scan metrics/ subfolder and load all CSV rows into a flat key→value dict.

    CSV files in outputs/<run>/metrics/ have columns like:
        experiment,language,direction,metric,value
    Keys are built as:  language.metric (lowercased, spaces→underscore)
    """
    metrics: dict[str, float] = {}
    metrics_dir = output_dir / "metrics"
    if not metrics_dir.exists():
        return metrics

    for csv_path in metrics_dir.glob("*.csv"):
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lang = row.get("language", "").lower().replace(" ", "_")
                    metric = row.get("metric", "").lower().replace(" ", "_")
                    value = row.get("value") or row.get("score", "")
                    key = f"{lang}.{metric}"
                    try:
                        metrics[key] = float(value)
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    return metrics


def load_baselines(paper_id: str) -> dict[str, str]:
    """Load published baselines from baselines.csv; return citation_key.language.metric → score."""
    baseline_path = PAPERS_DIR / paper_id / "baselines.csv"
    result: dict[str, str] = {}
    if not baseline_path.exists():
        return result
    with open(baseline_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("citation_key", "")
            lang = row.get("language", "").lower()
            metric = row.get("metric", "").lower().replace(" ", "_").replace("/", "_")
            score = row.get("score", "").strip()
            if key and lang and metric and score:
                result[f"{key}.{lang}.{metric}"] = score
    return result


def fill_placeholders(template: str, results: dict[str, str], baselines: dict[str, str]) -> tuple[str, list[str]]:
    """Replace [RESULT:key] and [BASELINE:key] in template. Return (filled_text, unfilled_keys)."""
    unfilled: list[str] = []

    def replace_result(m: re.Match) -> str:
        key = m.group(1)
        val = results.get(key)
        if val is not None:
            return f"**{val}**"
        unfilled.append(f"RESULT:{key}")
        return m.group(0)  # leave placeholder intact

    def replace_baseline(m: re.Match) -> str:
        key = m.group(1)
        val = baselines.get(key)
        if val is not None:
            return f"*{val}*"
        unfilled.append(f"BASELINE:{key}")
        return m.group(0)

    filled = re.sub(r"\[RESULT:([^\]]+)\]", replace_result, template)
    filled = re.sub(r"\[BASELINE:([^\]]+)\]", replace_baseline, filled)
    return filled, unfilled


def process_paper(paper_id: str, results_dir: Path | None = None) -> None:
    paper_dir = PAPERS_DIR / paper_id
    outline_path = paper_dir / "paper_outline.md"

    if not outline_path.exists():
        print(f"  [skip] {paper_id}: paper_outline.md not found")
        return

    template = outline_path.read_text()

    # Collect results from experiment outputs
    all_results: dict[str, str] = {}
    for exp_dir_name in EXPERIMENT_DIRS.get(paper_id, []):
        if results_dir:
            output_dir = results_dir / exp_dir_name
        else:
            mode = paper_id.split("_")[1]  # e.g. "benchmark", "adaptation"
            output_dir = find_latest_output(exp_dir_name, mode)

        if output_dir and output_dir.exists():
            metrics = load_metrics_from_dir(output_dir)
            all_results.update({k: str(round(v, 2)) for k, v in metrics.items()})

    # Load published baselines
    baselines = load_baselines(paper_id)

    # Fill
    filled, unfilled = fill_placeholders(template, all_results, baselines)

    # Write output
    out_path = paper_dir / "paper_outline.filled.md"
    out_path.write_text(filled)
    print(f"  Written: {out_path.relative_to(REPO_ROOT)}")

    if unfilled:
        print(f"  Unfilled placeholders ({len(unfilled)}):")
        for key in sorted(set(unfilled)):
            print(f"    [{key}]")
    else:
        print(f"  All placeholders filled.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paper_id", nargs="?", choices=PAPER_IDS + ["--all"])
    parser.add_argument("--all", action="store_true", help="Process all papers")
    parser.add_argument("--results-dir", type=Path, help="Explicit path to outputs directory")
    args = parser.parse_args()

    if args.all or args.paper_id is None:
        targets = PAPER_IDS
    else:
        targets = [args.paper_id]

    for paper_id in targets:
        print(f"\n{'─'*60}")
        print(f"  {paper_id}")
        print(f"{'─'*60}")
        process_paper(paper_id, args.results_dir)


if __name__ == "__main__":
    main()
