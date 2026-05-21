#!/usr/bin/env python3
"""Extract experiment results from Colab output folders into a flat results.csv.

Usage:
    python papers/extract_results.py paper1_benchmark
    python papers/extract_results.py --all

Workflow:
    1. Copy Colab output folders into results/<paper_id>/from_colab/<experiment>/
    2. Run this script — it finds the latest <mode>_full folder per experiment,
       reads text_metrics.csv / asr_metrics.csv / audio_metrics.csv (and any
       paper-specific extra CSVs), maps rows to result keys, and writes
       results/<paper_id>/results.csv.
    3. Then run:  python papers/fill_results.py <paper_id>

Result key format (standard):
    <experiment_prefix>.<language>.<metric>
    e.g.  fleurs_seamless.yoruba.bleu       ← audio S2TT
          fleurs_seamless.yoruba.textmt.bleu ← text MT ceiling (gold transcript)
          fleurs_whisper.hausa.wer           ← ASR WER
          ac_seamless.igbo.spbleu

Paper-specific aliases remap standard keys to the names used in each paper's
results.csv template (see PAPER_KEY_ALIASES below).
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPERS_DIR  = REPO_ROOT / "papers"

PAPER_IDS = [
    "paper1_benchmark",
    "paper2_adaptation",
    "paper3_audio",
    "paper4_cascade",
    "paper5_transfer",
]

# Map experiment dir name → standard result key prefix
EXPERIMENT_PREFIXES = {
    "FLEURS__SeamlessM4Tv2":        "fleurs_seamless",
    "FLEURS__WhisperNLLB":          "fleurs_whisper",
    "AfricanCeltic__SeamlessM4Tv2": "ac_seamless",
    "AfricanCeltic__WhisperNLLB":   "ac_whisper",
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

# ── Paper-specific key aliases ─────────────────────────────────────────────────
# Maps standard extracted key → paper-specific result key.
# Applied after standard extraction so both keys end up in the extracted dict.
PAPER_KEY_ALIASES: dict[str, dict[str, str]] = {
    "paper3_audio": {
        # SeamlessM4T audio paths
        "fleurs_seamless.yoruba.bleu":         "seamless.yoruba.s2tt.bleu",
        "fleurs_seamless.igbo.bleu":           "seamless.igbo.s2tt.bleu",
        "fleurs_seamless.swahili.bleu":        "seamless.swahili.s2tt.bleu",
        "fleurs_seamless.yoruba.wer":          "seamless.yoruba.asr.wer",
        "fleurs_seamless.igbo.wer":            "seamless.igbo.asr.wer",
        "fleurs_seamless.swahili.wer":         "seamless.swahili.asr.wer",
        "fleurs_seamless.yoruba.textmt.bleu":  "seamless.yoruba.textmt.bleu",
        "fleurs_seamless.igbo.textmt.bleu":    "seamless.igbo.textmt.bleu",
        "fleurs_seamless.swahili.textmt.bleu": "seamless.swahili.textmt.bleu",
        # Whisper+NLLB cascade paths
        "fleurs_whisper.yoruba.bleu":          "cascade.yoruba.s2tt.bleu",
        "fleurs_whisper.hausa.bleu":           "cascade.hausa.s2tt.bleu",
        "fleurs_whisper.swahili.bleu":         "cascade.swahili.s2tt.bleu",
        "fleurs_whisper.yoruba.wer":           "cascade.yoruba.asr.wer",
        "fleurs_whisper.hausa.wer":            "cascade.hausa.asr.wer",
        "fleurs_whisper.swahili.wer":          "cascade.swahili.asr.wer",
        "fleurs_whisper.yoruba.textmt.bleu":   "cascade.yoruba.textmt.bleu",
        "fleurs_whisper.hausa.textmt.bleu":    "cascade.hausa.textmt.bleu",
        "fleurs_whisper.swahili.textmt.bleu":  "cascade.swahili.textmt.bleu",
        "fleurs_whisper.igbo.textmt.bleu":     "cascade.igbo.textmt.bleu",
    },
    "paper4_cascade": {
        # E2E baselines (SeamlessM4T)
        "fleurs_seamless.yoruba.bleu":   "e2e.yoruba.bleu",
        "fleurs_seamless.igbo.bleu":     "e2e.igbo.bleu",
        "fleurs_seamless.swahili.bleu":  "e2e.swahili.bleu",
        # Cascade baselines (Whisper+NLLB)
        "fleurs_whisper.yoruba.bleu":    "cascade.yoruba.bleu",
        "fleurs_whisper.hausa.bleu":     "cascade.hausa.bleu",
        "fleurs_whisper.swahili.bleu":   "cascade.swahili.bleu",
        "fleurs_whisper.yoruba.wer":     "cascade.yoruba.wer",
        "fleurs_whisper.hausa.wer":      "cascade.hausa.wer",
        "fleurs_whisper.swahili.wer":    "cascade.swahili.wer",
    },
    "paper5_transfer": {
        # Zero-shot baselines (from Paper 1 run in transfer mode)
        "fleurs_seamless.yoruba.bleu":   "zero.yoruba.bleu",
        "fleurs_seamless.igbo.bleu":     "zero.igbo.bleu",
        "fleurs_seamless.swahili.bleu":  "zero.swahili.bleu",
        "fleurs_whisper.yoruba.wer":     "zero.yoruba.wer",
        "fleurs_whisper.hausa.wer":      "zero.hausa.wer",
        # Monolingual FT (after fine-tuning same language)
        "fleurs_seamless.yoruba.textmt.bleu":  "mono.yoruba.bleu",
        "fleurs_seamless.igbo.textmt.bleu":    "mono.igbo.bleu",
        "fleurs_seamless.swahili.textmt.bleu": "mono.swahili.bleu",
    },
}

# ── Extra CSVs written by paper-specific analysis scripts ─────────────────────
# Maps paper_id → list of (csv_filename, extraction_fn_name)
EXTRA_CSV_SOURCES: dict[str, list[str]] = {
    "paper2_adaptation": ["scaling_metrics.csv"],
    "paper4_cascade":    [
        "oracle_cascade_metrics.csv",
        "error_propagation_metrics.csv",
        "breakeven_metrics.csv",
        "latency_metrics.csv",
    ],
    "paper5_transfer":   [
        "typological_similarity.csv",
        "crosslingual_transfer_metrics.csv",
        "few_shot_scaling_metrics.csv",
        "interaction_regression.csv",
    ],
}


# ── File finders ───────────────────────────────────────────────────────────────

def find_latest_output(from_colab_dir: Path, experiment: str, mode: str) -> Path | None:
    """Find the most recent output folder for this experiment and mode."""
    exp_dir = from_colab_dir / experiment
    if not exp_dir.exists():
        return None
    for base in [exp_dir / "outputs", exp_dir]:
        pattern = str(base / f"*{mode}*full*")
        candidates = sorted(glob.glob(pattern), reverse=True)
        if not candidates:
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


# ── Key builders ──────────────────────────────────────────────────────────────

def language_to_key(lang_display: str) -> str:
    return lang_display.lower().replace(" ", "_")


def direction_to_suffix(direction: str, is_audio: bool = False) -> str:
    """Return the key suffix for a given direction.

    For TEXT experiments (is_audio=False):
      source_to_english → .textmt  (gold source transcript → English = text MT ceiling)
      english_to_source → .rev     (English → source, rarely used)

    For AUDIO experiments (is_audio=True):
      source_to_english → ""       (main S2TT / cascade result)
      english_to_source → .rev     (rarely used)
    """
    d = direction.lower()
    if "source_to_english" in d:
        return "" if is_audio else ".textmt"
    if "english_to_source" in d:
        return ".rev"
    return ""


def extract_text_metrics(rows: list[dict], prefix: str, is_audio: bool = False) -> dict[str, str]:
    """Map text_metrics.csv or audio_metrics.csv rows to result keys."""
    results: dict[str, str] = {}
    for row in rows:
        lang   = language_to_key(row.get("language", ""))
        dirsuf = direction_to_suffix(
            row.get("direction", "") + row.get("direction_label", ""),
            is_audio=is_audio,
        )
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
                    val = round(float(row[col]), 1)
                    key = f"{prefix}.{lang}.{metric_key}"
                    results[key] = str(val)
                except (ValueError, TypeError):
                    pass
    return results


def extract_scaling_metrics(rows: list[dict]) -> dict[str, str]:
    """Map scaling_metrics.csv rows → scaling.<lang>.<metric>.<budget> keys."""
    results: dict[str, str] = {}
    for row in rows:
        lang   = language_to_key(row.get("language", ""))
        budget = str(row.get("budget", "")).strip()
        task   = row.get("task", "")
        if not lang or not budget:
            continue
        for col, metric_key in METRIC_COL_MAP.items():
            if col in row and row[col] not in ("", None):
                try:
                    val = round(float(row[col]), 2)
                    key = f"scaling.{lang}.{metric_key}.{budget}"
                    results[key] = str(val)
                except (ValueError, TypeError):
                    pass
    return results


def extract_oracle_cascade_metrics(rows: list[dict]) -> dict[str, str]:
    """Map oracle_cascade_metrics.csv rows → oracle.<lang>.bleu keys."""
    results: dict[str, str] = {}
    for row in rows:
        lang = language_to_key(row.get("language", ""))
        for col, metric_key in METRIC_COL_MAP.items():
            if col in row and row[col] not in ("", None):
                try:
                    val = round(float(row[col]), 2)
                    results[f"oracle.{lang}.{metric_key}"] = str(val)
                except (ValueError, TypeError):
                    pass
    return results


def extract_breakeven_metrics(rows: list[dict]) -> dict[str, str]:
    """Map breakeven_metrics.csv rows → breakeven.<lang>.wer keys."""
    results: dict[str, str] = {}
    for row in rows:
        lang = language_to_key(row.get("language", ""))
        val  = row.get("breakeven_wer", "")
        if lang and val not in ("", None, "None"):
            try:
                results[f"breakeven.{lang}.wer"] = str(round(float(val) * 100, 1))
            except (ValueError, TypeError):
                pass
    return results


def extract_latency_metrics(rows: list[dict]) -> dict[str, str]:
    """Map latency_metrics.csv rows → latency.<arch>.<stat> keys."""
    results: dict[str, str] = {}
    for row in rows:
        arch = row.get("architecture", "").lower().strip()
        for col, stat in [("median_ms", "median_ms"), ("p95_ms", "p95_ms"), ("peak_vram_mb", "vram_mb")]:
            val = row.get(col, "")
            if val not in ("", None, "None"):
                try:
                    results[f"latency.{arch}.{stat}"] = str(round(float(val), 1))
                except (ValueError, TypeError):
                    pass
    return results


def extract_typological_similarity(rows: list[dict]) -> dict[str, str]:
    """Map typological_similarity.csv rows → topo.<lang_a>_<lang_b> keys."""
    results: dict[str, str] = {}
    for row in rows:
        a   = row.get("lang_a", "").strip()
        b   = row.get("lang_b", "").strip()
        val = row.get("cosine_similarity", "")
        if a and b and val not in ("", None, "nan"):
            try:
                results[f"topo.{a}_{b}"] = str(round(float(val), 4))
            except (ValueError, TypeError):
                pass
    return results


def extract_crosslingual_transfer(rows: list[dict]) -> dict[str, str]:
    """Map crosslingual_transfer_metrics.csv rows → xfer.<src>_to_<tgt>.<metric> keys."""
    results: dict[str, str] = {}
    for row in rows:
        src  = language_to_key(row.get("train_language_key", row.get("train_language", "")))
        tgt  = language_to_key(row.get("eval_language_key",  row.get("eval_language",  "")))
        task = row.get("task", "")
        if not src or not tgt:
            continue
        for col, metric_key in METRIC_COL_MAP.items():
            if col in row and row[col] not in ("", None):
                try:
                    val = round(float(row[col]), 2)
                    results[f"xfer.{src}_to_{tgt}.{metric_key}"] = str(val)
                except (ValueError, TypeError):
                    pass
    return results


def extract_few_shot_scaling(rows: list[dict]) -> dict[str, str]:
    """Map few_shot_scaling_metrics.csv rows → few.<lang>.<metric>.<budget> keys."""
    results: dict[str, str] = {}
    for row in rows:
        lang   = language_to_key(row.get("language", ""))
        budget = str(row.get("budget", "")).strip()
        if not lang or not budget:
            continue
        for col, metric_key in METRIC_COL_MAP.items():
            if col in row and row[col] not in ("", None):
                try:
                    val = round(float(row[col]), 2)
                    results[f"few.{lang}.{metric_key}.{budget}"] = str(val)
                except (ValueError, TypeError):
                    pass
    return results


def extract_interaction_regression(rows: list[dict]) -> dict[str, str]:
    """Map interaction_regression.csv rows → interaction.coeff / interaction.pvalue keys."""
    results: dict[str, str] = {}
    for row in rows:
        task   = row.get("task", "")
        metric = row.get("metric", "")
        coeff  = row.get("interaction_coeff", "")
        pval   = row.get("p_value", "")
        if task == "text_mt" and metric == "BLEU":
            if coeff not in ("", None):
                try:
                    results["interaction.coeff"] = str(round(float(coeff), 4))
                except (ValueError, TypeError):
                    pass
            if pval not in ("", None):
                try:
                    results["interaction.pvalue"] = str(round(float(pval), 4))
                except (ValueError, TypeError):
                    pass
    return results


# ── Extra CSV dispatcher ──────────────────────────────────────────────────────

_EXTRA_EXTRACTORS = {
    "scaling_metrics.csv":              extract_scaling_metrics,
    "oracle_cascade_metrics.csv":       extract_oracle_cascade_metrics,
    "breakeven_metrics.csv":            extract_breakeven_metrics,
    "latency_metrics.csv":              extract_latency_metrics,
    "typological_similarity.csv":       extract_typological_similarity,
    "crosslingual_transfer_metrics.csv": extract_crosslingual_transfer,
    "few_shot_scaling_metrics.csv":     extract_few_shot_scaling,
    "interaction_regression.csv":       extract_interaction_regression,
}


def extract_extra_metrics(paper_id: str, from_colab: Path, mode: str) -> dict[str, str]:
    """Read paper-specific extra CSVs from any experiment folder or results dir."""
    results: dict[str, str] = {}
    csv_names = EXTRA_CSV_SOURCES.get(paper_id, [])
    if not csv_names:
        return results

    # Collect all possible locations: latest output metrics/ dirs + results dir
    search_paths: list[Path] = []
    for exp in PAPER_EXPERIMENTS.get(paper_id, []):
        output_dir = find_latest_output(from_colab, exp, mode)
        if output_dir:
            search_paths.append(output_dir / "metrics")
    search_paths.append(RESULTS_DIR / paper_id)  # also check results dir directly

    for csv_name in csv_names:
        extractor = _EXTRA_EXTRACTORS.get(csv_name)
        if extractor is None:
            continue
        for search_path in search_paths:
            csv_path = search_path / csv_name
            rows = load_metrics_csv(csv_path)
            if rows:
                results.update(extractor(rows))
                print(f"  [extra] {csv_name} → {len(rows)} rows from {csv_path.parent.name}/")
                break
        else:
            pass  # not found — silent, user may not have run that analysis yet

    return results


# ── Template I/O ──────────────────────────────────────────────────────────────

def load_template_keys(paper_id: str) -> list[dict]:
    path = RESULTS_DIR / paper_id / "results.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def write_results_csv(paper_id: str, extracted: dict[str, str]) -> Path:
    template_rows = load_template_keys(paper_id)
    out_path = RESULTS_DIR / paper_id / "results.csv"

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


# ── Main extraction ───────────────────────────────────────────────────────────

def process_paper(paper_id: str) -> None:
    mode       = PAPER_MODE.get(paper_id, "benchmark")
    from_colab = RESULTS_DIR / paper_id / "from_colab"
    experiments = PAPER_EXPERIMENTS.get(paper_id, [])
    aliases     = PAPER_KEY_ALIASES.get(paper_id, {})

    all_extracted: dict[str, str] = {}
    found_any = False

    for exp in experiments:
        prefix     = EXPERIMENT_PREFIXES.get(exp, exp.lower().replace("__", "_"))
        output_dir = find_latest_output(from_colab, exp, mode)

        if not output_dir:
            print(f"  [skip] {exp} — no {mode}_full output found in {from_colab / exp}")
            continue

        found_any = True
        metrics_dir = output_dir / "metrics"
        print(f"  [read] {exp} → {output_dir.name}/metrics/")

        # Text experiments: source→English is the TEXT MT ceiling (.textmt suffix)
        text_rows  = load_metrics_csv(metrics_dir / "text_metrics.csv")
        all_extracted.update(extract_text_metrics(text_rows, prefix, is_audio=False))

        # ASR experiments
        asr_rows = load_metrics_csv(metrics_dir / "asr_metrics.csv")
        all_extracted.update(extract_asr_metrics(asr_rows, prefix))

        # Audio experiments: source→English is the S2TT/cascade result (no suffix)
        audio_rows = load_metrics_csv(metrics_dir / "audio_metrics.csv")
        all_extracted.update(extract_text_metrics(audio_rows, prefix, is_audio=True))

        # Paper 2 scaling (written to the same metrics dir by the run script)
        if paper_id == "paper2_adaptation":
            scale_rows = load_metrics_csv(metrics_dir / "scaling_metrics.csv")
            if scale_rows:
                all_extracted.update(extract_scaling_metrics(scale_rows))
                print(f"  [scaling] {len(scale_rows)} rows")

    # Extra paper-specific CSVs (analysis scripts)
    extra = extract_extra_metrics(paper_id, from_colab, mode)
    all_extracted.update(extra)

    # Apply paper-specific key aliases (add aliased versions to extracted dict)
    for std_key, paper_key in aliases.items():
        if std_key in all_extracted:
            all_extracted[paper_key] = all_extracted[std_key]

    if not found_any and not extra:
        print(f"  No experiment outputs found. Copy Colab output folders to:")
        print(f"  {from_colab}/<experiment>/outputs/")
        return

    out_path = write_results_csv(paper_id, all_extracted)
    print(f"\n  Written: {out_path.relative_to(REPO_ROOT)}")

    # Coverage report
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
