#!/usr/bin/env python3
"""Fill [RESULT:key] and [BASELINE:key] placeholders in paper_draft.md.

Reads results from:
    results/<paper_id>/results.csv           ← primary (from extract_results.py)
    papers/<paper_id>/baselines.csv          ← published comparison baselines

Writes:
    papers/<paper_id>/paper_draft_filled.md  ← complete paper with numbers inserted
    papers/<paper_id>/paper_draft.md         ← NEVER modified (original with placeholders)

Usage:
    python papers/fill_results.py paper1_benchmark
    python papers/fill_results.py --all

Full workflow:
    1. Run experiments on Colab
    2. Copy outputs to results/<paper_id>/from_colab/<experiment>/
    3. python papers/extract_results.py paper1_benchmark   → results.csv
    4. python papers/fill_results.py paper1_benchmark      → paper_draft_filled.md
"""

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
PAPERS_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"

PAPER_IDS = [
    "paper1_benchmark",
    "paper2_adaptation",
    "paper3_audio",
    "paper4_cascade",
    "paper5_transfer",
]


def load_results_csv(paper_id: str) -> dict[str, str]:
    """Load results.csv: key → value."""
    path = RESULTS_DIR / paper_id / "results.csv"
    results: dict[str, str] = {}
    if not path.exists():
        return results
    with open(path) as f:
        for row in csv.DictReader(f):
            key = row.get("key", "").strip()
            val = row.get("value", "").strip()
            if key and val:
                results[key] = val
    return results


def load_baselines_csv(paper_id: str) -> dict[str, str]:
    """Load baselines.csv: citation_key.language.metric → score."""
    path = PAPERS_DIR / paper_id / "baselines.csv"
    baselines: dict[str, str] = {}
    if not path.exists():
        return baselines
    with open(path) as f:
        for row in csv.DictReader(f):
            key  = (row.get("citation_key") or "").strip()
            lang = (row.get("language") or "").lower().replace(" ", "_")
            met  = (row.get("metric") or "").lower().replace(" ", "_").replace("/", "_").replace("+", "_")
            score = (row.get("score") or "").strip()
            if key and lang and met and score:
                baselines[f"{key}.{lang}.{met}"] = score
    return baselines


def fill_placeholders(
    template: str,
    results:  dict[str, str],
    baselines: dict[str, str],
) -> tuple[str, list[str], list[str]]:
    """Replace placeholders. Returns (filled_text, unfilled_results, unfilled_baselines)."""
    unfilled_results:   list[str] = []
    unfilled_baselines: list[str] = []

    def replace_result(m: re.Match) -> str:
        key = m.group(1).strip()
        val = results.get(key)
        if val is not None:
            return f"**{val}**"
        unfilled_results.append(key)
        return m.group(0)

    def replace_baseline(m: re.Match) -> str:
        key = m.group(1).strip()
        val = baselines.get(key)
        if val is not None:
            return f"*{val}*"
        unfilled_baselines.append(key)
        return m.group(0)

    filled = re.sub(r"\[RESULT:([^\]]+)\]",   replace_result,   template)
    filled = re.sub(r"\[BASELINE:([^\]]+)\]", replace_baseline, filled)

    return filled, unfilled_results, unfilled_baselines


def process_paper(paper_id: str) -> None:
    paper_dir = PAPERS_DIR / paper_id
    draft_path  = paper_dir / "paper_draft.md"

    if not draft_path.exists():
        print(f"  [skip] paper_draft.md not found in {paper_dir}")
        return

    template  = draft_path.read_text()
    results   = load_results_csv(paper_id)
    baselines = load_baselines_csv(paper_id)

    print(f"  Loaded {len(results)} results, {len(baselines)} baselines")

    filled, unfilled_r, unfilled_b = fill_placeholders(template, results, baselines)

    # Write filled version — never overwrite the original draft
    filled_path = paper_dir / "paper_draft_filled.md"
    filled_path.write_text(filled)
    print(f"  Written: {filled_path.relative_to(REPO_ROOT)}")
    print(f"  Original kept: {draft_path.relative_to(REPO_ROOT)}")

    # Report
    total_placeholders = len(unfilled_r) + len(unfilled_b)
    if total_placeholders == 0:
        print(f"  All placeholders filled — paper is ready.")
    else:
        if unfilled_r:
            print(f"\n  Unfilled [RESULT:*] ({len(unfilled_r)}) — run experiments first:")
            for k in sorted(set(unfilled_r)):
                print(f"    [RESULT:{k}]")
        if unfilled_b:
            print(f"\n  Unfilled [BASELINE:*] ({len(unfilled_b)}) — fill baselines.csv:")
            for k in sorted(set(unfilled_b)):
                print(f"    [BASELINE:{k}]")

    n_narrative = len(re.findall(r"\[NARRATIVE:[^\]]+\]", filled))
    if n_narrative:
        print(f"\n  [NARRATIVE:*] sections remaining: {n_narrative} — fill manually in paper_draft_filled.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paper_id", nargs="?", choices=PAPER_IDS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    targets = PAPER_IDS if (args.all or args.paper_id is None) else [args.paper_id]

    for paper_id in targets:
        print(f"\n{'─'*60}")
        print(f"  {paper_id}")
        print(f"{'─'*60}")
        process_paper(paper_id)


if __name__ == "__main__":
    main()
