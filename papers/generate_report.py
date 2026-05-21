#!/usr/bin/env python3
"""Generate a Markdown results report to facilitate paper writing.

Reads consolidated metric CSVs produced by the Colab notebooks and writes a
structured Markdown report containing:
  - Results tables (text MT, ASR, audio strategies)
  - Cross-experiment comparisons (SeamlessM4T vs Whisper+NLLB)
  - SOTA comparison and gap analysis (if a SOTA file is provided)
  - Paper-mode-specific discussion with key observations

Usage (from repo root):
    python papers/generate_report.py benchmark
    python papers/generate_report.py cascade --sota-file sota/paper1_benchmark/sota_results.csv
    python papers/generate_report.py --help

The consolidated directory is auto-detected as the most recent
/content/outputs/consolidated_*/ on Colab, or passed explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT  = Path(__file__).resolve().parent.parent
PAPERS_DIR = Path(__file__).resolve().parent

# ── SOTA column normalisation ─────────────────────────────────────────────────

SOTA_COL_ALIASES = {
    "system": "system", "model": "system", "citation": "system",
    "language": "language", "lang": "language",
    "bleu": "BLEU", "sacrebleu": "BLEU", "spbleu": "spBLEU",
    "chrf": "ChrF", "chrf2": "ChrF",
    "wer": "WER",
    "direction": "direction",
    "venue": "venue", "year": "year",
}

PAPER_MODE_NAMES = {
    "benchmark":  "Paper 1 — LinguoMT: Zero-Shot Baselines",
    "adaptation": "Paper 2 — LinguoMT-Adapt: Fine-Tuning",
    "audio":      "Paper 3 — LinguoMT-Audio: Audio Strategies",
    "cascade":    "Paper 4 — LinguoMT-Cascade: Cascade vs End-to-End",
    "transfer":   "Paper 5 — LinguoMT-Transfer: Cross-Lingual Transfer",
}

EXPERIMENT_LABELS = {
    "AfricanCeltic__SeamlessM4Tv2_Large": "AC × SeamlessM4T-v2",
    "AfricanCeltic__WhisperNLLB":         "AC × Whisper+NLLB",
    "FLEURS__SeamlessM4Tv2_Large":        "FLEURS × SeamlessM4T-v2",
    "FLEURS__WhisperNLLB":                "FLEURS × Whisper+NLLB",
}


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_sota(sota_file: str | None) -> list[dict]:
    if not sota_file:
        return []
    path = Path(sota_file) if Path(sota_file).is_absolute() else REPO_ROOT / sota_file
    if not path.exists():
        print(f"  [SOTA] File not found: {path}", file=sys.stderr)
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            normalised = {SOTA_COL_ALIASES.get(k.lower(), k): v for k, v in row.items()}
            rows.append(normalised)
    return rows


def label(exp_family: str) -> str:
    return EXPERIMENT_LABELS.get(exp_family, exp_family)


def find_consolidated(outputs_root: Path) -> Path | None:
    candidates = sorted(outputs_root.glob("consolidated_*/"), reverse=True)
    return candidates[0] if candidates else None


# ─────────────────────────────────────────────────────────────────────────────
# Markdown formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val: Any, decimals: int = 2) -> str:
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val) if val else "—"


def _md_table(headers: list[str], rows: list[list]) -> str:
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
             for i, h in enumerate(headers)]
    sep   = "| " + " | ".join("-" * w for w in col_w) + " |"
    head  = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |"
    lines = [head, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(col_w[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def _bold_max(values: list[str]) -> list[str]:
    """Bold the highest numeric value in a list of formatted strings."""
    floats = []
    for v in values:
        try:
            floats.append(float(v))
        except (TypeError, ValueError):
            floats.append(None)
    if not any(f is not None for f in floats):
        return values
    max_val = max(f for f in floats if f is not None)
    return [f"**{v}**" if f == max_val else v for v, f in zip(values, floats)]


# ─────────────────────────────────────────────────────────────────────────────
# Results section builders
# ─────────────────────────────────────────────────────────────────────────────

def _direction_label(d: str) -> str:
    d = d.lower()
    if "source_to_english" in d or "src→en" in d:
        return "Source → English"
    if "english_to_source" in d or "en→src" in d:
        return "English → Source"
    return d.replace("_", " ").title()


def section_text(rows: list[dict]) -> str:
    if not rows:
        return "_No text translation metrics available._\n"
    lines = ["### Text Translation Results\n"]
    lines.append(
        "_BLEU and ChrF are computed with SacreBLEU. "
        "Source→English = gold transcript MT (ceiling); English→Source = reverse direction._\n"
    )

    # Pivot: experiment × language × direction → BLEU / ChrF
    by_exp: dict[str, list[dict]] = {}
    for r in rows:
        exp = label(r.get("experiment", ""))
        by_exp.setdefault(exp, []).append(r)

    for exp_label_val, exp_rows in sorted(by_exp.items()):
        lines.append(f"\n#### {exp_label_val}\n")
        headers = ["Language", "Direction", "BLEU", "ChrF", "Samples"]
        table_rows = []
        bleu_vals = []
        for r in sorted(exp_rows, key=lambda x: (x.get("language", ""), x.get("direction_label", ""))):
            bleu = _fmt(r.get("BLEU", r.get("bleu", "")))
            bleu_vals.append(bleu)
            table_rows.append([
                r.get("language", "—"),
                _direction_label(r.get("direction_label", r.get("direction", ""))),
                bleu,
                _fmt(r.get("ChrF", r.get("chrf", ""))),
                r.get("num_samples", "—"),
            ])
        if table_rows:
            bleu_col = [row[2] for row in table_rows]
            bold     = _bold_max(bleu_col)
            for i, row in enumerate(table_rows):
                row[2] = bold[i]
        lines.append(_md_table(headers, table_rows))
        lines.append("")
    return "\n".join(lines)


def section_asr(rows: list[dict]) -> str:
    if not rows:
        return "_No ASR metrics available._\n"
    lines = ["### ASR Results\n"]
    lines.append("_WER = Word Error Rate; CER = Character Error Rate (lower is better)._\n")

    by_exp: dict[str, list[dict]] = {}
    for r in rows:
        exp = label(r.get("experiment", ""))
        by_exp.setdefault(exp, []).append(r)

    for exp_label_val, exp_rows in sorted(by_exp.items()):
        lines.append(f"\n#### {exp_label_val}\n")
        headers = ["Language", "WER (%)", "CER (%)", "Samples"]
        table_rows = []
        for r in sorted(exp_rows, key=lambda x: x.get("language", "")):
            table_rows.append([
                r.get("language", "—"),
                _fmt(r.get("WER", r.get("wer", ""))),
                _fmt(r.get("CER", r.get("cer", ""))),
                r.get("num_samples", "—"),
            ])
        wer_col = [row[1] for row in table_rows]
        # Bold LOWEST WER
        floats = []
        for v in wer_col:
            try:
                floats.append(float(v))
            except (TypeError, ValueError):
                floats.append(None)
        if any(f is not None for f in floats):
            min_val = min(f for f in floats if f is not None)
            for i, (row, f) in enumerate(zip(table_rows, floats)):
                if f == min_val:
                    row[1] = f"**{row[1]}**"
        lines.append(_md_table(headers, table_rows))
        lines.append("")
    return "\n".join(lines)


def section_audio(rows: list[dict]) -> str:
    if not rows:
        return "_No audio strategy metrics available._\n"
    lines = ["### Audio Strategy Results\n"]
    lines.append(
        "_Direct audio = raw waveform input; Normalised = amplitude-normalised; "
        "Trimmed = silence-trimmed; Chunk-based = fixed-length segments._\n"
    )
    by_exp: dict[str, list[dict]] = {}
    for r in rows:
        exp = label(r.get("experiment", ""))
        by_exp.setdefault(exp, []).append(r)

    for exp_label_val, exp_rows in sorted(by_exp.items()):
        lines.append(f"\n#### {exp_label_val}\n")
        headers = ["Language", "Strategy", "Direction", "BLEU", "ChrF", "Samples"]
        table_rows = []
        for r in sorted(exp_rows, key=lambda x: (x.get("language",""), x.get("strategy_label",""))):
            table_rows.append([
                r.get("language", "—"),
                r.get("strategy_label", r.get("strategy_key", "—")),
                _direction_label(r.get("direction_label", r.get("direction", ""))),
                _fmt(r.get("BLEU", r.get("bleu", ""))),
                _fmt(r.get("ChrF", r.get("chrf", ""))),
                r.get("num_samples", "—"),
            ])
        bleu_col = [row[3] for row in table_rows]
        bold     = _bold_max(bleu_col)
        for i, row in enumerate(table_rows):
            row[3] = bold[i]
        lines.append(_md_table(headers, table_rows))
        lines.append("")
    return "\n".join(lines)


def section_cross_experiment(text_rows: list[dict], asr_rows: list[dict]) -> str:
    if not text_rows:
        return ""
    lines = ["### Cross-Experiment Comparison\n"]
    lines.append(
        "_Source → English BLEU. Compares SeamlessM4T (end-to-end) against "
        "Whisper+NLLB (cascade) on overlapping languages._\n"
    )

    # Collect BLEU per (exp_family, language) for source→english direction
    bleu_map: dict[tuple, float] = {}
    for r in text_rows:
        d = (r.get("direction_label", "") + r.get("direction", "")).lower()
        if "source_to_english" not in d and "src→en" not in d:
            continue
        exp = label(r.get("experiment", ""))
        lang = r.get("language", "")
        try:
            bleu_map[(exp, lang)] = float(r.get("BLEU", r.get("bleu", 0)) or 0)
        except (TypeError, ValueError):
            pass

    experiments = sorted({k[0] for k in bleu_map})
    languages   = sorted({k[1] for k in bleu_map})

    if not experiments or not languages:
        return ""

    headers = ["Language"] + experiments
    table_rows = []
    for lang in languages:
        row = [lang]
        vals = [_fmt(bleu_map.get((exp, lang), "")) for exp in experiments]
        row.extend(vals)
        table_rows.append(row)

    # Bold the best per language (highest BLEU across experiments)
    for row in table_rows:
        vals = row[1:]
        bold = _bold_max(vals)
        for i, b in enumerate(bold):
            row[i + 1] = b

    lines.append(_md_table(headers, table_rows))
    lines.append("")

    # Add a brief best-system note
    best_by_lang = {}
    for lang in languages:
        best_exp  = max(experiments, key=lambda e: bleu_map.get((e, lang), -1))
        best_bleu = bleu_map.get((best_exp, lang), None)
        if best_bleu is not None and best_bleu > 0:
            best_by_lang[lang] = (best_exp, best_bleu)

    if best_by_lang:
        lines.append("\n**Best system per language (Source → English BLEU):**\n")
        for lang, (best_exp, best_bleu) in sorted(best_by_lang.items()):
            lines.append(f"- **{lang}**: {best_exp} ({best_bleu:.2f} BLEU)")
        lines.append("")

    return "\n".join(lines)


def section_sota(text_rows: list[dict], sota_rows: list[dict]) -> str:
    if not sota_rows:
        return ""
    lines = [
        "### SOTA Comparison\n",
        "_Published results from prior work. Our results are shown for the Source → English direction._\n",
    ]

    # Collect our best BLEU per language
    our_bleu: dict[str, float] = {}
    for r in text_rows:
        d = (r.get("direction_label", "") + r.get("direction", "")).lower()
        if "source_to_english" not in d and "src→en" not in d:
            continue
        lang = r.get("language", "").lower()
        try:
            v = float(r.get("BLEU", r.get("bleu", 0)) or 0)
            if v > our_bleu.get(lang, -1):
                our_bleu[lang] = v
        except (TypeError, ValueError):
            pass

    # Build SOTA table: system, language, BLEU, venue/year
    languages = sorted(our_bleu.keys())
    headers = ["System", "Language", "BLEU", "Venue / Year", "Gap vs Ours"]
    table_rows = []
    for r in sota_rows:
        lang = (r.get("language", "") or "").lower()
        try:
            sota_bleu = float(r.get("BLEU", r.get("bleu", "") or ""))
        except (TypeError, ValueError):
            sota_bleu = None
        our = our_bleu.get(lang)
        gap = f"{(sota_bleu - our):.2f}" if (sota_bleu is not None and our is not None) else "—"
        venue = f"{r.get('venue', '')} {r.get('year', '')}".strip()
        table_rows.append([
            r.get("system", r.get("model", "—")),
            r.get("language", "—"),
            _fmt(sota_bleu),
            venue or "—",
            gap,
        ])
    if table_rows:
        lines.append(_md_table(headers, table_rows))
        lines.append("")

    # Our results row
    if our_bleu:
        lines.append("\n**Our results (best across all experiments, Source → English):**\n")
        headers2 = ["Language", "Best BLEU", "vs SOTA (gap)"]
        rows2 = []
        for lang in sorted(our_bleu.keys()):
            our_val = our_bleu[lang]
            sota_vals = [float(r.get("BLEU", r.get("bleu", 0)) or 0)
                         for r in sota_rows
                         if r.get("language","").lower() == lang
                         and r.get("BLEU", r.get("bleu", ""))]
            sota_vals = [v for v in sota_vals if v > 0]
            best_sota = max(sota_vals) if sota_vals else None
            gap_str = f"{(our_val - best_sota):.2f}" if best_sota else "—"
            rows2.append([lang.title(), _fmt(our_val), gap_str])
        lines.append(_md_table(headers2, rows2))
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Discussion generator (paper-mode aware)
# ─────────────────────────────────────────────────────────────────────────────

def _top_bottom(values: dict[str, float], n: int = 2) -> tuple[list, list]:
    ranked = sorted(values.items(), key=lambda x: x[1], reverse=True)
    return ranked[:n], ranked[-n:]


def _language_bleu(rows: list[dict], direction_filter: str = "source_to_english") -> dict[str, float]:
    best: dict[str, float] = {}
    for r in rows:
        d = (r.get("direction_label", "") + r.get("direction", "")).lower()
        if direction_filter not in d:
            continue
        lang = r.get("language", "")
        try:
            v = float(r.get("BLEU", r.get("bleu", 0)) or 0)
            if v > best.get(lang, -1):
                best[lang] = v
        except (TypeError, ValueError):
            pass
    return best


def discussion_benchmark(text_rows, asr_rows, sota_rows) -> str:
    lines = []
    lang_bleu = _language_bleu(text_rows)
    if lang_bleu:
        top, bot = _top_bottom(lang_bleu)
        lines.append(
            f"- **Highest-performing language** (Source→English BLEU): "
            f"{top[0][0]} ({top[0][1]:.2f})"
            + (f", followed by {top[1][0]} ({top[1][1]:.2f})" if len(top) > 1 else "")
            + "."
        )
        lines.append(
            f"- **Lowest-performing language**: "
            f"{bot[-1][0]} ({bot[-1][1]:.2f})"
            + " — consider investigating data quality, vocabulary coverage, and tokenisation."
        )

    # ASR insight
    wer_by_lang: dict[str, float] = {}
    for r in asr_rows:
        lang = r.get("language", "")
        try:
            wer_by_lang[lang] = min(float(r.get("WER", r.get("wer", 0)) or 0), wer_by_lang.get(lang, 999))
        except (TypeError, ValueError):
            pass
    if wer_by_lang:
        worst_asr = max(wer_by_lang.items(), key=lambda x: x[1])
        best_asr  = min(wer_by_lang.items(), key=lambda x: x[1])
        lines.append(
            f"- **ASR**: {best_asr[0]} achieves the lowest WER ({best_asr[1]:.1f}%); "
            f"{worst_asr[0]} shows the highest ({worst_asr[1]:.1f}%) — "
            "high WER languages likely suffer compounded errors in cascade pipelines."
        )

    # SOTA gap
    if sota_rows and lang_bleu:
        gaps = []
        for lang, our in lang_bleu.items():
            sota_vals = [float(r.get("BLEU","") or 0) for r in sota_rows
                         if r.get("language","").lower() == lang.lower()
                         and r.get("BLEU","")]
            sota_vals = [v for v in sota_vals if v > 0]
            if sota_vals:
                gaps.append((lang, our, max(sota_vals), our - max(sota_vals)))
        if gaps:
            avg_gap = sum(g[3] for g in gaps) / len(gaps)
            lines.append(
                f"- **SOTA gap**: Averaged across languages, our zero-shot baseline is "
                f"{abs(avg_gap):.2f} BLEU {'above' if avg_gap >= 0 else 'below'} the best published system. "
                + ("This suggests competitive zero-shot performance." if avg_gap >= 0
                   else "Fine-tuning (Paper 2) is expected to close this gap.")
            )
    lines.append(
        "- **Direction asymmetry**: Compare Source→English vs English→Source BLEU to assess "
        "whether translation quality is symmetric — a large gap indicates the model is better "
        "calibrated for one direction."
    )
    return "\n".join(lines)


def discussion_adaptation(text_rows, asr_rows, sota_rows) -> str:
    lines = [
        "- Compare pre- vs post-fine-tuning metrics in the tables above. "
        "Positive deltas confirm that LoRA adaptation improves over the zero-shot baseline.",
        "- Check whether fine-tuning on one language degrades others — "
        "a drop in unseen languages would indicate catastrophic forgetting.",
        "- The data scaling curves (if `SCALING_BUDGETS` was set) reveal sample efficiency: "
        "steep early gains suggest the model is data-hungry; a plateau indicates diminishing returns.",
        "- Compare fine-tuned cascade (Whisper+NLLB) vs fine-tuned end-to-end (SeamlessM4T): "
        "does targeted adaptation close the gap between architectures?",
    ]
    return "\n".join(lines)


def discussion_audio(text_rows, asr_rows, sota_rows) -> str:
    lines = [
        "- Compare `Direct audio` (raw waveform) vs `Normalised audio`: "
        "a BLEU gain for normalised inputs indicates the model is sensitive to energy distribution.",
        "- `Trimmed audio` (silence removed) vs direct: gains suggest leading/trailing silence "
        "introduces errors, particularly in low-SNR recordings.",
        "- `Chunk-based audio` (fixed segments): check whether BLEU improves for long utterances "
        "(>10 s) where the model's context window limits performance.",
        "- The text MT ceiling (gold transcript → translation) is an upper bound for the audio path. "
        "The gap between audio and text BLEU quantifies ASR contribution to overall error.",
        "- If chunk-based consistently outperforms direct audio, this supports a pre-segmentation "
        "recommendation for production deployment.",
    ]
    return "\n".join(lines)


def discussion_cascade(text_rows, asr_rows, sota_rows) -> str:
    lang_bleu_e2e     = _language_bleu(text_rows)
    lines = [
        "- Compare SeamlessM4T (end-to-end) BLEU vs Whisper+NLLB (cascade) BLEU "
        "for the same language and direction — this is the central figure for Paper 4.",
    ]
    if lang_bleu_e2e:
        lines.append(
            "- Oracle cascade (gold transcript → NLLB MT) represents the MT ceiling: "
            "if oracle BLEU > end-to-end BLEU, ASR errors are the bottleneck and "
            "better ASR would close the gap."
        )
    lines += [
        "- Error propagation: plot BLEU as a function of injected WER to find the "
        "break-even WER where cascade matches end-to-end performance.",
        "- Latency and VRAM measurements from `run_cascade_analysis.py` quantify "
        "the operational trade-off: cascade models may be faster on constrained hardware.",
        "- If ASR WER is high (>40%), the cascade pipeline loses its advantage and "
        "end-to-end models are preferred — reference specific language rows in the ASR table.",
    ]
    return "\n".join(lines)


def discussion_transfer(text_rows, asr_rows, sota_rows) -> str:
    lines = [
        "- Niger-Congo family (Yoruba, Igbo): compare BLEU scores against Afro-Asiatic (Hausa). "
        "A consistent gap across experiments would support a typological distance hypothesis.",
        "- Cross-lingual transfer experiments test whether fine-tuning on Yoruba helps Igbo "
        "(same family) more than it helps Hausa (different family).",
        "- Typological similarity scores (lang2vec cosine, from `transfer.py`) should correlate "
        "positively with transfer efficiency — compute Pearson r for this claim.",
        "- The few-shot scaling curves (samples vs BLEU) can reveal whether a language benefits "
        "more from a small amount of target-language data or from cross-lingual transfer.",
        "- Include the interaction regression coefficient (log samples × language family) to "
        "quantify whether family membership moderates data efficiency.",
    ]
    return "\n".join(lines)


DISCUSSION_FN = {
    "benchmark":  discussion_benchmark,
    "adaptation": discussion_adaptation,
    "audio":      discussion_audio,
    "cascade":    discussion_cascade,
    "transfer":   discussion_transfer,
}


def section_discussion(paper_mode: str, text_rows, asr_rows, sota_rows) -> str:
    fn = DISCUSSION_FN.get(paper_mode, lambda *_: "")
    body = fn(text_rows, asr_rows, sota_rows)
    if not body:
        return ""
    lines = ["### Key Observations for Paper Writing\n"]
    lines.append(
        "_The following points are derived from the metric tables above. "
        "They are starting points — verify each claim against your own tables before citing._\n"
    )
    lines.append(body)
    lines.append("")
    return "\n".join(lines)


def section_language_notes() -> str:
    return """\
### Language Coverage Notes

| Language | ISO | Family | FLEURS | African-Celtic | SeamlessM4T | Whisper |
|---|---|---|:---:|:---:|:---:|:---:|
| Yoruba | yor | Niger-Congo | ✓ | ✓ | ✓ | ✓ |
| Igbo | ibo | Niger-Congo | ✓ | ✓ | ✓ | — |
| Hausa | hau | Afro-Asiatic | ✓ | — | — | ✓ |
| Swahili | swh | Bantu | ✓ | — | ✓ | ✓ |

- **Igbo**: excluded from Whisper experiments — no Whisper language token
- **Hausa**: excluded from SeamlessM4T speech experiments — `hau` absent from S2TT language list; excluded from African-Celtic — no audio in that split
- **Swahili**: substitutes Hausa in FLEURS × SeamlessM4T to maintain 3-language parity
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main report assembler
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    paper_mode: str,
    consolidated_dir: Path,
    sota_file: str | None,
    output_path: Path,
) -> None:
    text_rows  = load_csv(consolidated_dir / "text_metrics_all.csv")
    asr_rows   = load_csv(consolidated_dir / "asr_metrics_all.csv")
    audio_rows = load_csv(consolidated_dir / "audio_metrics_all.csv")
    sota_rows  = load_sota(sota_file)

    mode_name   = PAPER_MODE_NAMES.get(paper_mode, paper_mode)
    experiments = sorted({label(r.get("experiment","")) for r in text_rows + asr_rows + audio_rows})
    languages   = sorted({r.get("language","") for r in text_rows + asr_rows + audio_rows if r.get("language")})
    run_date    = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = []

    # ── Header ────────────────────────────────────────────────────────────────
    sections.append(f"""\
# LinguoMT Results Report — {mode_name}

**Generated:** {run_date}
**Experiments:** {", ".join(experiments) if experiments else "—"}
**Languages:** {", ".join(languages) if languages else "—"}
**SOTA file:** {sota_file or "not provided"}

> **Important — read before using this report in a paper:**
>
> This document is auto-generated from experiment metric CSVs to **give direction for writing**.
> It provides a starting structure, highlights patterns in the data, and flags gaps versus published
> SOTA systems. It is **not a finished paper section**.
>
> Before citing any number or claim in this report you must:
> - Verify every metric against the raw output files in `outputs/*/metrics/`
> - Conduct a systematic review of the relevant SOTA literature (not just the CSV rows provided)
> - Critically assess the discussion observations — they are data-driven heuristics, not conclusions
> - Fill all `[NARRATIVE:...]` sections through your own analysis and expert judgement
> - Cross-check SOTA comparisons against the original papers, not just this CSV summary
>
> The section headers, table structure, and observation bullets are scaffolding.
> The intellectual contribution — interpretation, discussion, and conclusions — is yours.

---
""")

    # ── Language notes ─────────────────────────────────────────────────────
    sections.append("## Experiment Overview\n\n" + section_language_notes())

    # ── Results ───────────────────────────────────────────────────────────────
    sections.append("## Results\n")
    sections.append(section_text(text_rows))
    sections.append(section_asr(asr_rows))
    if audio_rows:
        sections.append(section_audio(audio_rows))

    # ── Cross-experiment comparison ────────────────────────────────────────────
    if len(experiments) > 1:
        sections.append("## Comparisons\n")
        sections.append(section_cross_experiment(text_rows, asr_rows))

    # ── SOTA ──────────────────────────────────────────────────────────────────
    if sota_rows:
        sections.append("## SOTA Comparison\n")
        sections.append(section_sota(text_rows, sota_rows))

    # ── Discussion ────────────────────────────────────────────────────────────
    sections.append("## Discussion\n")
    sections.append(section_discussion(paper_mode, text_rows, asr_rows, sota_rows))

    # Add paper-mode-specific narrative prompts
    narrative_prompts = {
        "benchmark": [
            "[NARRATIVE: Introduce the zero-shot evaluation setup and motivate the choice of models and datasets.]",
            "[NARRATIVE: Discuss why certain languages perform better — consider script type, training data volume, and typological distance from English.]",
            "[NARRATIVE: Compare FLEURS vs African-Celtic results: do models perform consistently across domains?]",
        ],
        "adaptation": [
            "[NARRATIVE: Describe the LoRA fine-tuning setup and hyperparameters used.]",
            "[NARRATIVE: Discuss which languages benefit most from adaptation and hypothesize why.]",
            "[NARRATIVE: Analyse the data scaling curves and estimate a recommended minimum training budget.]",
        ],
        "audio": [
            "[NARRATIVE: Explain the audio preprocessing strategies and the conditions under which each is expected to help.]",
            "[NARRATIVE: Discuss the ASR-MT gap: how much of the overall error comes from ASR vs MT?]",
            "[NARRATIVE: Provide recommendations for deployment: when to prefer normalisation, trimming, or chunking.]",
        ],
        "cascade": [
            "[NARRATIVE: Introduce the oracle cascade experiment and what it reveals about the MT ceiling.]",
            "[NARRATIVE: Discuss the break-even WER analysis: at what ASR quality does cascade match end-to-end?]",
            "[NARRATIVE: Analyse latency and memory trade-offs for practical deployment decisions.]",
        ],
        "transfer": [
            "[NARRATIVE: Introduce the typological distance hypothesis and the language families used.]",
            "[NARRATIVE: Discuss whether cross-lingual transfer follows family boundaries and what exceptions you observe.]",
            "[NARRATIVE: Propose directions for future work in zero-shot transfer for low-resource African languages.]",
        ],
    }
    prompts = narrative_prompts.get(paper_mode, [])
    if prompts:
        sections.append("### Narrative Sections (Manual)\n")
        sections.append("_Fill in the sections below when writing the paper:_\n")
        for p in prompts:
            sections.append(f"\n{p}\n")
        sections.append("")

    # ── Raw data appendix ─────────────────────────────────────────────────────
    sections.append("---\n\n## Appendix — Raw Metric Tables\n")
    if text_rows:
        sections.append("### Full Text Metrics\n")
        all_cols = list(dict.fromkeys(k for r in text_rows for k in r))
        sections.append(_md_table(all_cols, [[r.get(c,"") for c in all_cols] for r in text_rows]))
        sections.append("")
    if asr_rows:
        sections.append("### Full ASR Metrics\n")
        all_cols = list(dict.fromkeys(k for r in asr_rows for k in r))
        sections.append(_md_table(all_cols, [[r.get(c,"") for c in all_cols] for r in asr_rows]))
        sections.append("")
    if audio_rows:
        sections.append("### Full Audio Metrics\n")
        all_cols = list(dict.fromkeys(k for r in audio_rows for k in r))
        sections.append(_md_table(all_cols, [[r.get(c,"") for c in all_cols] for r in audio_rows]))
        sections.append("")

    report = "\n".join(sections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report written: {output_path}")
    print(f"  {len(text_rows)} text rows, {len(asr_rows)} ASR rows, "
          f"{len(audio_rows)} audio rows, {len(sota_rows)} SOTA rows")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

PAPER_MODE_CHOICES = list(PAPER_MODE_NAMES.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "paper_mode",
        choices=PAPER_MODE_CHOICES,
        help="Paper mode: benchmark | adaptation | audio | cascade | transfer",
    )
    parser.add_argument(
        "--consolidated-dir", "-d",
        default=None,
        help="Path to consolidated metrics directory (default: most recent /content/outputs/consolidated_*/)",
    )
    parser.add_argument(
        "--sota-file", "-s",
        default=None,
        help="Path to SOTA CSV (columns: system, language, BLEU, venue, year)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output Markdown path (default: papers/<paper_id>/results_report.md)",
    )
    args = parser.parse_args()

    # Resolve consolidated dir
    if args.consolidated_dir:
        consolidated_dir = Path(args.consolidated_dir)
    else:
        outputs_root = Path("/content/outputs")
        if not outputs_root.exists():
            outputs_root = REPO_ROOT / "outputs"
        consolidated_dir = find_consolidated(outputs_root)
        if consolidated_dir is None:
            print(
                f"No consolidated directory found under {outputs_root}.\n"
                "Run Step 6 (consolidate) in the Colab notebook first, or pass --consolidated-dir.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Consolidated dir : {consolidated_dir}")

    # Resolve output path
    paper_id_map = {
        "benchmark":  "paper1_benchmark",
        "adaptation": "paper2_adaptation",
        "audio":      "paper3_audio",
        "cascade":    "paper4_cascade",
        "transfer":   "paper5_transfer",
    }
    paper_id = paper_id_map[args.paper_mode]
    output_path = Path(args.output) if args.output else PAPERS_DIR / paper_id / "results_report.md"

    generate_report(args.paper_mode, consolidated_dir, args.sota_file, output_path)


if __name__ == "__main__":
    main()
