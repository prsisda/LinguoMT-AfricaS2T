"""
SOTA comparison module for the LinguoMT publication series.

Load published baselines from sota_results.csv or published_baselines.json
and generate comparison tables against our system's results.

Schema (per row / entry):
    paper_title    str  — e.g. "SeamlessM4T: Massively Multilingual & Multimodal Machine Translation"
    authors        str  — e.g. "Barrault et al."
    year           int  — e.g. 2023
    model          str  — e.g. "SeamlessM4T-v2-large"
    dataset        str  — e.g. "FLEURS" | "African-Celtic" | "CommonVoice"
    language       str  — display name, e.g. "Yoruba"
    direction      str  — e.g. "Source → English" | "English → Source"
    metric         str  — "BLEU" | "ChrF" | "WER" | "CER"
    score          float
    citation_key   str  — BibTeX key, e.g. "barrault2023seamless"
    notes          str  — optional free text
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SOTA_COLUMNS = [
    "paper_title", "authors", "year", "model", "dataset",
    "language", "direction", "metric", "score", "citation_key", "notes",
]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_sota(path: str | Path) -> pd.DataFrame:
    """Load SOTA baselines from a CSV or JSON file. Returns empty DataFrame if file missing."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=SOTA_COLUMNS)
    if p.suffix.lower() == ".json":
        return _load_json(p)
    return _load_csv(p)


def _load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    for col in SOTA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[SOTA_COLUMNS].copy()


def _load_json(p: Path) -> pd.DataFrame:
    with p.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and "baselines" in data:
        rows = data["baselines"]
    else:
        rows = []
    df = pd.DataFrame(rows)
    for col in SOTA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[SOTA_COLUMNS].copy()


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _norm(s) -> str:
    return str(s or "").strip().lower()


def _match(our_row: pd.Series, sota_row: pd.Series, metric: str) -> bool:
    lang_match = _norm(our_row.get("language")) == _norm(sota_row.get("language"))
    dir_match  = _norm(our_row.get("direction_label")) == _norm(sota_row.get("direction"))
    met_match  = _norm(sota_row.get("metric")) == metric.lower()
    return lang_match and dir_match and met_match


# ── Comparison tables ─────────────────────────────────────────────────────────

def build_sota_comparison_table(
    our_results: pd.DataFrame,
    sota_df: pd.DataFrame,
    metric: str = "BLEU",
    dataset_filter: str | None = None,
) -> pd.DataFrame:
    """
    T_SOTA1 — our best result per language×direction vs best published SOTA.

    Columns: language | direction | our_score | sota_score | delta | sota_model | citation_key
    """
    if our_results.empty or sota_df.empty:
        return pd.DataFrame()

    sota_sub = sota_df[sota_df["metric"].str.upper() == metric.upper()].copy()
    if dataset_filter:
        sota_sub = sota_sub[sota_sub["dataset"].str.lower() == dataset_filter.lower()]

    # Best score per language × direction from our system (latest/max experiment)
    key_cols = [c for c in ["language", "direction_label"] if c in our_results.columns]
    if not key_cols or metric not in our_results.columns:
        return pd.DataFrame()

    our_best = (
        our_results.groupby(key_cols)[metric]
        .max()
        .reset_index()
        .rename(columns={"direction_label": "direction", metric: "our_score"})
    )

    rows = []
    for _, our in our_best.iterrows():
        mask = (
            (sota_sub["language"].str.lower() == _norm(our["language"])) &
            (sota_sub["direction"].str.lower() == _norm(our.get("direction", "")))
        )
        matching = sota_sub[mask]
        if matching.empty:
            rows.append({
                "language":    our["language"],
                "direction":   our.get("direction", ""),
                "our_score":   our["our_score"],
                "sota_score":  float("nan"),
                "delta":       float("nan"),
                "sota_model":  "",
                "citation_key": "",
            })
        else:
            best_sota = matching.loc[matching["score"].astype(float).idxmax()]
            sota_score = float(best_sota["score"])
            rows.append({
                "language":    our["language"],
                "direction":   our.get("direction", ""),
                "our_score":   float(our["our_score"]),
                "sota_score":  sota_score,
                "delta":       float(our["our_score"]) - sota_score,
                "sota_model":  best_sota.get("model", ""),
                "citation_key": best_sota.get("citation_key", ""),
            })

    return pd.DataFrame(rows)


def build_gap_table(
    our_results: pd.DataFrame,
    sota_df: pd.DataFrame,
    metric: str = "BLEU",
) -> pd.DataFrame:
    """
    T_SOTA2 — language-specific SOTA gap: best published, our score, gap, rank.

    Useful for LinguoMT-Benchmark and LinguoMT-Transfer.
    """
    sota_sub = sota_df[sota_df["metric"].str.upper() == metric.upper()]
    if sota_sub.empty or our_results.empty or metric not in our_results.columns:
        return pd.DataFrame()

    rows = []
    langs = our_results["language"].dropna().unique() if "language" in our_results.columns else []
    for lang in langs:
        our_sub  = our_results[our_results["language"] == lang]
        our_max  = our_sub[metric].max() if not our_sub.empty else float("nan")
        sota_sub_lang = sota_sub[sota_sub["language"].str.lower() == _norm(lang)]
        if sota_sub_lang.empty:
            rows.append({"language": lang, "our_best": our_max,
                         "sota_best": float("nan"), "gap": float("nan"),
                         "sota_model": "", "citation_key": ""})
            continue
        sota_max_idx = sota_sub_lang["score"].astype(float).idxmax()
        sota_row = sota_sub_lang.loc[sota_max_idx]
        sota_max = float(sota_row["score"])
        rows.append({
            "language":    lang,
            "our_best":    float(our_max),
            "sota_best":   sota_max,
            "gap":         float(our_max) - sota_max,
            "sota_model":  sota_row.get("model", ""),
            "citation_key": sota_row.get("citation_key", ""),
        })

    df = pd.DataFrame(rows)
    if not df.empty and "gap" in df.columns:
        df = df.sort_values("gap", ascending=False)
    return df


def build_system_ranking_table(
    our_label: str,
    our_results: pd.DataFrame,
    sota_df: pd.DataFrame,
    metric: str = "BLEU",
) -> pd.DataFrame:
    """
    T_SOTA3 — all systems ranked by metric score across languages.

    Combines our result rows with SOTA rows into a single ranked table.
    Useful for LinguoMT-Cascade and LinguoMT-Benchmark.
    """
    rows = []

    if not our_results.empty and metric in our_results.columns:
        key_cols = [c for c in ["language", "direction_label", "experiment"] if c in our_results.columns]
        for _, r in our_results.iterrows():
            rows.append({
                "system":    our_label,
                "model":     our_label,
                "dataset":   r.get("dataset_name", ""),
                "language":  r.get("language", ""),
                "direction": r.get("direction_label", ""),
                "metric":    metric,
                "score":     r.get(metric, float("nan")),
                "source":    "ours",
                "citation_key": "",
            })

    sota_sub = sota_df[sota_df["metric"].str.upper() == metric.upper()]
    for _, r in sota_sub.iterrows():
        rows.append({
            "system":    r.get("paper_title", r.get("model", "")),
            "model":     r.get("model", ""),
            "dataset":   r.get("dataset", ""),
            "language":  r.get("language", ""),
            "direction": r.get("direction", ""),
            "metric":    metric,
            "score":     float(r.get("score", float("nan"))),
            "source":    "published",
            "citation_key": r.get("citation_key", ""),
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def build_improvement_table(
    before_results: pd.DataFrame,
    after_results: pd.DataFrame,
    metric: str = "BLEU",
) -> pd.DataFrame:
    """
    T_SOTA4 — metric improvement: pretrained → fine-tuned (or baseline → our system).

    Columns: language | direction | before | after | delta | pct_change
    """
    if before_results.empty or after_results.empty or metric not in before_results.columns:
        return pd.DataFrame()

    key_cols = [c for c in ["language", "direction_label"] if c in before_results.columns]
    bf = before_results.groupby(key_cols)[metric].mean().reset_index().rename(
        columns={metric: "before", "direction_label": "direction"}
    )
    af = after_results.groupby(key_cols)[metric].mean().reset_index().rename(
        columns={metric: "after", "direction_label": "direction"}
    )
    merged = bf.merge(af, on=[c if c != "direction_label" else "direction" for c in key_cols], how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["delta"]      = merged["after"] - merged["before"]
    merged["pct_change"] = ((merged["delta"] / merged["before"].replace(0, float("nan"))) * 100).round(1)
    return merged.sort_values("delta", ascending=False).reset_index(drop=True)


def generate_sota_interpretation(
    sota_cmp: pd.DataFrame,
    gap_table: pd.DataFrame,
    model_name: str,
    metric: str = "BLEU",
) -> str:
    lines = [f"## State-of-the-Art Comparison ({metric})\n"]

    if not sota_cmp.empty and "delta" in sota_cmp.columns:
        above = (sota_cmp["delta"] > 0).sum()
        below = (sota_cmp["delta"] < 0).sum()
        lines.append(f"**{model_name}** outperforms published SOTA in "
                     f"{above}/{len(sota_cmp)} language×direction settings.")
        if below > 0:
            lines.append(f"Underperforms SOTA in {below} settings — see gap table for details.\n")
        best = sota_cmp.loc[sota_cmp["delta"].idxmax()] if not sota_cmp.empty else None
        if best is not None and not pd.isna(best["delta"]):
            lines.append(f"Largest improvement: **{best['language']}** "
                         f"({best.get('direction','')}): "
                         f"+{best['delta']:.2f} {metric} over {best.get('sota_model','SOTA')}")
        lines.append("")

    if not gap_table.empty and "gap" in gap_table.columns:
        lines.append("**Language-specific SOTA gap:**")
        for _, r in gap_table.iterrows():
            sign = "+" if r["gap"] >= 0 else ""
            lines.append(f"- {r['language']}: {sign}{r['gap']:.2f} {metric} "
                         f"(our: {r['our_best']:.2f}, SOTA: {r['sota_best']:.2f} "
                         f"[{r.get('citation_key', '')}])")
        lines.append("")

    return "\n".join(lines)
