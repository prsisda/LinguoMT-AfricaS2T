"""
SOTA comparison module for the LinguoMT publication series.

Each paper folder under sota/ contains a schema.json that declares which fields
are required and which are optional. load_sota() validates the loaded data against
that schema and prints warnings for missing required values.

Base required fields (all papers):
    paper_title, authors, year, model, dataset, language, direction,
    metric, score, citation_key

Base optional fields (all papers):
    notes

Paper-specific fields are declared in each folder's schema.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


# ── Base schema (shared across all papers) ────────────────────────────────────

BASE_REQUIRED_FIELDS = [
    "paper_title", "authors", "year", "model", "dataset",
    "language", "direction", "metric", "score", "citation_key",
]
BASE_OPTIONAL_FIELDS = ["notes"]
SOTA_COLUMNS = BASE_REQUIRED_FIELDS + BASE_OPTIONAL_FIELDS

VALID_METRICS    = {"BLEU", "ChrF", "WER", "CER", "CHRF", "CHRF++"}
VALID_DIRECTIONS = {"Source → English", "English → Source"}


# ── Schema loader ─────────────────────────────────────────────────────────────

def load_schema(folder: str | Path) -> dict:
    """
    Load schema.json from a paper folder.
    Returns the base schema if no schema.json exists.
    """
    schema_path = Path(folder) / "schema.json"
    if not schema_path.exists():
        return {
            "required_fields": BASE_REQUIRED_FIELDS,
            "optional_fields": BASE_OPTIONAL_FIELDS,
        }
    with schema_path.open() as f:
        raw = json.load(f)
    required = BASE_REQUIRED_FIELDS + raw.get("paper_specific_required", [])
    optional = BASE_OPTIONAL_FIELDS + raw.get("paper_specific_optional", [])
    return {**raw, "required_fields": required, "optional_fields": optional}


# ── Validator ─────────────────────────────────────────────────────────────────

def validate_sota(df: pd.DataFrame, schema: dict) -> list[str]:
    """
    Validate a loaded SOTA DataFrame against a schema.
    Returns a list of warning strings (empty = all good).
    Rows with missing required fields are dropped from the returned DataFrame
    via filter_valid_sota().
    """
    warnings: list[str] = []
    required = schema.get("required_fields", BASE_REQUIRED_FIELDS)

    # Missing columns entirely
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        warnings.append(f"[SOTA] Missing required columns: {missing_cols}")

    # Rows with empty required fields
    for col in required:
        if col not in df.columns:
            continue
        empty_mask = df[col].isnull() | (df[col].astype(str).str.strip() == "") | (df[col].astype(str) == "None")
        n_empty = empty_mask.sum()
        if n_empty > 0:
            warnings.append(
                f"[SOTA] Column '{col}' is required but empty in {n_empty} row(s). "
                f"These rows will be skipped."
            )

    # Score column must be numeric
    if "score" in df.columns:
        non_numeric = pd.to_numeric(df["score"], errors="coerce").isnull().sum()
        if non_numeric > 0:
            warnings.append(
                f"[SOTA] 'score' is non-numeric in {non_numeric} row(s). Those rows will be skipped."
            )

    # Metric values should be known
    if "metric" in df.columns:
        unknown = df["metric"].dropna()
        unknown = unknown[~unknown.str.upper().isin({m.upper() for m in VALID_METRICS})]
        if not unknown.empty:
            warnings.append(
                f"[SOTA] Unrecognised metric value(s): {unknown.unique().tolist()}. "
                f"Expected one of {sorted(VALID_METRICS)}."
            )

    return warnings


def filter_valid_sota(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Drop rows that are missing any required field or have a non-numeric score."""
    required = schema.get("required_fields", BASE_REQUIRED_FIELDS)
    mask = pd.Series([True] * len(df), index=df.index)
    for col in required:
        if col not in df.columns:
            return pd.DataFrame(columns=df.columns)
        empty = df[col].isnull() | (df[col].astype(str).str.strip() == "") | (df[col].astype(str) == "None")
        mask &= ~empty
    if "score" in df.columns:
        mask &= pd.to_numeric(df["score"], errors="coerce").notnull()
    return df[mask].reset_index(drop=True)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_sota(path: str | Path, schema: Optional[dict] = None) -> pd.DataFrame:
    """
    Load SOTA baselines from a CSV or JSON file.
    Validates against schema (auto-loaded from the file's parent folder if not provided).
    Prints warnings for missing required fields; drops invalid rows.
    Returns empty DataFrame if file is missing.
    """
    p = Path(path)
    if not p.exists():
        print(f"[SOTA] File not found: {p}  — skipping SOTA comparison.")
        return pd.DataFrame(columns=SOTA_COLUMNS)

    df = _load_json(p) if p.suffix.lower() == ".json" else _load_csv(p)

    if schema is None:
        schema = load_schema(p.parent)

    warnings = validate_sota(df, schema)
    for w in warnings:
        print(w)

    df = filter_valid_sota(df, schema)
    if df.empty:
        print("[SOTA] No valid rows found after validation. Check required fields.")
    else:
        print(f"[SOTA] Loaded {len(df)} valid baseline row(s) from {p.name}")

    return df


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
