"""
SOTA comparison module for the LinguoMT publication series.

Two complementary data sources are supported per paper folder:

1. sota_results.csv / published_baselines.json
   Flat tabular format — one row per system × language × metric result.
   Used for bulk data entry and direct comparison table generation.

2. references.yaml
   YAML list format with standard bibliographic fields PLUS comparison fields
   (model, dataset, language, direction, metric, score, summary) that store
   comparison metadata directly on each reference entry.
   Validated at output time; never blocks model training or fine-tuning.

Both are validated against schema.json (required vs optional fields, including
a reference_fields section that declares the schema for references.yaml entries).
Validation is always advisory — warnings are printed but never raised as errors.

YAML reference fields (no prefix — all are plain field names):
  model      — model name used for evaluation
  datasets   — list of dataset names, e.g. [FLEURS, AfriSpeech-200]
  language   — display language name (e.g. Yoruba)
  directions — list of task directions, e.g. [Source → English, ASR]
  metrics    — list of reported metrics, e.g. [BLEU, WER]; score matches metrics[0]
  score      — numeric score for metrics[0] (null if unknown — excluded from comparison tables)
  summary    — 2–5 sentences: what the paper does, why it is relevant
  notes      — evaluation conditions, split, caveats (optional)
  + paper-specific fields declared in schema.json reference_fields section
"""
from __future__ import annotations

import json
import re
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

VALID_METRICS    = {"BLEU", "spBLEU", "ChrF", "WER", "CER", "CHRF", "CHRF++"}
VALID_DIRECTIONS = {"Source → English", "English → Source", "ASR",
                    "Text MT (Source → English)", "Text MT (English → Source)"}


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


# ── Validator (SOTA CSV/JSON) ─────────────────────────────────────────────────

def validate_sota(df: pd.DataFrame, schema: dict) -> list[str]:
    """
    Validate a loaded SOTA DataFrame against a schema.
    Returns a list of warning strings (empty = all good).
    """
    warnings: list[str] = []
    required = schema.get("required_fields", BASE_REQUIRED_FIELDS)

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        warnings.append(f"[SOTA] Missing required columns: {missing_cols}")

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

    if "score" in df.columns:
        non_numeric = pd.to_numeric(df["score"], errors="coerce").isnull().sum()
        if non_numeric > 0:
            warnings.append(
                f"[SOTA] 'score' is non-numeric in {non_numeric} row(s). Those rows will be skipped."
            )

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
    Load SOTA baselines from a CSV, JSON, or YAML (.yaml) file.
    Validates against schema (auto-loaded from the file's parent folder if not provided).
    Prints warnings for missing required fields; drops invalid rows.
    Returns empty DataFrame if file is missing.
    Never raises; never blocks training or fine-tuning.
    """
    p = Path(path)
    if not p.exists():
        print(f"[SOTA] File not found: {p}  — skipping SOTA comparison.")
        return pd.DataFrame(columns=SOTA_COLUMNS)

    if p.suffix.lower() in (".yaml", ".yml"):
        return load_yaml_references(p.parent, schema)

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


# ── YAML reference parser ─────────────────────────────────────────────────────

# Scalar YAML fields → DataFrame column names
_YAML_SCALAR_MAP = {
    "author":   "authors",
    "title":    "paper_title",
    "year":     "year",
    "model":    "model",
    "language": "language",
    "score":    "score",
    "notes":    "notes",
}

# List YAML fields → DataFrame column names
# Each list is joined as ", " for display; first element is used for metric filtering.
_YAML_LIST_MAP = {
    "datasets":   "dataset",
    "directions": "direction",
    "metrics":    "metric",
}

# Required standard fields for every reference entry
_YAML_STD_REQUIRED = {"citation_key", "type", "author", "title", "year"}
# Required comparison fields for entries to appear in comparison tables
_YAML_COMPARISON_REQUIRED = {"model", "datasets", "language", "directions", "metrics", "summary"}


def _load_yaml_entries(yaml_path: Path) -> list[dict]:
    """
    Load a list of YAML reference entries. Uses PyYAML if available,
    otherwise prints a warning and returns an empty list.
    """
    try:
        import yaml  # PyYAML
    except ImportError:
        print(
            "[REF] PyYAML is not installed. Run `pip install pyyaml` to enable "
            "references.yaml loading. SOTA comparison will be skipped."
        )
        return []

    try:
        with yaml_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        print(f"[REF] Could not parse {yaml_path}: {exc} — skipping.")
        return []

    if not isinstance(data, list):
        print(
            f"[REF] {yaml_path.name}: expected a YAML list of entries (starting with '- '), "
            f"got {type(data).__name__}. Skipping."
        )
        return []

    return [e for e in data if isinstance(e, dict)]


def validate_references(entries: list[dict], schema: dict) -> list[str]:
    """
    Validate a list of YAML reference entries against a schema.
    Returns warning strings. Never raises; never blocks training/fine-tuning.

    Checks per entry:
      - Required standard fields (citation_key, type, author, title, year)
      - Required comparison fields (model, datasets, language, directions, metrics, summary)
        where datasets/directions/metrics must be non-empty lists
      - score is numeric or null
      - Each value in metrics[] is a known metric name
      - Paper-specific required fields from schema reference_fields section
    """
    ref_cfg        = schema.get("reference_fields", {})
    extra_req      = set(ref_cfg.get("paper_specific_required", []))
    comparison_req = _YAML_COMPARISON_REQUIRED | extra_req
    warnings: list[str] = []

    for e in entries:
        key = e.get("citation_key", "?")

        # Standard scalar fields
        for f in _YAML_STD_REQUIRED:
            val = e.get(f)
            if val is None or str(val).strip() == "":
                warnings.append(f"[REF] Entry '{key}': missing standard field '{f}'")

        # Comparison fields — list fields must be non-empty lists
        for f in comparison_req:
            val = e.get(f)
            if f in _YAML_LIST_MAP:
                if not isinstance(val, list) or len(val) == 0:
                    warnings.append(
                        f"[REF] Entry '{key}': '{f}' must be a non-empty list "
                        f"(e.g. {f}: [BLEU]) — entry will be excluded from comparison tables"
                    )
            elif f == "summary":
                continue  # summary is informational, doesn't block row
            else:
                if val is None or str(val).strip() == "":
                    warnings.append(
                        f"[REF] Entry '{key}': missing comparison field '{f}' "
                        f"— entry will be excluded from comparison tables"
                    )

        # Score must be numeric if not null
        score_raw = e.get("score")
        if score_raw is not None:
            try:
                float(score_raw)
            except (ValueError, TypeError):
                warnings.append(
                    f"[REF] Entry '{key}': score='{score_raw}' is not numeric. "
                    f"Set to null if unknown."
                )

        # Each metric in metrics[] should be a known value
        metrics_raw = e.get("metrics")
        if isinstance(metrics_raw, list):
            for m in metrics_raw:
                m_str = str(m).strip()
                if m_str and m_str.upper() not in {mv.upper() for mv in VALID_METRICS}:
                    warnings.append(
                        f"[REF] Entry '{key}': unrecognised metric '{m_str}' in metrics[]. "
                        f"Expected one of {sorted(VALID_METRICS)}."
                    )

    return warnings


def _yaml_entry_to_row(entry: dict, paper_specific_req: set[str]) -> dict | None:
    """
    Convert a YAML reference entry to a DataFrame row.
    List fields (datasets, directions, metrics) are joined as ", " strings;
    the first element of metrics[] is used as the primary metric (must match score).
    Returns None if any required comparison field is missing or score is non-numeric/null.
    """
    row: dict = {"citation_key": entry.get("citation_key", "")}

    # Scalar fields
    for yaml_field, col_name in _YAML_SCALAR_MAP.items():
        row[col_name] = entry.get(yaml_field, "")

    # List fields: join for display, first element for primary filtering
    for yaml_field, col_name in _YAML_LIST_MAP.items():
        val = entry.get(yaml_field)
        if isinstance(val, list) and val:
            row[col_name] = ", ".join(str(v) for v in val)
        elif val is not None:
            row[col_name] = str(val)
        else:
            row[col_name] = ""

    # Score: null → skip from comparison table
    score_raw = entry.get("score")
    if score_raw is None:
        return None
    try:
        row["score"] = float(score_raw)
    except (ValueError, TypeError):
        return None

    # Required comparison fields must be present (list fields need non-empty lists)
    all_required = _YAML_COMPARISON_REQUIRED | paper_specific_req
    for f in all_required:
        if f == "summary":
            continue
        if f in _YAML_LIST_MAP:
            val = entry.get(f)
            if not isinstance(val, list) or len(val) == 0:
                return None
        else:
            col = _YAML_SCALAR_MAP.get(f, f)
            if not str(row.get(col, "")).strip():
                return None

    # Copy summary into notes if no explicit notes provided
    summary = str(entry.get("summary", "")).strip()
    notes   = str(row.get("notes", "")).strip()
    if summary and not notes:
        row["notes"] = summary[:200]

    return row


def load_yaml_references(folder: str | Path, schema: Optional[dict] = None) -> pd.DataFrame:
    """
    Load references.yaml from a paper folder.
    Validates structure (warn-only). Returns a DataFrame compatible with load_sota().
    Entries with null score or missing required comparison fields are excluded
    from the returned DataFrame but remain valid bibliography entries.
    Never raises; never blocks training or fine-tuning.
    """
    folder = Path(folder)
    yaml_path = folder / "references.yaml"
    if not yaml_path.exists():
        return pd.DataFrame(columns=SOTA_COLUMNS)

    entries = _load_yaml_entries(yaml_path)
    if not entries:
        return pd.DataFrame(columns=SOTA_COLUMNS)

    if schema is None:
        schema = load_schema(folder)

    warnings = validate_references(entries, schema)
    for w in warnings:
        print(w)

    ref_cfg = schema.get("reference_fields", {})
    paper_specific_req = set(ref_cfg.get("paper_specific_required", []))

    rows = [
        r for e in entries
        if (r := _yaml_entry_to_row(e, paper_specific_req)) is not None
    ]

    if not rows:
        n = len(entries)
        print(
            f"[REF] {yaml_path.name}: {n} entries parsed, "
            f"0 usable for comparison (score is null or required fields missing). "
            f"Fill in 'score' for each entry to enable comparison tables."
        )
        return pd.DataFrame(columns=SOTA_COLUMNS)

    df = pd.DataFrame(rows)
    for col in SOTA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    print(f"[REF] {yaml_path.name}: {len(entries)} entries, {len(df)} usable for comparison.")
    return df[SOTA_COLUMNS].copy()


# ── Backwards-compatibility aliases ──────────────────────────────────────────

def load_bib_references(folder: str | Path, schema: Optional[dict] = None) -> pd.DataFrame:
    """Deprecated alias → load_yaml_references (YAML format replaced BibTeX)."""
    print("[REF] load_bib_references is deprecated — use load_yaml_references instead.")
    return load_yaml_references(folder, schema)


def validate_bib(entries: list[dict], schema: dict) -> list[str]:
    """Deprecated alias → validate_references."""
    return validate_references(entries, schema)


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
    """
    sota_sub = sota_df[sota_df["metric"].str.upper() == metric.upper()]
    if sota_sub.empty or our_results.empty or metric not in our_results.columns:
        return pd.DataFrame()

    rows = []
    langs = our_results["language"].dropna().unique() if "language" in our_results.columns else []
    for lang in langs:
        our_sub       = our_results[our_results["language"] == lang]
        our_max       = our_sub[metric].max() if not our_sub.empty else float("nan")
        sota_sub_lang = sota_sub[sota_sub["language"].str.lower() == _norm(lang)]
        if sota_sub_lang.empty:
            rows.append({"language": lang, "our_best": our_max,
                         "sota_best": float("nan"), "gap": float("nan"),
                         "sota_model": "", "citation_key": ""})
            continue
        sota_max_idx = sota_sub_lang["score"].astype(float).idxmax()
        sota_row     = sota_sub_lang.loc[sota_max_idx]
        sota_max     = float(sota_row["score"])
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
    """
    rows = []

    if not our_results.empty and metric in our_results.columns:
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
