from __future__ import annotations
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset, Audio as HFAudio


def _load_no_torchcodec(dataset_id: str, config: str, split: str):
    """Load a streaming dataset with audio decoding disabled (avoids torchcodec/FFmpeg)."""
    ds = load_dataset(dataset_id, config, split=split, streaming=True, trust_remote_code=False)
    try:
        ds = ds.cast_column("audio", HFAudio(decode=False))
    except Exception:
        pass
    return ds


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def _get_text(item: dict) -> str:
    return str(
        item.get("transcription") or item.get("raw_transcription") or
        item.get("text") or item.get("sentence") or ""
    ).strip()


class DatasetCache:
    """
    Loads aligned (source, english) pairs for all languages in a single pass.
    Supports FLEURS (config-per-language) and African-Celtic (filter-by-language-field).
    Call build() once; then get_pairs(language_key, max_samples) repeatedly.
    """

    def __init__(
        self,
        dataset_id: str,
        adapter_type: str,
        language_configs: list[dict],
        split: str,
        max_pairs: int,
        max_scan_rows: int,
        cache_dir: Path | None = None,
        force_rerun: bool = False,
    ) -> None:
        self.dataset_id       = dataset_id
        self.adapter_type     = adapter_type
        self.language_configs = language_configs
        self.split            = split
        self.max_pairs        = max_pairs
        self.max_scan_rows    = max_scan_rows
        self.cache_dir        = cache_dir
        self.force_rerun      = force_rerun
        self._cache: dict[str, list[dict]] = {}

    # ── public ───────────────────────────────────────────────────────────────

    def build(self, monitor=None) -> None:
        if self.adapter_type == "fleurs":
            self._build_fleurs(monitor)
        elif self.adapter_type == "african_celtic":
            self._build_african_celtic(monitor)
        else:
            raise ValueError(f"Unknown adapter: {self.adapter_type}")

    def get_pairs(self, language_key: str, max_samples: int | None = None) -> list[dict]:
        pairs = self._cache.get(language_key, [])
        return pairs[:max_samples] if max_samples is not None else pairs

    def stats(self) -> dict[str, int]:
        return {lk: len(v) for lk, v in self._cache.items()}

    # ── FLEURS ───────────────────────────────────────────────────────────────

    def _build_fleurs(self, monitor) -> None:
        for cfg in self.language_configs:
            lk = cfg["language_key"]
            fc = cfg.get("fleurs_config")
            if not fc:
                self._cache[lk] = []
                continue
            if monitor:
                monitor.step(f"Loading FLEURS [{fc}]", self.split)
            pairs: list[dict] = []
            try:
                en_stream  = _load_no_torchcodec(self.dataset_id, "en_us", self.split)
                src_stream = _load_no_torchcodec(self.dataset_id, fc,      self.split)
                for i, (en_item, src_item) in enumerate(zip(en_stream, src_stream)):
                    if i >= self.max_scan_rows or len(pairs) >= self.max_pairs:
                        break
                    eng_text = _norm(_get_text(en_item))
                    src_text = _norm(_get_text(src_item))
                    if not eng_text or not src_text:
                        continue
                    pairs.append({
                        "pair_idx":  len(pairs),
                        "src_text":  src_text,
                        "eng_text":  eng_text,
                        "src_audio": src_item.get("audio"),
                        "eng_audio": en_item.get("audio"),
                    })
            except Exception as e:
                if monitor:
                    monitor.step(f"  Error [{fc}]", str(e)[:120])
            self._cache[lk] = pairs
            if monitor:
                monitor.step(f"  {cfg['display']} cached", f"{len(pairs)} pairs")

    # ── African-Celtic ────────────────────────────────────────────────────────

    def _build_african_celtic(self, monitor) -> None:
        if monitor:
            monitor.step("Single-pass African-Celtic scan", self.split)
        all_lang_values = {cfg.get("african_celtic_value") for cfg in self.language_configs
                          if cfg.get("african_celtic_value")}
        relevant = all_lang_values | {"english"}

        id_to_items: dict[str, dict[str, Any]] = {}
        scanned = 0
        try:
            stream = _load_no_torchcodec(self.dataset_id, "default", self.split)
            for item in stream:
                if scanned >= self.max_scan_rows:
                    break
                lang = item.get("language", "")
                tid  = item.get("text_id", "")
                if lang in relevant and tid:
                    id_to_items.setdefault(tid, {})
                    if lang not in id_to_items[tid]:
                        id_to_items[tid][lang] = item
                scanned += 1
        except Exception as e:
            if monitor:
                monitor.step("  Scan error", str(e)[:120])

        if monitor:
            monitor.step(f"  Scanned {scanned} rows", f"{len(id_to_items)} text_ids found")

        for cfg in self.language_configs:
            lk = cfg["language_key"]
            av = cfg.get("african_celtic_value")
            if not av:
                self._cache[lk] = []
                continue
            pairs: list[dict] = []
            for tid, lang_items in id_to_items.items():
                if len(pairs) >= self.max_pairs:
                    break
                src_item = lang_items.get(av)
                eng_item = lang_items.get("english")
                if not src_item or not eng_item:
                    continue
                src_text = _norm(_get_text(src_item))
                eng_text = _norm(_get_text(eng_item))
                if not src_text or not eng_text:
                    continue
                pairs.append({
                    "pair_idx":  len(pairs),
                    "src_text":  src_text,
                    "eng_text":  eng_text,
                    "src_audio": src_item.get("audio"),
                    "eng_audio": eng_item.get("audio"),
                })
            self._cache[lk] = pairs
            if monitor:
                monitor.step(f"  {cfg['display']}", f"{len(pairs)} pairs")


# ── helpers ───────────────────────────────────────────────────────────────────

def make_translation_rows(
    lang_cfg: dict,
    pairs: list[dict],
    directions: list[str],
    model_lang_code: str,
    english_lang_code: str,
) -> list[dict]:
    """Build per-sample row dicts for text/audio evaluation loops."""
    rows: list[dict] = []
    for p in pairs:
        for direction in directions:
            if direction == "source_to_english":
                rows.append({
                    "pair_idx":       p["pair_idx"],
                    "language_key":   lang_cfg["language_key"],
                    "language":       lang_cfg["display"],
                    "direction":      "source_to_english",
                    "direction_label": f"{lang_cfg['display']}→English",
                    "source_text":    p["src_text"],
                    "target_text":    p["eng_text"],
                    "source_audio":   p["src_audio"],
                    "target_audio":   p["eng_audio"],
                    "source_lang":    model_lang_code,
                    "target_lang":    english_lang_code,
                })
            elif direction == "english_to_source":
                rows.append({
                    "pair_idx":       p["pair_idx"],
                    "language_key":   lang_cfg["language_key"],
                    "language":       lang_cfg["display"],
                    "direction":      "english_to_source",
                    "direction_label": f"English→{lang_cfg['display']}",
                    "source_text":    p["eng_text"],
                    "target_text":    p["src_text"],
                    "source_audio":   p["eng_audio"],
                    "target_audio":   p["src_audio"],
                    "source_lang":    english_lang_code,
                    "target_lang":    model_lang_code,
                })
    return rows
