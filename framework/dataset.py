from __future__ import annotations
import hashlib
import pickle
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


def _align_key(text_id: str) -> str:
    """Strip the language-initial prefix from African-Celtic text_ids.

    Each language uses a single-character prefix (E=English, H=Hausa,
    I=Igbo, Y=Yoruba) followed by a shared suffix, e.g. 'ETE_0001' and
    'ITE_0001' both align to 'TE_0001'.
    """
    tid = str(text_id).strip().upper()
    if len(tid) >= 2 and tid[0] in {"E", "H", "I", "Y"}:
        return tid[1:]
    return tid


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

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        langs = ",".join(sorted(c["language_key"] for c in self.language_configs))
        key   = f"{self.dataset_id}|{self.split}|{langs}|{self.max_pairs}|{self.max_scan_rows}"
        tag   = hashlib.md5(key.encode()).hexdigest()[:10]
        return Path(self.cache_dir) / f"pairs_{self.adapter_type}_{tag}.pkl"

    def build(self, monitor=None) -> None:
        cache_path = self._cache_path()
        if cache_path and cache_path.exists() and not self.force_rerun:
            with open(cache_path, "rb") as fh:
                self._cache = pickle.load(fh)
            total = sum(len(v) for v in self._cache.values())
            print(f"[Dataset] Loaded {total} pairs from disk cache: {cache_path.name}")
            if monitor:
                monitor.step("Dataset loaded from disk cache", f"{total} pairs | {cache_path.name}")
            return

        if self.adapter_type == "fleurs":
            self._build_fleurs(monitor)
        elif self.adapter_type == "african_celtic":
            self._build_african_celtic(monitor)
        else:
            raise ValueError(f"Unknown adapter: {self.adapter_type}")

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(self._cache, fh)
            print(f"[Dataset] Saved to disk cache: {cache_path.name}")
            if monitor:
                monitor.step("Dataset cache saved", cache_path.name)

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

        # Build a lowercase → original mapping so matching is case-insensitive
        av_lower: dict[str, str] = {}
        for cfg in self.language_configs:
            av = cfg.get("african_celtic_value")
            if av:
                av_lower[av.lower()] = av
        av_lower["english"] = "english"
        relevant_lower = set(av_lower.keys())

        print(f"\n[Dataset] Starting African-Celtic scan: split={self.split!r}, "
              f"max_scan_rows={self.max_scan_rows}, max_pairs={self.max_pairs}")
        print(f"[Dataset] Searching for languages: {sorted(relevant_lower)}")

        id_to_items: dict[str, dict[str, Any]] = {}
        all_langs:   set[str] = set()
        found_langs: set[str] = set()
        scanned = 0
        try:
            stream = _load_no_torchcodec(self.dataset_id, "default", self.split)
            for item in stream:
                if scanned >= self.max_scan_rows:
                    break
                lang_raw  = item.get("language", "")
                lang_norm = lang_raw.lower()
                tid       = item.get("text_id", "")
                akey      = _align_key(tid)
                all_langs.add(lang_raw)
                if lang_norm in relevant_lower and akey:
                    canonical = av_lower[lang_norm]
                    id_to_items.setdefault(akey, {})
                    if canonical not in id_to_items[akey]:
                        id_to_items[akey][canonical] = item
                    found_langs.add(lang_raw)
                scanned += 1
                if scanned % 500 == 0:
                    print(f"[Dataset]   ... scanned {scanned} rows, "
                          f"{len(id_to_items)} text_ids collected, "
                          f"matched langs so far: {sorted(found_langs) or 'none yet'}")
        except Exception as e:
            print(f"[Dataset] ERROR during scan: {e}")
            if monitor:
                monitor.step("  Scan error", str(e)[:120])

        print(f"[Dataset] Scan complete: {scanned} rows scanned, {len(id_to_items)} text_ids")
        print(f"[Dataset] All langs in dataset:  {sorted(all_langs)}")
        print(f"[Dataset] Matched target langs:  {sorted(found_langs) or 'NONE — check african_celtic_value in languages.py'}")

        if monitor:
            monitor.step(f"  Scanned {scanned} rows",
                         f"{len(id_to_items)} text_ids | all langs: {sorted(all_langs)}")
            monitor.step("  Matched target langs",
                         f"{sorted(found_langs) or 'NONE — check african_celtic_value in languages.py'}")

        # Print sample item to verify field names
        if id_to_items:
            sample_tid   = next(iter(id_to_items))
            sample_langs = list(id_to_items[sample_tid].keys())
            sample_item  = id_to_items[sample_tid][sample_langs[0]]
            raw_tid      = sample_item.get("text_id", "?")
            print(f"[Dataset] Sample align_key={sample_tid!r} (raw text_id={raw_tid!r}): langs_present={sample_langs}")
            print(f"[Dataset] Sample item fields: {sorted(sample_item.keys())}")
            # Show actual text values for each field so we can see which one has content
            for field in ["transcription", "raw_transcription", "text", "sentence"]:
                val = sample_item.get(field)
                print(f"[Dataset]   field={field!r}: {str(val)[:80] if val is not None else '<missing>'}")
            if monitor:
                monitor.step("  Sample item fields", f"tid={sample_tid} | fields={sorted(sample_item.keys())}")
        else:
            print("[Dataset] WARNING: id_to_items is empty — no rows matched any target language")

        for cfg in self.language_configs:
            lk = cfg["language_key"]
            av = cfg.get("african_celtic_value")
            if not av:
                self._cache[lk] = []
                continue

            has_src  = sum(1 for li in id_to_items.values() if av in li)
            has_eng  = sum(1 for li in id_to_items.values() if "english" in li)
            has_both = sum(1 for li in id_to_items.values() if av in li and "english" in li)
            print(f"[Dataset] [{cfg['display']}] text_id overlap: "
                  f"has_src={has_src}  has_eng={has_eng}  has_both={has_both}")
            if monitor:
                monitor.step(f"  [{cfg['display']}] text_id overlap",
                             f"has_src={has_src} has_eng={has_eng} has_both={has_both}")

            pairs: list[dict] = []
            skipped_no_pair = 0
            skipped_empty   = 0
            for tid, lang_items in id_to_items.items():
                if len(pairs) >= self.max_pairs:
                    break
                src_item = lang_items.get(av) or lang_items.get(av.lower()) or lang_items.get(av.capitalize())
                eng_item = lang_items.get("english")
                if not src_item or not eng_item:
                    skipped_no_pair += 1
                    continue
                src_text = _norm(_get_text(src_item))
                eng_text = _norm(_get_text(eng_item))
                if not src_text or not eng_text:
                    skipped_empty += 1
                    continue
                pairs.append({
                    "pair_idx":  len(pairs),
                    "src_text":  src_text,
                    "eng_text":  eng_text,
                    "src_audio": src_item.get("audio"),
                    "eng_audio": eng_item.get("audio"),
                })

            print(f"[Dataset] [{cfg['display']}] pairs built: {len(pairs)}  "
                  f"(skipped_no_pair={skipped_no_pair}, skipped_empty_text={skipped_empty})")
            if pairs:
                print(f"[Dataset]   First pair src: {pairs[0]['src_text'][:80]!r}")
                print(f"[Dataset]   First pair eng: {pairs[0]['eng_text'][:80]!r}")
            self._cache[lk] = pairs
            if monitor:
                monitor.step(f"  {cfg['display']}",
                             f"{len(pairs)} pairs (skipped_no_pair={skipped_no_pair}, skipped_empty={skipped_empty})")


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
