from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .capabilities import ModelCapabilities

# Comprehensive African language registry.
# Each entry maps a canonical key to dataset/model-specific codes.
AFRICAN_LANGUAGES: dict[str, dict] = {
    "igbo":     {"display": "Igbo",     "iso3": "ibo", "seamless_code": "ibo",  "whisper_code": None,  "nllb_code": "ibo_Latn", "fleurs_config": "ig_ng", "african_celtic_value": "igbo"},
    "yoruba":   {"display": "Yoruba",   "iso3": "yor", "seamless_code": "yor",  "whisper_code": "yo",  "nllb_code": "yor_Latn", "fleurs_config": "yo_ng", "african_celtic_value": "yoruba"},
    "hausa":    {"display": "Hausa",    "iso3": "hau", "seamless_code": "hau",  "whisper_code": "ha",  "nllb_code": "hau_Latn", "fleurs_config": "ha_ng", "african_celtic_value": "hausa"},
    "swahili":  {"display": "Swahili",  "iso3": "swh", "seamless_code": "swh",  "whisper_code": "sw",  "nllb_code": "swh_Latn", "fleurs_config": "sw_ke", "african_celtic_value": None},
    "amharic":  {"display": "Amharic",  "iso3": "amh", "seamless_code": "amh",  "whisper_code": None,  "nllb_code": "amh_Ethi", "fleurs_config": "am_et", "african_celtic_value": None},
    "wolof":    {"display": "Wolof",    "iso3": "wol", "seamless_code": "wol",  "whisper_code": None,  "nllb_code": "wol_Latn", "fleurs_config": "wo_sn", "african_celtic_value": None},
    "zulu":     {"display": "Zulu",     "iso3": "zul", "seamless_code": "zul",  "whisper_code": None,  "nllb_code": "zul_Latn", "fleurs_config": "zu_za", "african_celtic_value": None},
    "xhosa":    {"display": "Xhosa",    "iso3": "xho", "seamless_code": "xho",  "whisper_code": None,  "nllb_code": "xho_Latn", "fleurs_config": "xh_za", "african_celtic_value": None},
    "somali":   {"display": "Somali",   "iso3": "som", "seamless_code": "som",  "whisper_code": None,  "nllb_code": "som_Latn", "fleurs_config": "so_so", "african_celtic_value": None},
    "oromo":    {"display": "Oromo",    "iso3": "orm", "seamless_code": "orm",  "whisper_code": None,  "nllb_code": "orm_Latn", "fleurs_config": "om_et", "african_celtic_value": None},
    "fula":     {"display": "Fula",     "iso3": "ful", "seamless_code": "ful",  "whisper_code": None,  "nllb_code": "ful_Latn", "fleurs_config": "ff_sn", "african_celtic_value": None},
    "lingala":  {"display": "Lingala",  "iso3": "lin", "seamless_code": "lin",  "whisper_code": None,  "nllb_code": "lin_Latn", "fleurs_config": "ln_cd", "african_celtic_value": None},
    "twi":      {"display": "Twi",      "iso3": "twi", "seamless_code": "twi",  "whisper_code": None,  "nllb_code": "twi_Latn", "fleurs_config": "tw_gh", "african_celtic_value": None},
    "ewe":      {"display": "Ewe",      "iso3": "ewe", "seamless_code": "ewe",  "whisper_code": None,  "nllb_code": "ewe_Latn", "fleurs_config": "ee_gh", "african_celtic_value": None},
    "kikuyu":   {"display": "Kikuyu",   "iso3": "kik", "seamless_code": "kik",  "whisper_code": None,  "nllb_code": "kik_Latn", "fleurs_config": "ki_ke", "african_celtic_value": None},
    "luo":      {"display": "Luo",      "iso3": "luo", "seamless_code": "luo",  "whisper_code": None,  "nllb_code": "luo_Latn", "fleurs_config": None,    "african_celtic_value": None},
}

_DATASET_ADAPTER: dict[str, str] = {
    "google/fleurs":                       "fleurs",
    "McGill-NLP/african_celtic_dataset":   "african_celtic",
}


def get_adapter_type(dataset_id: str) -> str:
    for k, v in _DATASET_ADAPTER.items():
        if k in dataset_id:
            return v
    return "unknown"


def _model_supports(lang_key: str, caps: "ModelCapabilities") -> bool:
    ld = AFRICAN_LANGUAGES.get(lang_key, {})
    mt = caps.model_type
    if mt == "seamless_m4t":
        return bool(ld.get("seamless_code"))
    if mt == "whisper":
        return bool(ld.get("whisper_code"))
    if mt == "nllb":
        return bool(ld.get("nllb_code"))
    if mt == "whisper_nllb":
        return bool(ld.get("whisper_code")) and bool(ld.get("nllb_code"))
    return False


def _dataset_has(lang_key: str, adapter: str) -> bool:
    ld = AFRICAN_LANGUAGES.get(lang_key, {})
    if adapter == "fleurs":
        return bool(ld.get("fleurs_config"))
    if adapter == "african_celtic":
        return bool(ld.get("african_celtic_value"))
    return False


def _probe_model_languages(caps: "ModelCapabilities", processor) -> set[str] | None:
    """Try to verify supported languages from the loaded tokenizer/processor."""
    try:
        mt = caps.model_type
        tokenizer = getattr(processor, "tokenizer", None) or processor
        if mt == "seamless_m4t":
            special = getattr(tokenizer, "additional_special_tokens", [])
            codes = {t.strip("_") for t in special if t.startswith("__") and t.endswith("__")}
            if not codes:
                return None
            return {lk for lk, ld in AFRICAN_LANGUAGES.items() if ld.get("seamless_code") in codes}
        if mt == "whisper":
            special = getattr(tokenizer, "additional_special_tokens", [])
            codes = {t.strip("<|>") for t in special if t.startswith("<|") and len(t) <= 8}
            return {lk for lk, ld in AFRICAN_LANGUAGES.items() if ld.get("whisper_code") in codes}
        if mt == "nllb":
            vocab = getattr(tokenizer, "lang_code_to_id", {})
            return {lk for lk, ld in AFRICAN_LANGUAGES.items() if ld.get("nllb_code") in vocab}
        if mt == "whisper_nllb":
            special = getattr(tokenizer, "additional_special_tokens", [])
            codes = {t.strip("<|>") for t in special if t.startswith("<|") and len(t) <= 8}
            return {lk for lk, ld in AFRICAN_LANGUAGES.items() if ld.get("whisper_code") in codes}
    except Exception:
        pass
    return None


def select_supported_languages(
    caps: "ModelCapabilities",
    dataset_id: str,
    processor=None,
    max_langs: int = 3,
    manual_override: list[str] | None = None,
) -> list[dict]:
    """
    Returns up to max_langs language config dicts that are both in the dataset
    and supported by the model.  manual_override bypasses detection entirely.
    """
    if manual_override:
        return [
            {"language_key": lk, **AFRICAN_LANGUAGES[lk]}
            for lk in manual_override
            if lk in AFRICAN_LANGUAGES
        ]

    adapter = get_adapter_type(dataset_id)
    model_confirmed: set[str] | None = None
    if processor is not None:
        model_confirmed = _probe_model_languages(caps, processor)

    candidates: list[dict] = []
    for lk, ld in AFRICAN_LANGUAGES.items():
        if not _dataset_has(lk, adapter):
            continue
        if model_confirmed is not None:
            if lk not in model_confirmed:
                continue
        elif not _model_supports(lk, caps):
            continue
        candidates.append({"language_key": lk, **ld})

    return candidates[:max_langs]


def build_lang_lookup(language_configs: list[dict]) -> dict[str, dict]:
    return {c["language_key"]: c for c in language_configs}


def get_model_lang_code(lang_cfg: dict, caps: "ModelCapabilities") -> str | None:
    mt = caps.model_type
    if mt in ("seamless_m4t",):
        return lang_cfg.get("seamless_code")
    if mt == "whisper":
        return lang_cfg.get("whisper_code")
    if mt == "nllb":
        return lang_cfg.get("nllb_code")
    if mt == "whisper_nllb":
        return lang_cfg.get("nllb_code")  # MT direction
    return None
