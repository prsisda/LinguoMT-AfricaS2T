from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ModelCapabilities:
    model_type: str       # seamless_m4t | whisper | nllb | whisper_nllb | unknown
    direct_s2tt: bool     # end-to-end speech → text translation
    asr: bool             # speech → transcript
    t2tt: bool            # text → text translation
    bidirectional: bool   # can translate english → source language
    is_cascade: bool      # multi-model pipeline (e.g. whisper+nllb)
    english_code: str     # model's internal code for English
    s2tt_lang_attr: str | None = "seamless_code"
    asr_lang_attr: str | None  = "whisper_code"
    t2tt_lang_attr: str | None = "nllb_code"

    @property
    def cascade_available(self) -> bool:
        return self.asr and self.t2tt

    @property
    def enabled_strategies(self) -> list[str]:
        out = []
        if self.direct_s2tt:
            out += ["baseline_direct", "normalized_audio", "trimmed_audio", "chunk_based_audio"]
        if self.t2tt:
            out += ["transcript_translation"]
        if self.bidirectional and self.t2tt:
            out += ["english_to_source"]
        if self.cascade_available:
            out += ["gold_asr_cascade"]
        return out


_REGISTRY: dict[str, dict] = {
    "facebook/seamless-m4t-v2-large": dict(
        model_type="seamless_m4t", direct_s2tt=True, asr=True, t2tt=True,
        bidirectional=True, is_cascade=False, english_code="eng",
        s2tt_lang_attr="seamless_code", asr_lang_attr="seamless_code", t2tt_lang_attr="seamless_code",
    ),
    "facebook/seamless-m4t-large": dict(
        model_type="seamless_m4t", direct_s2tt=True, asr=True, t2tt=True,
        bidirectional=True, is_cascade=False, english_code="eng",
        s2tt_lang_attr="seamless_code", asr_lang_attr="seamless_code", t2tt_lang_attr="seamless_code",
    ),
    "openai/whisper-large-v3": dict(
        model_type="whisper", direct_s2tt=False, asr=True, t2tt=False,
        bidirectional=False, is_cascade=False, english_code="en",
        s2tt_lang_attr=None, asr_lang_attr="whisper_code", t2tt_lang_attr=None,
    ),
    "openai/whisper-large-v2": dict(
        model_type="whisper", direct_s2tt=False, asr=True, t2tt=False,
        bidirectional=False, is_cascade=False, english_code="en",
        s2tt_lang_attr=None, asr_lang_attr="whisper_code", t2tt_lang_attr=None,
    ),
    "facebook/nllb-200-600M": dict(
        model_type="nllb", direct_s2tt=False, asr=False, t2tt=True,
        bidirectional=True, is_cascade=False, english_code="eng_Latn",
        s2tt_lang_attr=None, asr_lang_attr=None, t2tt_lang_attr="nllb_code",
    ),
    "facebook/nllb-200-1.3B": dict(
        model_type="nllb", direct_s2tt=False, asr=False, t2tt=True,
        bidirectional=True, is_cascade=False, english_code="eng_Latn",
        s2tt_lang_attr=None, asr_lang_attr=None, t2tt_lang_attr="nllb_code",
    ),
    # Virtual model id for whisper+nllb cascade drivers
    "whisper_nllb": dict(
        model_type="whisper_nllb", direct_s2tt=False, asr=True, t2tt=True,
        bidirectional=True, is_cascade=True, english_code="eng_Latn",
        s2tt_lang_attr=None, asr_lang_attr="whisper_code", t2tt_lang_attr="nllb_code",
    ),
}


def detect_model_capabilities(model_id: str) -> ModelCapabilities:
    cfg = _REGISTRY.get(model_id)
    if cfg is None:
        mid = model_id.lower()
        if "seamless" in mid:
            cfg = _REGISTRY["facebook/seamless-m4t-v2-large"]
        elif "nllb" in mid and "whisper" in mid:
            cfg = _REGISTRY["whisper_nllb"]
        elif "whisper" in mid:
            cfg = _REGISTRY["openai/whisper-large-v3"]
        elif "nllb" in mid:
            cfg = _REGISTRY["facebook/nllb-200-600M"]
        else:
            cfg = dict(
                model_type="unknown", direct_s2tt=False, asr=False, t2tt=False,
                bidirectional=False, is_cascade=False, english_code="eng",
                s2tt_lang_attr=None, asr_lang_attr=None, t2tt_lang_attr=None,
            )
    return ModelCapabilities(**cfg)
