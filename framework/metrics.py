from __future__ import annotations
import numpy as np
import sacrebleu

try:
    from jiwer import wer as _wer, cer as _cer
    _JIWER_OK = True
except Exception:
    _JIWER_OK = False


def compute_translation_metrics(predictions: list[str], references: list[str]) -> dict:
    preds = [str(p).strip() for p in predictions]
    refs  = [str(r).strip() for r in references]
    n     = len(preds)
    empty = sum(1 for p in preds if not p)
    base  = {"num_samples": n, "num_empty": empty}

    if not n or not any(preds):
        return {**base, "BLEU": 0.0, "ChrF": 0.0,
                "avg_pred_len": 0.0, "avg_ref_len": 0.0, "pred_ref_ratio": float("nan")}
    try:
        bleu = float(sacrebleu.corpus_bleu(preds, [refs]).score)
        chrf = float(sacrebleu.corpus_chrf(preds, [refs]).score)
    except Exception:
        bleu = chrf = 0.0

    avg_pred = float(np.mean([len(p.split()) for p in preds if p]) or 0)
    avg_ref  = float(np.mean([len(r.split()) for r in refs  if r]) or 0)
    return {
        **base,
        "BLEU":          bleu,
        "ChrF":          chrf,
        "avg_pred_len":  avg_pred,
        "avg_ref_len":   avg_ref,
        "pred_ref_ratio": avg_pred / avg_ref if avg_ref > 0 else float("nan"),
    }


def compute_asr_metrics(predictions: list[str], references: list[str]) -> dict:
    preds = [str(p).strip() for p in predictions]
    refs  = [str(r).strip() for r in references]
    n     = len(preds)
    empty = sum(1 for p in preds if not p)
    base  = {"num_samples": n, "num_empty": empty}
    if not _JIWER_OK or not n:
        return {**base, "WER": float("nan"), "CER": float("nan")}
    try:
        wer = float(_wer(refs, preds))
        cer = float(_cer(refs, preds))
    except Exception:
        wer = cer = float("nan")
    return {**base, "WER": wer, "CER": cer}
