"""
Fine-tuning module — LoRA-based fine-tuning for all supported model families.

Usage: instantiate the appropriate FineTuner subclass from the run script,
pass it to ExperimentRunner as finetune_fn=tuner.finetune.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    # Which tasks to fine-tune
    finetune_text_translation:        bool = True
    finetune_reverse_translation:     bool = True
    finetune_asr:                     bool = True
    finetune_direct_speech_translation: bool = False  # S2TT, only for Seamless

    # Method
    finetuning_method: str = "lora"     # "lora" | "full"
    lora_r:            int = 8
    lora_alpha:        int = 16
    lora_dropout:      float = 0.05

    # Sample counts (per language, per direction)
    text_finetune_samples:   int = 1000
    asr_finetune_samples:    int = 500
    st_finetune_samples:     int = 200   # direct S2TT samples

    # Training hyperparameters — text / MT
    text_epochs:      int   = 3
    text_batch_size:  int   = 8
    text_lr:          float = 5e-5
    text_max_length:  int   = 128

    # Training hyperparameters — ASR
    asr_epochs:       int   = 3
    asr_batch_size:   int   = 4
    asr_lr:           float = 1e-5
    asr_max_length:   int   = 225

    # Training hyperparameters — direct S2TT
    st_epochs:        int   = 3
    st_batch_size:    int   = 4
    st_lr:            float = 1e-5

    # Efficiency
    gradient_accumulation_steps: int  = 4
    fp16:                        bool = True   # auto-disabled if CPU
    early_stopping_patience:     int  = 2

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir:   str  = ""   # set at runtime from OutputDirs

    # Evaluation
    eval_before_after: bool = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_import_peft():
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        return get_peft_model, LoraConfig, TaskType
    except ImportError:
        raise ImportError(
            "peft is required for LoRA fine-tuning. "
            "Install with: pip install peft"
        )


def _lora_model(model, target_modules: list[str], r: int, alpha: int,
                dropout: float, task_type_str: str):
    get_peft_model, LoraConfig, TaskType = _try_import_peft()
    task_type = getattr(TaskType, task_type_str, TaskType.SEQ_2_SEQ_LM)
    lora_cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, task_type=task_type, bias="none",
    )
    return get_peft_model(model, lora_cfg)


def _collate_seq2seq(pairs: list[dict], tokenizer, src_field: str, tgt_field: str,
                     src_lang: str, tgt_lang: str, max_length: int, device):
    import torch
    tokenizer.src_lang = src_lang
    src_texts = [str(p.get(src_field, "") or "") for p in pairs]
    tgt_texts = [str(p.get(tgt_field, "") or "") for p in pairs]
    model_inputs = tokenizer(src_texts, max_length=max_length,
                             truncation=True, padding=True, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt_texts, max_length=max_length,
                           truncation=True, padding=True, return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return {k: v.to(device) for k, v in model_inputs.items()}


def _train_loop(model, optimizer, batches: list, cfg: FinetuneConfig,
                task_label: str, fp16: bool, device) -> list[float]:
    import torch
    scaler = torch.cuda.amp.GradScaler() if fp16 and device.type == "cuda" else None
    losses = []
    model.train()
    best_loss, patience_left = float("inf"), cfg.early_stopping_patience
    for epoch in range(cfg.text_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(batches):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                out = model(**batch)
                loss = out.loss / cfg.gradient_accumulation_steps
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item() * cfg.gradient_accumulation_steps
        avg = epoch_loss / max(len(batches), 1)
        losses.append(avg)
        print(f"    {task_label} epoch {epoch+1}/{cfg.text_epochs}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg; patience_left = cfg.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    model.eval()
    return losses


def _save_checkpoint(model, path: Path, label: str) -> None:
    try:
        ckpt = path / label
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt))
        print(f"    Checkpoint saved → {ckpt}")
    except Exception as e:
        print(f"    Checkpoint save failed: {e}")


# ── Base class ────────────────────────────────────────────────────────────────

class FineTuner:
    """Base class — subclasses implement _finetune_text, _finetune_asr, etc."""

    def __init__(self, ft_cfg: FinetuneConfig, device):
        self.ft_cfg = ft_cfg
        self.device = device
        self._ckpt_root = Path(ft_cfg.checkpoint_dir) if ft_cfg.checkpoint_dir else None

    def finetune(self, pairs: list[dict], lang_cfg: dict, exp: dict) -> None:
        """Called by ExperimentRunner._run_finetuning() for each language × experiment."""
        lk   = lang_cfg["language_key"]
        disp = lang_cfg["display"]
        n    = len(pairs)
        print(f"\n[FineTune] {disp} — {n} pairs  (exp={exp['experiment']})")

        if self.ft_cfg.finetune_text_translation:
            t0 = time.time()
            self._finetune_text(pairs[:self.ft_cfg.text_finetune_samples], lang_cfg, "src_to_eng")
            print(f"  text MT done in {time.time()-t0:.1f}s")

        if self.ft_cfg.finetune_reverse_translation:
            t0 = time.time()
            self._finetune_text(pairs[:self.ft_cfg.text_finetune_samples], lang_cfg, "eng_to_src")
            print(f"  reverse MT done in {time.time()-t0:.1f}s")

        if self.ft_cfg.finetune_asr:
            t0 = time.time()
            self._finetune_asr(pairs[:self.ft_cfg.asr_finetune_samples], lang_cfg)
            print(f"  ASR done in {time.time()-t0:.1f}s")

        if self.ft_cfg.finetune_direct_speech_translation:
            t0 = time.time()
            self._finetune_st(pairs[:self.ft_cfg.st_finetune_samples], lang_cfg)
            print(f"  direct S2TT done in {time.time()-t0:.1f}s")

    def _finetune_text(self, pairs, lang_cfg, direction): pass
    def _finetune_asr(self, pairs, lang_cfg): pass
    def _finetune_st(self, pairs, lang_cfg): pass

    def _fp16(self) -> bool:
        import torch
        return self.ft_cfg.fp16 and self.device.type == "cuda"

    def _batch(self, items: list, size: int) -> list[list]:
        return [items[i:i+size] for i in range(0, len(items), size)]


# ── NLLB fine-tuner ───────────────────────────────────────────────────────────

class NLLBTextFineTuner(FineTuner):
    """Fine-tunes NLLB for text translation (src↔eng)."""

    def __init__(self, model, tokenizer, ft_cfg: FinetuneConfig, device):
        super().__init__(ft_cfg, device)
        self.tokenizer = tokenizer
        if ft_cfg.finetuning_method == "lora":
            self.model = _lora_model(
                model,
                target_modules=["q_proj", "v_proj"],
                r=ft_cfg.lora_r, alpha=ft_cfg.lora_alpha,
                dropout=ft_cfg.lora_dropout, task_type_str="SEQ_2_SEQ_LM",
            )
        else:
            self.model = model

    def _finetune_text(self, pairs: list[dict], lang_cfg: dict, direction: str):
        import torch
        cfg = self.ft_cfg
        if direction == "src_to_eng":
            src_lang = lang_cfg["nllb_code"]
            tgt_lang = "eng_Latn"
            src_f, tgt_f = "src_text", "eng_text"
        else:
            src_lang = "eng_Latn"
            tgt_lang = lang_cfg["nllb_code"]
            src_f, tgt_f = "eng_text", "src_text"

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.text_lr
        )
        batches = self._batch(pairs, cfg.text_batch_size)
        batch_inputs = [
            _collate_seq2seq(b, self.tokenizer, src_f, tgt_f, src_lang, tgt_lang,
                             cfg.text_max_length, self.device)
            for b in batches
        ]
        _train_loop(self.model, opt, batch_inputs, cfg, f"NLLB-{direction}", self._fp16(), self.device)
        if cfg.save_checkpoints and self._ckpt_root:
            _save_checkpoint(self.model, self._ckpt_root, f"nllb_{direction}_{lang_cfg['language_key']}")


# ── Whisper ASR fine-tuner ────────────────────────────────────────────────────

class WhisperASRFineTuner(FineTuner):
    """Fine-tunes Whisper for ASR on source language."""

    def __init__(self, model, processor, ft_cfg: FinetuneConfig, device):
        super().__init__(ft_cfg, device)
        self.processor = processor
        if ft_cfg.finetuning_method == "lora":
            self.model = _lora_model(
                model,
                target_modules=["q_proj", "v_proj"],
                r=ft_cfg.lora_r, alpha=ft_cfg.lora_alpha,
                dropout=ft_cfg.lora_dropout, task_type_str="SEQ_2_SEQ_LM",
            )
        else:
            self.model = model

    def _finetune_asr(self, pairs: list[dict], lang_cfg: dict):
        import torch
        cfg = self.ft_cfg
        whisper_code = lang_cfg.get("whisper_code")
        if not whisper_code:
            print(f"    Whisper ASR FT skipped — no whisper_code for {lang_cfg['display']}")
            return

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.asr_lr
        )

        def _make_batch(batch_pairs):
            audios = []
            for p in batch_pairs:
                arr = p.get("src_audio")
                if arr is None:
                    arr = np.zeros(16000, dtype=np.float32)
                elif hasattr(arr, "array"):
                    arr = np.array(arr.array, dtype=np.float32)
                else:
                    arr = np.array(arr, dtype=np.float32)
                audios.append(arr)
            feats = self.processor(audios, sampling_rate=16000, return_tensors="pt",
                                   padding=True).input_features.to(self.device)
            forced = self.processor.get_decoder_prompt_ids(language=whisper_code, task="transcribe")
            labels_text = [str(p.get("src_text", "") or "") for p in batch_pairs]
            with self.processor.as_target_processor():
                lbl = self.processor(labels_text, return_tensors="pt",
                                     padding=True, max_length=cfg.asr_max_length,
                                     truncation=True).input_ids.to(self.device)
            lbl[lbl == self.processor.tokenizer.pad_token_id] = -100
            return {"input_features": feats, "labels": lbl,
                    "forced_decoder_ids": torch.tensor(forced, device=self.device)}

        batches_raw = self._batch(pairs, cfg.asr_batch_size)
        batch_inputs = []
        for b in batches_raw:
            b_filt = [p for p in b if p.get("src_audio") is not None or p.get("src_text")]
            if b_filt:
                try:
                    batch_inputs.append(_make_batch(b_filt))
                except Exception as e:
                    print(f"    ASR batch prep error: {e}")

        _train_loop(self.model, opt, batch_inputs, cfg, "Whisper-ASR", self._fp16(), self.device)
        if cfg.save_checkpoints and self._ckpt_root:
            _save_checkpoint(self.model, self._ckpt_root, f"whisper_asr_{lang_cfg['language_key']}")


# ── SeamlessM4T fine-tuner ────────────────────────────────────────────────────

class SeamlessM4TFineTuner(FineTuner):
    """Fine-tunes SeamlessM4T-v2 for text MT, ASR, and optionally direct S2TT."""

    def __init__(self, model, processor, ft_cfg: FinetuneConfig, device):
        super().__init__(ft_cfg, device)
        self.processor = processor
        if ft_cfg.finetuning_method == "lora":
            self.model = _lora_model(
                model,
                target_modules=["q_proj", "v_proj", "k_proj"],
                r=ft_cfg.lora_r, alpha=ft_cfg.lora_alpha,
                dropout=ft_cfg.lora_dropout, task_type_str="SEQ_2_SEQ_LM",
            )
        else:
            self.model = model

    def _finetune_text(self, pairs: list[dict], lang_cfg: dict, direction: str):
        import torch
        cfg = self.ft_cfg
        seamless_code = lang_cfg.get("seamless_code")
        if not seamless_code:
            print(f"    Seamless text FT skipped — no seamless_code for {lang_cfg['display']}")
            return

        if direction == "src_to_eng":
            src_lang, tgt_lang = seamless_code, "eng"
            src_f, tgt_f = "src_text", "eng_text"
        else:
            src_lang, tgt_lang = "eng", seamless_code
            src_f, tgt_f = "eng_text", "src_text"

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.text_lr
        )

        def _make_batch(b):
            src_texts = [str(p.get(src_f, "") or "") for p in b]
            tgt_texts = [str(p.get(tgt_f, "") or "") for p in b]
            inp = self.processor(text=src_texts, src_lang=src_lang,
                                 return_tensors="pt", padding=True,
                                 truncation=True, max_length=cfg.text_max_length).to(self.device)
            with self.processor.as_target_processor():
                lbl = self.processor(text=tgt_texts, tgt_lang=tgt_lang,
                                     return_tensors="pt", padding=True,
                                     truncation=True, max_length=cfg.text_max_length).input_ids.to(self.device)
            lbl[lbl == self.processor.tokenizer.pad_token_id] = -100
            inp["labels"] = lbl
            return inp

        batches = [_make_batch(b) for b in self._batch(pairs, cfg.text_batch_size)
                   if any(p.get(src_f) for p in b)]
        _train_loop(self.model, opt, batches, cfg, f"Seamless-text-{direction}", self._fp16(), self.device)
        if cfg.save_checkpoints and self._ckpt_root:
            _save_checkpoint(self.model, self._ckpt_root, f"seamless_text_{direction}_{lang_cfg['language_key']}")

    def _finetune_asr(self, pairs: list[dict], lang_cfg: dict):
        import torch
        cfg = self.ft_cfg
        seamless_code = lang_cfg.get("seamless_code")
        if not seamless_code:
            return

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.asr_lr
        )

        def _make_batch(b):
            audios, texts = [], []
            for p in b:
                arr = p.get("src_audio")
                if arr is None:
                    continue
                if hasattr(arr, "array"):
                    arr = np.array(arr.array, dtype=np.float32)
                else:
                    arr = np.array(arr, dtype=np.float32)
                audios.append(arr)
                texts.append(str(p.get("src_text", "") or ""))
            if not audios:
                return None
            inp = self.processor(audio=audios, sampling_rate=16000, src_lang=seamless_code,
                                 return_tensors="pt", padding=True).to(self.device)
            with self.processor.as_target_processor():
                lbl = self.processor(text=texts, tgt_lang=seamless_code,
                                     return_tensors="pt", padding=True,
                                     max_length=cfg.asr_max_length, truncation=True).input_ids.to(self.device)
            lbl[lbl == self.processor.tokenizer.pad_token_id] = -100
            inp["labels"] = lbl
            return inp

        batches = [r for b in self._batch(pairs, cfg.asr_batch_size)
                   if (r := _make_batch([p for p in b if p.get("src_audio") is not None])) is not None]
        _train_loop(self.model, opt, batches, cfg, "Seamless-ASR", self._fp16(), self.device)
        if cfg.save_checkpoints and self._ckpt_root:
            _save_checkpoint(self.model, self._ckpt_root, f"seamless_asr_{lang_cfg['language_key']}")

    def _finetune_st(self, pairs: list[dict], lang_cfg: dict):
        import torch
        cfg = self.ft_cfg
        seamless_code = lang_cfg.get("seamless_code")
        if not seamless_code:
            return

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.st_lr
        )

        def _make_batch(b):
            audios, texts = [], []
            for p in b:
                arr = p.get("src_audio")
                if arr is None:
                    continue
                if hasattr(arr, "array"):
                    arr = np.array(arr.array, dtype=np.float32)
                else:
                    arr = np.array(arr, dtype=np.float32)
                audios.append(arr)
                texts.append(str(p.get("eng_text", "") or ""))
            if not audios:
                return None
            inp = self.processor(audio=audios, sampling_rate=16000, src_lang=seamless_code,
                                 return_tensors="pt", padding=True).to(self.device)
            with self.processor.as_target_processor():
                lbl = self.processor(text=texts, tgt_lang="eng",
                                     return_tensors="pt", padding=True,
                                     max_length=cfg.text_max_length, truncation=True).input_ids.to(self.device)
            lbl[lbl == self.processor.tokenizer.pad_token_id] = -100
            inp["labels"] = lbl
            return inp

        batches = [r for b in self._batch(pairs, cfg.st_batch_size)
                   if (r := _make_batch([p for p in b if p.get("src_audio") is not None])) is not None]
        _train_loop(self.model, opt, batches, cfg, "Seamless-S2TT", self._fp16(), self.device)
        if cfg.save_checkpoints and self._ckpt_root:
            _save_checkpoint(self.model, self._ckpt_root, f"seamless_st_{lang_cfg['language_key']}")


# ── Whisper+NLLB cascade fine-tuner ──────────────────────────────────────────

class WhisperNLLBCascadeFineTuner(FineTuner):
    """Fine-tunes both Whisper (ASR) and NLLB (MT) for the cascade pipeline."""

    def __init__(
        self,
        whisper_model, whisper_processor,
        nllb_model, nllb_tokenizer,
        ft_cfg: FinetuneConfig, device,
    ):
        super().__init__(ft_cfg, device)
        self._asr_tuner  = WhisperASRFineTuner(whisper_model, whisper_processor, ft_cfg, device)
        self._text_tuner = NLLBTextFineTuner(nllb_model, nllb_tokenizer, ft_cfg, device)

    def _finetune_text(self, pairs, lang_cfg, direction):
        self._text_tuner._finetune_text(pairs, lang_cfg, direction)

    def _finetune_asr(self, pairs, lang_cfg):
        self._asr_tuner._finetune_asr(pairs, lang_cfg)

    # expose fine-tuned models for the run script to swap in
    @property
    def whisper_model(self):
        return self._asr_tuner.model

    @property
    def nllb_model(self):
        return self._text_tuner.model


# ── Fine-tuning comparison tables ─────────────────────────────────────────────

def build_finetune_comparison_table(
    before_results: list[dict],
    after_results:  list[dict],
    task: str = "text",
) -> "pd.DataFrame":
    """T_FT — before/after comparison table for a given task."""
    import pandas as pd
    if not before_results or not after_results:
        return pd.DataFrame()
    bf = pd.DataFrame(before_results)
    af = pd.DataFrame(after_results)
    key_cols = ["experiment", "language", "direction_label"]
    key_cols = [c for c in key_cols if c in bf.columns and c in af.columns]
    metric_cols = ["BLEU", "ChrF"] if task in ("text", "audio") else ["WER", "CER"]
    rows = []
    for _, row_b in bf.iterrows():
        mask = pd.Series([True] * len(af), index=af.index)
        for k in key_cols:
            if k in af.columns:
                mask &= af[k] == row_b.get(k)
        matches = af[mask]
        if matches.empty:
            continue
        row_a = matches.iloc[0]
        r = {k: row_b.get(k) for k in key_cols}
        r["task"] = task
        for m in metric_cols:
            if m in bf.columns and m in af.columns:
                r[f"{m}_before"] = row_b.get(m, float("nan"))
                r[f"{m}_after"]  = row_a.get(m, float("nan"))
                r[f"{m}_delta"]  = r[f"{m}_after"] - r[f"{m}_before"]
        rows.append(r)
    return pd.DataFrame(rows)
