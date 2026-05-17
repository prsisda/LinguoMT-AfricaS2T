from __future__ import annotations
from typing import Any
import numpy as np
import librosa

TARGET_SR             = 16_000
MIN_DURATION_SEC      = 1.0
MAX_DURATION_SEC      = 20.0
CHUNK_SECONDS         = 6
CHUNK_OVERLAP_SECONDS = 1

AUDIO_STRATEGIES: list[dict] = [
    {"key": "baseline_direct",   "label": "Direct audio → English",      "normalize": False, "trim": False, "chunk": False, "expensive": False},
    {"key": "normalized_audio",  "label": "Normalized audio → English",  "normalize": True,  "trim": False, "chunk": False, "expensive": False},
    {"key": "trimmed_audio",     "label": "Trimmed audio → English",     "normalize": True,  "trim": True,  "chunk": False, "expensive": True},
    {"key": "chunk_based_audio", "label": "Chunk-based audio → English", "normalize": True,  "trim": False, "chunk": True,  "expensive": True},
]


def _decode_raw_audio(audio_dict: dict, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Decode a raw {bytes, path} audio dict using soundfile (no torchcodec needed)."""
    import io
    import soundfile as sf
    raw_bytes = audio_dict.get("bytes")
    path      = audio_dict.get("path")
    if raw_bytes:
        audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    elif path:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    else:
        return np.zeros(int(MIN_DURATION_SEC * target_sr), dtype=np.float32), target_sr
    return audio.astype(np.float32), int(sr)


def extract_audio_array(audio_obj: Any, target_sr: int = TARGET_SR) -> np.ndarray:
    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
            sr    = audio_obj.get("sampling_rate", target_sr)
            audio = audio_obj["array"]
        elif "bytes" in audio_obj or "path" in audio_obj:
            # Raw (undecoded) audio dict — decode with soundfile
            audio, sr = _decode_raw_audio(audio_obj, target_sr)
        else:
            return np.zeros(int(MIN_DURATION_SEC * target_sr), dtype=np.float32)
    else:
        sr, audio = target_sr, audio_obj
    if audio is None:
        return np.zeros(int(MIN_DURATION_SEC * target_sr), dtype=np.float32)
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32).flatten())
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return _pad_to_min(audio, target_sr)


def normalize_waveform(audio: np.ndarray) -> np.ndarray:
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32).flatten())
    if not len(audio):
        return audio
    audio -= np.mean(audio)
    mx = np.max(np.abs(audio))
    return (audio / mx).astype(np.float32) if mx > 0 else audio


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).flatten()
    nsi = np.where(np.abs(audio) > threshold)[0]
    if not len(nsi):
        return audio
    trimmed = audio[nsi[0]: nsi[-1] + 1]
    return trimmed if len(trimmed) >= int(MIN_DURATION_SEC * TARGET_SR) else audio.astype(np.float32)


def chunk_audio(
    audio: np.ndarray,
    chunk_sec: float = CHUNK_SECONDS,
    overlap_sec: float = CHUNK_OVERLAP_SECONDS,
    sr: int = TARGET_SR,
) -> list[np.ndarray]:
    audio      = np.asarray(audio, dtype=np.float32).flatten()
    chunk_size = int(chunk_sec * sr)
    step_size  = max(1, chunk_size - int(overlap_sec * sr))
    chunks     = []
    for start in range(0, len(audio), step_size):
        ch = audio[start: start + chunk_size]
        if len(ch) >= int(MIN_DURATION_SEC * sr):
            chunks.append(ch.astype(np.float32))
        if start + chunk_size >= len(audio):
            break
    return chunks or [audio]


def apply_strategy(audio_obj: Any, strategy: dict) -> np.ndarray:
    audio = extract_audio_array(audio_obj)
    if strategy.get("trim"):
        audio = trim_silence(audio)
    if strategy.get("normalize"):
        audio = normalize_waveform(audio)
    return _pad_to_min(audio).astype(np.float32)


def audio_duration(audio_obj: Any) -> float:
    arr = extract_audio_array(audio_obj) if not isinstance(audio_obj, np.ndarray) else audio_obj
    return len(arr) / float(TARGET_SR)


def compute_audio_quality(audio_obj: Any) -> dict:
    audio = extract_audio_array(audio_obj) if not isinstance(audio_obj, np.ndarray) else np.asarray(audio_obj, dtype=np.float32).flatten()
    audio = np.nan_to_num(audio)
    if not len(audio):
        return {k: 0.0 for k in ["duration_sec", "rms_energy", "peak_amplitude",
                                   "silence_ratio", "clipping_ratio", "zero_crossing_rate", "dynamic_range"]}
    return {
        "duration_sec":       len(audio) / float(TARGET_SR),
        "rms_energy":         float(np.sqrt(np.mean(audio ** 2))),
        "peak_amplitude":     float(np.max(np.abs(audio))),
        "silence_ratio":      float(np.mean(np.abs(audio) < 0.01)),
        "clipping_ratio":     float(np.mean(np.abs(audio) >= 0.99)),
        "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio)[0])),
        "dynamic_range":      float(np.percentile(audio, 95) - np.percentile(audio, 5)),
    }


def _pad_to_min(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    mn = int(MIN_DURATION_SEC * sr)
    if not len(audio):
        return np.zeros(mn, dtype=np.float32)
    return audio if len(audio) >= mn else np.pad(audio, (0, mn - len(audio))).astype(np.float32)
