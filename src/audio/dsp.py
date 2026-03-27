"""
Module: src.audio.dsp
SRS Reference: FR-AUD-001-011
SDLC Phase: 3 - Environment Setup & MCP Configuration
Sprint: A
Pipeline Stage: Audio DSP
Purpose: Convert input audio into the fixed Mel-spectrogram tensor contract consumed by the CV module.
Dependencies: librosa, numpy, soundfile, torch.
Interface Contract:
  Input:  Path to WAV or FLAC file
  Output: torch.Tensor [3, 224, 224] float32
Latency Target: <= 500 ms per NFR-Performance
Open Questions Resolved: None (module scaffold)
Open Questions Blocking: None for Sprint A (Q3 affects training only in Sprint B)
MCP Tools Used: context7-mcp (librosa) | huggingface-mcp | stitch-mcp
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-22
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F

from src.utils.errors import DSDBAError
from src.utils.logger import log_info


def _audio_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return audio section from project configuration."""
    return cfg["audio"]


def load_audio(file_path: Path, cfg: dict[str, Any]) -> tuple[np.ndarray, int]:
    """
    [FR-AUD-001, FR-AUD-011] Load waveform from disk and validate supported format.

    Args:
        file_path: Path to input audio file.
        cfg: Full config mapping.

    Returns:
        Tuple `(waveform_np, sample_rate)` where waveform is `[channels, samples]`.

    Raises:
        DSDBAError: AUD-002 for unsupported/corrupt audio input.
    """

    audio_cfg = _audio_cfg(cfg)
    ext = file_path.suffix.lower().lstrip(".")
    supported = set(audio_cfg["supported_formats"]) | set(audio_cfg["optional_formats"])
    if ext not in supported:
        raise DSDBAError(code=str(audio_cfg["error_code_bad_format"]), message="Unsupported format", stage="audio_dsp")

    try:
        waveform, sample_rate = torchaudio.load(str(file_path))
    except Exception as exc:
        raise DSDBAError(code=str(audio_cfg["error_code_bad_format"]), message="Failed to load audio", stage="audio_dsp") from exc

    waveform_np = waveform.numpy().astype(np.float32, copy=False)
    if waveform_np.size == 0:
        raise DSDBAError(code=str(audio_cfg["error_code_bad_format"]), message="Empty audio payload", stage="audio_dsp")
    return waveform_np, int(sample_rate)


def validate_duration(waveform: np.ndarray, sample_rate: int, cfg: dict[str, Any]) -> None:
    """
    [FR-AUD-005] Validate minimum clip duration threshold.

    Args:
        waveform: Audio array shaped `[channels, samples]` or `[samples]`.
        sample_rate: Sampling rate in Hz.
        cfg: Full config mapping.

    Returns:
        None.

    Raises:
        DSDBAError: AUD-001 when duration is below minimum threshold.
    """

    audio_cfg = _audio_cfg(cfg)
    samples = waveform.shape[-1]
    duration_sec = float(samples) / float(sample_rate)
    if duration_sec < float(audio_cfg["min_duration_sec"]):
        raise DSDBAError(code=str(audio_cfg["error_code_too_short"]), message="Audio too short", stage="audio_dsp")


def resample_audio(waveform: np.ndarray, orig_sr: int, cfg: dict[str, Any]) -> np.ndarray:
    """
    [FR-AUD-002] Resample waveform to configured sample rate using configured method.

    Args:
        waveform: Audio array shaped `[channels, samples]` or `[samples]`.
        orig_sr: Original sample rate.
        cfg: Full config mapping.

    Returns:
        Resampled waveform as `np.ndarray` preserving channel-first representation.
    """

    audio_cfg = _audio_cfg(cfg)
    target_sr = int(audio_cfg["sample_rate"])
    resampling_method = str(audio_cfg.get("resampling_method", "kaiser_best"))

    if orig_sr == target_sr:
        return waveform.astype(np.float32, copy=False)

    return librosa.resample(
        y=waveform.astype(np.float32, copy=False),
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type=resampling_method,
        axis=-1,
    ).astype(np.float32, copy=False)


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """
    [FR-AUD-003] Convert multi-channel waveform to mono by averaging channels.

    Args:
        waveform: Audio array shaped `[channels, samples]` or `[samples]`.

    Returns:
        Mono waveform shaped `[samples]` float32.
    """

    if waveform.ndim == 1:
        return waveform.astype(np.float32, copy=False)
    return waveform.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)


def fix_duration(waveform: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """
    [FR-AUD-004] Force waveform length to configured sample count.

    Args:
        waveform: Mono waveform shaped `[samples]`.
        cfg: Full config mapping.

    Returns:
        Waveform with exactly `audio.n_samples` samples.
    """

    target_samples = int(_audio_cfg(cfg)["n_samples"])
    current_samples = int(waveform.shape[-1])

    if current_samples == target_samples:
        return waveform.astype(np.float32, copy=False)

    if current_samples > target_samples:
        start = (current_samples - target_samples) // 2
        end = start + target_samples
        return waveform[start:end].astype(np.float32, copy=False)

    pad_len = target_samples - current_samples
    return np.pad(waveform, (0, pad_len), mode="constant").astype(np.float32, copy=False)


def extract_mel_spectrogram(waveform: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """
    [FR-AUD-006] Compute Mel spectrogram from mono waveform.

    Args:
        waveform: Mono waveform shaped `[samples]`.
        cfg: Full config mapping.

    Returns:
        Mel power spectrogram shaped `[n_mels, time_frames]`.
    """

    audio_cfg = _audio_cfg(cfg)
    return librosa.feature.melspectrogram(
        y=waveform.astype(np.float32, copy=False),
        sr=int(audio_cfg["sample_rate"]),
        n_mels=int(audio_cfg["n_mels"]),
        n_fft=int(audio_cfg["n_fft"]),
        hop_length=int(audio_cfg["hop_length"]),
        window=str(audio_cfg["window"]),
        power=2.0,
    ).astype(np.float32, copy=False)


def normalise_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    [FR-AUD-007] Convert power spectrogram to dB and min-max normalise to [0, 1].

    Args:
        spec: Mel power spectrogram.

    Returns:
        Normalised spectrogram in range [0, 1], float32.
    """

    db = librosa.power_to_db(spec, ref=np.max)
    min_val = float(np.min(db))
    max_val = float(np.max(db))
    denom = max_val - min_val
    if denom <= 1e-12:
        return np.zeros_like(db, dtype=np.float32)
    norm = (db - min_val) / denom
    return norm.astype(np.float32, copy=False)


def to_tensor(spec: np.ndarray, cfg: dict[str, Any]) -> torch.Tensor:
    """
    [FR-AUD-008] Convert spectrogram to tensor contract `[3, 224, 224]` float32.

    Args:
        spec: Normalised spectrogram shaped `[n_mels, time_frames]`.
        cfg: Full config mapping.

    Returns:
        Torch tensor with shape `[3, 224, 224]`, dtype `torch.float32`.
    """

    audio_cfg = _audio_cfg(cfg)
    out_shape = tuple(int(v) for v in audio_cfg["output_tensor_shape"])
    tensor = torch.from_numpy(spec.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=(out_shape[1], out_shape[2]),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.squeeze(0).repeat(out_shape[0], 1, 1).to(dtype=torch.float32)
    assert tuple(resized.shape) == out_shape
    assert resized.dtype == torch.float32
    return resized


def preprocess_audio(file_path: Path, cfg: dict[str, Any]) -> torch.Tensor:
    """
    [FR-AUD-010] Full audio preprocessing pipeline entry point.

    Args:
        file_path: Input audio path.
        cfg: Full config mapping.

    Returns:
        Preprocessed tensor contract `[3, 224, 224]` float32.
    """

    start = time.perf_counter()
    waveform, sample_rate = load_audio(file_path=file_path, cfg=cfg)
    validate_duration(waveform=waveform, sample_rate=sample_rate, cfg=cfg)
    resampled = resample_audio(waveform=waveform, orig_sr=sample_rate, cfg=cfg)
    mono = to_mono(resampled)
    fixed = fix_duration(mono, cfg=cfg)
    mel = extract_mel_spectrogram(fixed, cfg=cfg)
    normalised = normalise_spectrogram(mel)
    output = to_tensor(normalised, cfg=cfg)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    channel_count = int(waveform.shape[0]) if waveform.ndim > 1 else 1
    duration_sec = float(waveform.shape[-1]) / float(sample_rate)
    log_info(
        stage="audio_dsp",
        message="audio_preprocess_complete",
        data={
            "sample_rate": sample_rate,
            "duration_sec": round(duration_sec, 6),
            "channel_count": channel_count,
            "latency_ms": round(elapsed_ms, 3),
        },
    )
    return output


def batch_preprocess(file_paths: list[Path], cfg: dict[str, Any]) -> list[torch.Tensor]:
    """
    [FR-AUD-009] Batch wrapper for preprocessing multiple files.

    Args:
        file_paths: List of input audio paths.
        cfg: Full config mapping.

    Returns:
        List of tensors, one per input file.
    """

    return [preprocess_audio(file_path=path, cfg=cfg) for path in file_paths]