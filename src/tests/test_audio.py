"""
Module: src.tests.test_audio
SRS Reference: FR-AUD-001-011, NFR-Performance
SDLC Phase: 4 - Audio DSP Implementation
Sprint: A
Pipeline Stage: Audio DSP
Interface Contract:
Input: Temporary WAV files generated in test runtime
Output: Assertions on DSP output tensor [3, 224, 224] float32 and error behavior
Latency Target: <= 500 ms per FR/NFR gate (test-based verification)
Open Questions Resolved: Q2, Q3
Open Questions Blocking: None
MCP Tools Used: context7-mcp (librosa)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-24
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from src.audio.dsp import fix_duration, preprocess_audio, to_mono
from src.utils.errors import DSDBAError


def _load_cfg() -> dict:
    """Load project config.yaml for test-driven contract checks."""
    root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((root / "config.yaml").read_text())


def _write_wav(path: Path, sample_rate: int, waveform: np.ndarray) -> None:
    """Write waveform to WAV file."""
    sf.write(path, waveform, samplerate=sample_rate, subtype="PCM_16")


def _sine(seconds: float, sample_rate: int, freq: float = 440.0) -> np.ndarray:
    """Generate mono sine waveform."""
    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False, dtype=np.float32)
    return 0.1 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def test_too_short_audio(tmp_path: Path) -> None:
    """0.4 s clip raises DSDBAError AUD-001 per FR-AUD-005."""
    cfg = _load_cfg()
    sr = int(cfg["audio"]["sample_rate"])
    wav_path = tmp_path / "too_short.wav"
    _write_wav(wav_path, sr, _sine(seconds=0.4, sample_rate=sr))

    with pytest.raises(DSDBAError) as exc_info:
        preprocess_audio(wav_path, cfg)
    assert exc_info.value.code == "AUD-001"


def test_exact_duration(tmp_path: Path) -> None:
    """2.0 s clip is no-op in fix_duration and keeps output contract."""
    cfg = _load_cfg()
    sr = int(cfg["audio"]["sample_rate"])
    n_samples = int(cfg["audio"]["n_samples"])

    fixed = fix_duration(_sine(seconds=2.0, sample_rate=sr), cfg)
    assert fixed.shape[0] == n_samples
    np.testing.assert_allclose(fixed, _sine(seconds=2.0, sample_rate=sr), atol=1e-6)

    wav_path = tmp_path / "exact.wav"
    _write_wav(wav_path, sr, _sine(seconds=2.0, sample_rate=sr))

    tensor = preprocess_audio(wav_path, cfg)
    assert tuple(tensor.shape) == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_long_audio(tmp_path: Path) -> None:
    """3.5 s clip is center-cropped and keeps output contract."""
    cfg = _load_cfg()
    sr = int(cfg["audio"]["sample_rate"])
    n_samples = int(cfg["audio"]["n_samples"])

    long_wave = _sine(seconds=3.5, sample_rate=sr)
    fixed = fix_duration(long_wave, cfg)
    start = (long_wave.shape[0] - n_samples) // 2
    np.testing.assert_allclose(fixed, long_wave[start : start + n_samples], atol=1e-6)

    wav_path = tmp_path / "long.wav"
    _write_wav(wav_path, sr, long_wave)

    tensor = preprocess_audio(wav_path, cfg)
    assert tuple(tensor.shape) == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_short_padded(tmp_path: Path) -> None:
    """1.0 s clip is right-padded and keeps output contract."""
    cfg = _load_cfg()
    sr = int(cfg["audio"]["sample_rate"])
    n_samples = int(cfg["audio"]["n_samples"])

    short_wave = _sine(seconds=1.0, sample_rate=sr)
    fixed = fix_duration(short_wave, cfg)
    assert fixed.shape[0] == n_samples
    np.testing.assert_allclose(fixed[: short_wave.shape[0]], short_wave, atol=1e-6)
    assert np.allclose(fixed[short_wave.shape[0] :], 0.0, atol=1e-7)

    wav_path = tmp_path / "short.wav"
    _write_wav(wav_path, sr, short_wave)

    tensor = preprocess_audio(wav_path, cfg)
    assert tuple(tensor.shape) == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_stereo_to_mono() -> None:
    """Stereo input is reduced to mono without shape errors."""
    sr = 16000
    mono = _sine(seconds=1.0, sample_rate=sr)
    stereo = np.stack([mono, 0.5 * mono], axis=0).astype(np.float32)
    reduced = to_mono(stereo)

    assert reduced.ndim == 1
    assert reduced.shape[0] == mono.shape[0]
    assert reduced.dtype == np.float32


def test_output_contract(tmp_path: Path) -> None:
    """Valid input yields torch.Tensor [3,224,224] float32 per FR-AUD-008."""
    cfg = _load_cfg()
    sr = int(cfg["audio"]["sample_rate"])
    wav_path = tmp_path / "contract.wav"
    _write_wav(wav_path, sr, _sine(seconds=2.0, sample_rate=sr))

    tensor = preprocess_audio(wav_path, cfg)
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (3, 224, 224)
    assert tensor.dtype == torch.float32


@pytest.mark.timeout(2)
def test_latency(tmp_path: Path) -> None:
    """2.0 s WAV is processed <= 500 ms on CPU (NFR-Performance)."""
    cfg = _load_cfg()
    sr = int(cfg["audio"]["sample_rate"])
    wav_path = tmp_path / "latency.wav"
    _write_wav(wav_path, sr, _sine(seconds=2.0, sample_rate=sr))

    start = time.perf_counter()
    _ = preprocess_audio(wav_path, cfg)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms <= 500.0
