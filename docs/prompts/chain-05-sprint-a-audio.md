## CHAIN 05 — SPRINT A: AUDIO DSP MODULE
> **Mode:** Cursor Agent Mode  
> **Blocking Q:** None (Q2 resolved — FoR 2s split confirmed)  
> **Produces:** `src/audio/dsp.py` complete + `src/tests/test_audio.py` + `src/utils/logger.py`

```
# ════════════════════════════════════════════════════════════════════════
# DSDBA Chain 05 — Sprint A: Audio DSP Module
# SRS: FR-AUD-001–011 | Phase: 4-A | Mode: Agent
# V.E.R.I.F.Y. Level: 2 | No blocking Q
# ════════════════════════════════════════════════════════════════════════

@file .cursorrules
@file config.yaml
@file docs/context/session-cheatsheet.md
@file docs/adr/phase2-interface-contracts.md
@file src/audio/dsp.py
@file src/utils/errors.py

[S] SITUATION: Environment setup complete. Implementing Sprint A — the Audio DSP
    module. This is Stage 1 of the pipeline and the sole entry point for raw data.
    Every downstream module depends on its output contract: [3, 224, 224] float32.

[C] CHALLENGE: Implement src/audio/dsp.py satisfying FR-AUD-001–011 (all SHALL
    and SHOULD requirements). Implement src/tests/test_audio.py covering all 6
    required edge cases. Implement src/utils/logger.py.

[F] FOUNDATIONS: Interface contract locked in docs/adr/phase2-interface-contracts.md.
    All parameters from config.yaml (zero magic numbers). Latency ≤ 500 ms on CPU.

Use context7-mcp to verify librosa 0.10.x API signatures before implementing.
# → mcp.json server: "context7" (stdio/npx)

IMPLEMENT: src/utils/logger.py
  Structured JSON logger. Functions: log_info(), log_warning(), log_error().
  Each call produces a JSON dict: {timestamp, level, stage, message, data}.
  No print() calls anywhere in this module.

IMPLEMENT: src/audio/dsp.py
  Implement these functions in order (each must cite its FR ID in docstring):

  1. load_audio(file_path: Path, cfg) -> tuple[np.ndarray, int]
     - Accept WAV, FLAC (SHALL) and optionally MP3, OGG [FR-AUD-001, FR-AUD-011]
     - Use torchaudio.load() — return (waveform_np, sample_rate)
     - Raise DSDBAError(AUD_002) for unsupported formats [FR-AUD-001]

  2. validate_duration(waveform: np.ndarray, sample_rate: int, cfg) -> None
     - Raise DSDBAError(AUD_001) if duration < cfg.audio.min_duration_sec [FR-AUD-005]

  3. resample_audio(waveform: np.ndarray, orig_sr: int, cfg) -> np.ndarray
     - Resample to cfg.audio.sample_rate using Kaiser-best window [FR-AUD-002]
     - Use torchaudio.functional.resample or librosa.resample

  4. to_mono(waveform: np.ndarray) -> np.ndarray
     - Average all channels to mono [FR-AUD-003]

  5. fix_duration(waveform: np.ndarray, cfg) -> np.ndarray
     - Centre-crop if len > cfg.audio.n_samples [FR-AUD-004]
     - Zero-pad right if len < cfg.audio.n_samples [FR-AUD-004]
     - No-op if len == cfg.audio.n_samples

  6. extract_mel_spectrogram(waveform: np.ndarray, cfg) -> np.ndarray
     - STFT → 128-band Mel Spectrogram using cfg.audio params [FR-AUD-006]
     - librosa.feature.melspectrogram(y, sr, n_mels, n_fft, hop_length, window)

  7. normalise_spectrogram(spec: np.ndarray) -> np.ndarray
     - power_to_db(ref=np.max) → min-max normalise to [0, 1] [FR-AUD-007]

  8. to_tensor(spec: np.ndarray, cfg) -> torch.Tensor
     - Stack single-channel to 3-channel, resize to [3, 224, 224] [FR-AUD-008]
     - Validate output: assert shape == (3,224,224) and dtype == float32
     - Return torch.Tensor

  9. preprocess_audio(file_path: Path, cfg) -> torch.Tensor  [MAIN PUBLIC API]
     - Compose all above steps in sequence
     - Log to logger.py: sample_rate, duration, channel_count [FR-AUD-010]
     - Latency target: ≤ 500 ms on CPU [NFR-Performance]

  10. batch_preprocess(file_paths: list[Path], cfg) -> list[torch.Tensor]  [FR-AUD-009 SHOULD]
      - Call preprocess_audio for each path, collect results

IMPLEMENT: src/tests/test_audio.py
  Write pytest tests covering ALL 6 required edge cases:
  1. test_too_short_audio: 0.4 s clip → DSDBAError(AUD-001) raised
  2. test_exact_duration: 2.0 s clip → no crop, no pad, shape [3,224,224]
  3. test_long_audio: 3.5 s clip → centre-crop → shape [3,224,224]
  4. test_short_padded: 1.0 s clip → zero-pad right → shape [3,224,224]
  5. test_stereo_to_mono: stereo WAV → averaged to mono → no shape error
  6. test_output_contract: valid input → output is torch.Tensor, shape [3,224,224], float32
  7. test_latency: valid 2.0 s WAV processed in ≤ 500 ms (use pytest-timeout)

After implementation, run: pytest src/tests/test_audio.py -v
All 7 tests must pass before running Gate Check.
```

---