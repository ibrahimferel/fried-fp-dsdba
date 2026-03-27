from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.audio.dsp import extract_mel_spectrogram, fix_duration, resample_audio, to_mono


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize waveform before and after Mel transform")
    parser.add_argument("--audio", type=Path, required=True, help="Path to input audio file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/previews"),
        help="Directory for output visualization",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Before: raw waveform from source file.
    raw_waveform, raw_sr = librosa.load(str(args.audio), sr=None, mono=False)
    raw_waveform = raw_waveform.astype(np.float32, copy=False)
    if raw_waveform.ndim == 1:
        raw_mono = raw_waveform
    else:
        raw_mono = raw_waveform.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)

    # After: waveform passed through DSP stages before Mel transform.
    resampled = resample_audio(raw_waveform, raw_sr, cfg)
    mono = to_mono(resampled)
    fixed = fix_duration(mono, cfg)
    mel_spec = extract_mel_spectrogram(fixed, cfg)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    t_raw = np.linspace(0.0, len(raw_mono) / raw_sr, num=len(raw_mono), endpoint=False)
    t_fixed = np.linspace(
        0.0,
        len(fixed) / int(cfg["audio"]["sample_rate"]),
        num=len(fixed),
        endpoint=False,
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    fig.suptitle(f"Before vs After Mel Transform: {args.audio.name}")

    axes[0].plot(t_raw, raw_mono, color="#1f77b4", linewidth=0.8)
    axes[0].set_title(f"Before: Raw Waveform (sr={raw_sr}, samples={len(raw_mono)})")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t_fixed, fixed, color="#ff7f0e", linewidth=0.8)
    axes[1].set_title(
        "After DSP Align: Resampled+Mono+Fixed Duration "
        f"(sr={cfg['audio']['sample_rate']}, samples={len(fixed)})"
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")

    im = axes[2].imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
    axes[2].set_title(f"After Transform: Mel Spectrogram dB (shape={mel_db.shape})")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Mel Bin")
    fig.colorbar(im, ax=axes[2], fraction=0.03, pad=0.02, label="dB")

    fig.tight_layout()
    out_path = out_dir / f"{args.audio.stem}_before_after_mel.png"
    fig.savefig(out_path, dpi=180)

    print(f"Input file: {args.audio}")
    print(f"Saved visualization: {out_path}")
    print(f"Raw shape: {raw_waveform.shape}, raw sr: {raw_sr}")
    print(f"Fixed waveform shape: {fixed.shape}")
    print(f"Mel shape: {mel_spec.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
