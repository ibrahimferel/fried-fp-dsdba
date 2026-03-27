from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.audio.dsp import preprocess_audio


def _render_single(audio_path: Path, out_dir: Path, cfg: dict) -> None:
    tensor = preprocess_audio(audio_path, cfg)
    array = tensor.detach().cpu().numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    npy_path = out_dir / f"{stem}_tensor.npy"
    png_path = out_dir / f"{stem}_3layer_mel.png"

    np.save(npy_path, array)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"DSP Output (shape={tuple(array.shape)}, dtype={array.dtype})")

    for i, ax in enumerate(axes):
        im = ax.imshow(array[i], origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"Channel {i}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel Bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(png_path, dpi=180)

    print(f"Input: {audio_path}")
    print(f"Tensor shape: {tuple(array.shape)}")
    print(f"Tensor dtype: {array.dtype}")
    print(f"Saved array: {npy_path}")
    print(f"Saved preview: {png_path}")
    print(f"Channel0==Channel1: {bool(np.allclose(array[0], array[1]))}")
    print(f"Channel1==Channel2: {bool(np.allclose(array[1], array[2]))}")


def _render_batch(per_class: int, out_dir: Path, cfg: dict, dataset_root: Path) -> None:
    exts = {".wav", ".flac", ".mp3", ".ogg"}
    class_names = ["bonafide", "spoof"]
    selected: dict[str, list[Path]] = {}

    for class_name in class_names:
        class_dir = dataset_root / class_name
        files = [p for p in sorted(class_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
        selected[class_name] = files[:per_class]

    cols = per_class
    rows = len(class_names)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    fig.suptitle("Batch DSP Preview (Channel 0)")

    for r, class_name in enumerate(class_names):
        for c in range(cols):
            ax = axes[r][c]
            files = selected[class_name]
            if c >= len(files):
                ax.axis("off")
                continue

            tensor = preprocess_audio(files[c], cfg)
            array = tensor.detach().cpu().numpy()
            im = ax.imshow(array[0], origin="lower", aspect="auto", cmap="magma")
            ax.set_title(f"{class_name} | {files[c].name[:28]}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Mel Bin")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"batch_preview_{dataset_root.name}_{per_class}x2.png"
    fig.savefig(out_png, dpi=170)
    print(f"Saved batch preview: {out_png}")
    for class_name in class_names:
        print(f"Used {len(selected[class_name])} files from {dataset_root / class_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview DSP output tensor as 3 mel-spectrogram layers")
    parser.add_argument("--audio", type=Path, help="Path to input audio file for single preview mode")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/previews"),
        help="Directory to save preview image and tensor array",
    )
    parser.add_argument(
        "--batch-per-class",
        type=int,
        default=0,
        help="If > 0, render a batch comparison grid using this many files per class",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/samples"),
        help="Dataset root containing bonafide and spoof folders for batch mode",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text())

    if args.batch_per_class > 0:
        _render_batch(
            per_class=args.batch_per_class,
            out_dir=args.out_dir,
            cfg=cfg,
            dataset_root=(root / args.dataset_root),
        )
        return 0

    if args.audio is None:
        raise ValueError("Provide --audio for single preview mode, or use --batch-per-class for batch mode")

    _render_single(audio_path=args.audio, out_dir=args.out_dir, cfg=cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
