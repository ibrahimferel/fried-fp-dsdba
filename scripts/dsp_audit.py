from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from statistics import mean

import torch
import yaml

from src.audio.dsp import preprocess_audio
from src.utils.errors import DSDBAError


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DSP preprocessing audit on dataset splits")
    parser.add_argument("--max-per-bucket", type=int, default=120, help="Max files sampled per split/class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Silence per-file INFO logs from DSP to keep audit output readable.
    logging.getLogger("dsdba").setLevel(logging.WARNING)

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text())

    random.seed(args.seed)
    exts = {".wav"}

    buckets = [
        ("train", "bonafide"),
        ("train", "spoof"),
        ("validation", "bonafide"),
        ("validation", "spoof"),
        ("test", "bonafide"),
        ("test", "spoof"),
    ]

    summary: dict[tuple[str, str], dict[str, object]] = {}
    all_latencies: list[float] = []
    failures: list[tuple[str, str]] = []

    for split, label in buckets:
        folder = root / "data" / split / label
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        sampled = files if len(files) <= args.max_per_bucket else random.sample(files, args.max_per_bucket)

        ok = 0
        failed = 0
        latencies: list[float] = []
        err_codes: dict[str, int] = {}

        for fp in sampled:
            t0 = time.perf_counter()
            try:
                tensor = preprocess_audio(fp, cfg)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt_ms)
                all_latencies.append(dt_ms)

                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError("NON_TENSOR")
                if tuple(tensor.shape) != (3, 224, 224):
                    raise RuntimeError(f"BAD_SHAPE:{tuple(tensor.shape)}")
                if tensor.dtype != torch.float32:
                    raise RuntimeError(f"BAD_DTYPE:{tensor.dtype}")
                ok += 1
            except DSDBAError as exc:
                failed += 1
                err_codes[exc.code] = err_codes.get(exc.code, 0) + 1
                failures.append((str(fp), exc.code))
            except Exception as exc:  
                failed += 1
                code = type(exc).__name__
                err_codes[code] = err_codes.get(code, 0) + 1
                failures.append((str(fp), code))

        lat_sorted = sorted(latencies)
        p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else None

        summary[(split, label)] = {
            "total": len(files),
            "sampled": len(sampled),
            "ok": ok,
            "failed": failed,
            "avg_ms": round(mean(latencies), 3) if latencies else None,
            "p95_ms": round(p95, 3) if p95 is not None else None,
            "max_ms": round(max(latencies), 3) if latencies else None,
            "errors": err_codes,
        }

    print("=== DSP PREPROCESS AUDIT ===")
    print(f"max_per_bucket={args.max_per_bucket}")

    for (split, label), metrics in summary.items():
        print(
            f"[{split}/{label}] total={metrics['total']} sampled={metrics['sampled']} "
            f"ok={metrics['ok']} failed={metrics['failed']} avg_ms={metrics['avg_ms']} "
            f"p95_ms={metrics['p95_ms']} max_ms={metrics['max_ms']} errors={metrics['errors']}"
        )

    if all_latencies:
        all_sorted = sorted(all_latencies)
        p95_all = all_sorted[int(0.95 * (len(all_sorted) - 1))]
        print(
            f"ALL avg_ms={round(mean(all_latencies), 3)} p95_ms={round(p95_all, 3)} "
            f"max_ms={round(max(all_latencies), 3)} n={len(all_latencies)}"
        )

    print(f"TOTAL failures={len(failures)}")
    if failures:
        print("Top 10 failures:")
        for fp, code in failures[:10]:
            print(f"- {code}: {fp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
