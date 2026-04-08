"""
Module: src.tests.test_e2e
SRS Reference: FR-DEP-001–010 (end-to-end verification)
SDLC Phase: 4 - Deployment UI (Sprint E)
Sprint: E
Pipeline Stage: Deployment
Purpose: End-to-end-ish tests for `app.py` pipeline wiring under CPU constraints.
MCP Tools Used: N/A (tests)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-04-02
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml


def _load_cfg() -> dict[str, Any]:
  root = Path(__file__).resolve().parents[2]
  return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
  return _load_cfg()


@pytest.mark.timeout(20)
def test_file_too_large_rejected_before_processing(cfg: dict[str, Any], tmp_path: Path) -> None:
  # Create a 21 MB dummy file.
  big = tmp_path / "big.wav"
  big.write_bytes(b"\x00" * (21 * 1024 * 1024))

  import app as dsdba_app

  with pytest.raises(ValueError) as exc:
    dsdba_app.run_pipeline(big, cfg=dsdba_app.CFG, onnx_session=dsdba_app.ONNX_SESSION, model=dsdba_app.MODEL)
  assert str(exc.value) == "FILE_TOO_LARGE"


@pytest.mark.asyncio
async def test_nlp_fallback_does_not_block_cv(cfg: dict[str, Any]) -> None:
  import app as dsdba_app

  samples = dsdba_app.ensure_demo_samples(dsdba_app.CFG)
  assert len(samples) >= 1

  started = asyncio.Event()
  unblock = asyncio.Event()

  async def _slow_explain(*args: Any, **kwargs: Any):
    started.set()
    await unblock.wait()
    return ("rule-based", False)

  with patch("app.generate_explanation", new=_slow_explain):
    label, confidence, spec_path, heatmap_path, band_pct, task = dsdba_app.run_pipeline(
      samples[0],
      cfg=dsdba_app.CFG,
      onnx_session=dsdba_app.ONNX_SESSION,
      model=dsdba_app.MODEL,
    )
    # CV outputs exist even though NLP task is pending.
    assert isinstance(label, str)
    assert 0.0 <= float(confidence) <= 1.0
    assert Path(spec_path).exists()
    assert Path(heatmap_path).exists()
    assert isinstance(band_pct, dict)

    await started.wait()
    assert not task.done()
    unblock.set()
    explanation_text, api_used = await task
    assert isinstance(explanation_text, str)
    assert api_used is False


@pytest.mark.timeout(30)
def test_e2e_latency_budget(cfg: dict[str, Any]) -> None:
  import app as dsdba_app

  samples = dsdba_app.ensure_demo_samples(dsdba_app.CFG)
  start = time.perf_counter()
  _ = dsdba_app.run_pipeline(samples[0], cfg=dsdba_app.CFG, onnx_session=dsdba_app.ONNX_SESSION, model=dsdba_app.MODEL)
  elapsed_ms = (time.perf_counter() - start) * 1000.0
  assert elapsed_ms <= float(dsdba_app.CFG["deployment"]["e2e_latency_target_ms"])

