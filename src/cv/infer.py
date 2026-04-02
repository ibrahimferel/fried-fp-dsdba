"""
Module: src.cv.infer
SRS Reference: FR-DEP-010 | FR-CV-004
SDLC Phase: 4 — Implementation (Sprint B)
Sprint: B
Pipeline Stage: CV Inference
Interface Contract:
  Input: torch.Tensor [3, 224, 224] float32 (batched as [1,3,224,224] internally as needed)
  Output: tuple[str, float] label in {bonafide, spoof}, confidence in (0, 1)
Latency Target: ≤ 1,500 ms ONNX CPU (FR-DEP-010, NFR-Scalability)
Open Questions Resolved: Q3 VRAM; Q6 Gradio
Open Questions Blocking: None for ONNX export path
MCP Tools Used: context7-mcp (torch.onnx) | huggingface-mcp (artifact hosting)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-29
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor

from src.cv.model import DSDBAModel


def _clamp_open_unit_interval(raw: float, cfg: dict[str, Any]) -> float:
  """Map a probability into (0, 1) using `model.confidence_epsilon` from config."""
  eps = float(cfg["model"]["confidence_epsilon"])
  x = float(raw)
  return min(max(x, eps), 1.0 - eps)


def export_to_onnx(model: DSDBAModel, cfg: dict[str, Any]) -> Path:
  """[FR-DEP-010] Export trained PyTorch model to ONNX format."""
  root = Path(__file__).resolve().parents[2]
  out_dir = root / "models" / "checkpoints"
  out_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = out_dir / "dsdba_efficientnet_b4.onnx"

  in_shape = tuple(int(v) for v in cfg["audio"]["output_tensor_shape"])
  model.eval()
  device = next(model.parameters()).device
  dummy = torch.randn(1, *in_shape, dtype=torch.float32, device=device)

  export_kw: dict[str, Any] = {
    "model": model,
    "args": dummy,
    "f": str(onnx_path),
    "input_names": ["input"],
    "output_names": ["logits"],
    "dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}},
    "opset_version": int(cfg["deployment"]["onnx_opset_version"]),
    "do_constant_folding": True,
  }
  try:
    torch.onnx.export(**export_kw, dynamo=False)
  except TypeError:
    torch.onnx.export(**export_kw)
  return onnx_path


def load_onnx_session(onnx_path: Path, cfg: dict[str, Any]) -> ort.InferenceSession:
  """[FR-DEP-010] Load ONNX Runtime session with CPUExecutionProvider only."""
  providers = list(cfg["deployment"].get("onnx_execution_providers", ["CPUExecutionProvider"]))
  cpu_only = [p for p in providers if p == "CPUExecutionProvider"]
  if not cpu_only:
    cpu_only = ["CPUExecutionProvider"]
  session = ort.InferenceSession(str(onnx_path), providers=cpu_only)
  return session


def verify_onnx_equivalence(model: DSDBAModel, onnx_path: Path, cfg: dict[str, Any]) -> bool:
  """Check ONNX vs PyTorch output difference on the same input sample."""
  tol = float(cfg["deployment"].get("onnx_equivalence_tolerance", 1.0e-5))
  session = load_onnx_session(onnx_path, cfg)

  in_shape = tuple(int(v) for v in cfg["audio"]["output_tensor_shape"])
  device = next(model.parameters()).device
  x = torch.randn(1, *in_shape, dtype=torch.float32, device=device)

  model.eval()
  with torch.no_grad():
    pt_out = model(x).detach().cpu().numpy()

  ort_out = session.run(None, {session.get_inputs()[0].name: x.detach().cpu().numpy()})[0]
  max_abs = float(np.max(np.abs(pt_out - ort_out)))
  return max_abs < tol


def run_onnx_inference(session: ort.InferenceSession, tensor: Tensor, cfg: dict[str, Any]) -> tuple[str, float]:
  """Run ONNX inference and return (label, confidence)."""
  if tensor.ndim == 3:
    tensor = tensor.unsqueeze(0)

  arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
  output = session.run(None, {session.get_inputs()[0].name: arr})[0]
  logits = output[:, 1]
  spoof_prob = 1.0 / (1.0 + np.exp(-logits))

  threshold = float(cfg["model"].get("decision_threshold", 0.5))
  spoof_score = float(spoof_prob[0])
  label = "spoof" if spoof_score >= threshold else "bonafide"
  raw_confidence = spoof_score if label == "spoof" else (1.0 - spoof_score)
  confidence = _clamp_open_unit_interval(raw_confidence, cfg)
  return label, confidence


def run_inference(tensor: Tensor, model: DSDBAModel, cfg: dict[str, Any]) -> tuple[str, float]:
  """Run PyTorch inference for quick validation path."""
  if tensor.ndim == 3:
    tensor = tensor.unsqueeze(0)

  model.eval()
  with torch.no_grad():
    logits = model(tensor)
    spoof_prob = torch.sigmoid(logits[:, 1]).item()

  threshold = float(cfg["model"].get("decision_threshold", 0.5))
  label = "spoof" if spoof_prob >= threshold else "bonafide"
  raw_confidence = spoof_prob if label == "spoof" else (1.0 - spoof_prob)
  confidence = _clamp_open_unit_interval(raw_confidence, cfg)
  return label, confidence


def timed_onnx_inference(session: ort.InferenceSession, tensor: Tensor, cfg: dict[str, Any]) -> tuple[tuple[str, float], float]:
  """Run ONNX inference and return ((label, confidence), latency_ms)."""
  start = time.perf_counter()
  result = run_onnx_inference(session, tensor, cfg)
  latency_ms = (time.perf_counter() - start) * 1000.0
  return result, float(latency_ms)