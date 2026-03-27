"""
Module: src.cv.infer
SRS Reference: FR-DEP-010
SDLC Phase: 3 - Environment Setup & MCP Configuration
Sprint: B
Pipeline Stage: CV Inference
Purpose: Run CPU inference using ONNX Runtime and return label + confidence for the CV stage.
Dependencies: onnxruntime, numpy, torch.
Interface Contract:
  Input:  torch.Tensor [3, 224, 224] float32
  Output: tuple[str, float] (label: bonafide|spoof, confidence: float in (0,1))
Latency Target: <= 1,500 ms on CPU per FR-DEP-010
Open Questions Resolved: Q3/Q4/Q5/Q6 resolved only for interface contracts (runtime still pending Q3 empirical check)
Open Questions Blocking: Q3 - affects feasible training/export cycle before serving
MCP Tools Used: context7-mcp | huggingface-mcp | stitch-mcp
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-22
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


def export_to_onnx(model: DSDBAModel, cfg: dict[str, Any]) -> Path:
  """[FR-DEP-010] Export trained PyTorch model to ONNX format."""
  root = Path(__file__).resolve().parents[2]
  out_dir = root / "models" / "checkpoints"
  out_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = out_dir / "dsdba_efficientnet_b4.onnx"

  in_shape = tuple(int(v) for v in cfg["audio"]["output_tensor_shape"])
  dummy = torch.randn(1, *in_shape, dtype=torch.float32)

  model.eval()
  torch.onnx.export(
    model,
    dummy,
    str(onnx_path),
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=13,
  )
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
  x = torch.randn(1, *in_shape, dtype=torch.float32)

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
  confidence = spoof_score if label == "spoof" else (1.0 - spoof_score)
  return label, float(confidence)


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
  confidence = spoof_prob if label == "spoof" else (1.0 - spoof_prob)
  return label, float(confidence)


def timed_onnx_inference(session: ort.InferenceSession, tensor: Tensor, cfg: dict[str, Any]) -> tuple[tuple[str, float], float]:
  """Run ONNX inference and return ((label, confidence), latency_ms)."""
  start = time.perf_counter()
  result = run_onnx_inference(session, tensor, cfg)
  latency_ms = (time.perf_counter() - start) * 1000.0
  return result, float(latency_ms)