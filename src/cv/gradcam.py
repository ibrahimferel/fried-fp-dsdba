"""
Module: src.cv.gradcam
SRS Reference: FR-CV-010 | FR-CV-011 | FR-CV-012 | FR-CV-013 | FR-CV-014 | FR-CV-015 | FR-CV-016
SDLC Phase: 4 — Implementation (Sprint C)
Sprint: C
Pipeline Stage: XAI (Grad-CAM + Mel band attribution)
Interface Contract:
  Input: torch.Tensor [3, 224, 224] float32 + DSDBAModel + config
  Output: run_gradcam → (heatmap PNG Path, band_pct dict with 4 keys summing to 100.0 ± 0.001)
Latency Target: ≤ 3,000 ms Grad-CAM on CPU (FR-CV-015)
Open Questions Resolved: Q4 target layer; Q5 mel Hz ↔ bin mapping
Open Questions Blocking: None
MCP Tools Used: context7-mcp (pytorch-grad-cam, librosa)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-29
"""

from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import librosa
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
# Import from submodule to avoid importing optional sklearn-backed features at package import time.
# `pytorch_grad_cam.__init__` pulls in Deep Feature Factorization (sklearn), which can break on
# mismatched numpy/sklearn versions in Colab environments.
from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor, nn

from src.cv.model import DSDBAModel


_BAND_ORDER: tuple[str, ...] = ("low", "low_mid", "high_mid", "high")


def get_target_layer(model: DSDBAModel, cfg: dict[str, Any]) -> nn.Module:
  """
  Resolve Grad-CAM target layer from `cfg["gradcam"]["target_layer"]` (Q4).

  The path is evaluated with restricted ``eval`` (only the name ``model`` is bound to
  ``DSDBAModel``; no builtins). Config must use the backbone path, e.g.
  ``model.backbone.features[8]``.
  """
  path = str(cfg["gradcam"]["target_layer"]).strip()
  return eval(path, {"__builtins__": {}}, {"model": model})


def compute_gradcam(model: DSDBAModel, tensor: Tensor, cfg: dict[str, Any]) -> np.ndarray:
  """
  Compute Grad-CAM saliency with pytorch-grad-cam (FR-CV-011).

  Returns:
    ``np.ndarray`` of shape ``(H, W)`` matching ``audio.output_tensor_shape``, values in ``[0, 1]``.
  """
  target_layer = get_target_layer(model, cfg)
  device = next(model.parameters()).device
  x = tensor.unsqueeze(0).to(device=device, dtype=torch.float32) if tensor.ndim == 3 else tensor.to(device)
  if x.ndim != 4:
    raise ValueError("CAM-001 per FR-CV-011: expected tensor [3,224,224] or [1,3,224,224]")

  model.eval()
  use_cuda = device.type == "cuda"
  cls_idx = int(cfg["gradcam"]["cam_target_class"])
  cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda)
  targets = [ClassifierOutputTarget(cls_idx)]

  try:
    with torch.set_grad_enabled(True):
      grayscale = cam(input_tensor=x, targets=targets)
  finally:
    if hasattr(cam, "release"):
      cam.release()

  while isinstance(grayscale, (list, tuple)):
    grayscale = grayscale[0]
  if isinstance(grayscale, torch.Tensor):
    grayscale = grayscale.detach().cpu().numpy()
  while grayscale.ndim > 2:
    grayscale = grayscale[0]

  g = np.asarray(grayscale, dtype=np.float64)
  gmin, gmax = float(np.min(g)), float(np.max(g))
  if gmax - gmin < 1e-12:
    g = np.full_like(g, 0.5, dtype=np.float64)
  else:
    g = (g - gmin) / (gmax - gmin)

  h = int(cfg["audio"]["output_tensor_shape"][1])
  w = int(cfg["audio"]["output_tensor_shape"][2])
  if g.shape != (h, w):
    t = torch.from_numpy(g.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    g = t.squeeze().numpy().astype(np.float64)

  return g.astype(np.float32)


def create_heatmap_overlay(tensor: Tensor, saliency: np.ndarray, cfg: dict[str, Any]) -> Path:
  """Jet overlay on RGB tensor; save PNG (FR-CV-012)."""
  alpha = float(cfg["gradcam"]["overlay_alpha"])
  cmap_name = str(cfg["gradcam"]["colormap"])
  root = Path(__file__).resolve().parents[2]
  rel = Path(str(cfg["gradcam"]["heatmap_output_dir"]))
  out_dir = root / rel
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / f"gradcam_{uuid.uuid4().hex}.png"

  rgb = tensor.detach().cpu().float().numpy()
  if rgb.ndim == 3:
    rgb = np.transpose(rgb, (1, 2, 0))
  rgb = np.clip(rgb, 0.0, 1.0)

  cmap = cm.get_cmap(cmap_name)
  heat = cmap(np.clip(saliency, 0.0, 1.0))[:, :, :3]

  blended = (1.0 - alpha) * rgb + alpha * heat
  blended_u8 = (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)
  Image.fromarray(blended_u8).save(out_path, format=str(cfg["gradcam"]["output_format"]).upper())
  return out_path


def get_mel_band_row_indices(cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
  """
  Map Hz bands to row index ranges on the resized spectrogram height (FR-CV-013, Q5).

  Uses ``librosa.mel_frequencies`` for bin centers, then maps bin index ranges to
  ``[0, H)`` rows where ``H`` is ``audio.output_tensor_shape[1]`` (224).
  """
  n_mels = int(cfg["audio"]["n_mels"])
  fmin = float(cfg["gradcam"]["mel_fmin_hz"])
  fmax = float(cfg["gradcam"]["mel_fmax_hz"])
  mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
  h = int(cfg["audio"]["output_tensor_shape"][1])
  band_hz = cfg["gradcam"]["band_hz"]
  out: dict[str, tuple[int, int]] = {}

  for name in _BAND_ORDER:
    lo, hi = float(band_hz[name][0]), float(band_hz[name][1])
    if name == "high":
      mask = (mel_freqs >= lo) & (mel_freqs <= hi)
    else:
      mask = (mel_freqs >= lo) & (mel_freqs < hi)
    indices = np.where(mask)[0]
    if indices.size == 0:
      j_min = j_max = 0
    else:
      j_min, j_max = int(indices.min()), int(indices.max())
    r0 = int(j_min * h / n_mels)
    r1 = int((j_max + 1) * h / n_mels) - 1
    r1 = min(max(r1, r0), h - 1)
    out[name] = (r0, r1)

  return out


def compute_band_attributions(saliency: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
  """Sum saliency per band then softmax → 100% (FR-CV-014)."""
  rows = get_mel_band_row_indices(cfg)
  raw: list[float] = []
  for name in _BAND_ORDER:
    r0, r1 = rows[name]
    patch = saliency[r0 : r1 + 1, :]
    raw.append(float(np.sum(patch)))

  x = np.asarray(raw, dtype=np.float64)
  x = x - np.max(x)
  ex = np.exp(x)
  denom = float(np.sum(ex))
  if denom < 1e-12:
    p = np.ones(4, dtype=np.float64) * 25.0
  else:
    p = (ex / denom) * 100.0

  result = {name: float(p[i]) for i, name in enumerate(_BAND_ORDER)}
  total = float(sum(result.values()))
  if not math.isclose(total, 100.0, rel_tol=0.0, abs_tol=0.001):
    raise ValueError("CAM-002 per FR-CV-014: band softmax must sum to 100.0 ± 0.001")
  return result


def get_raw_saliency_json(saliency: np.ndarray) -> dict[str, Any]:
  """Nested list + metadata for JSON APIs (FR-CV-016 SHOULD)."""
  return {
    "saliency": saliency.tolist(),
    "shape": list(saliency.shape),
    "dtype": str(saliency.dtype),
  }


def save_saliency_json(saliency: np.ndarray, cfg: dict[str, Any]) -> Path:
  """
  Persist raw saliency payload as a JSON file (FR-CV-016 SHOULD).

  Output location follows ``cfg["gradcam"]["heatmap_output_dir"]`` to keep Grad-CAM artifacts together.

  Returns:
    Path to the written ``.json`` file.
  """
  root = Path(__file__).resolve().parents[2]
  rel = Path(str(cfg["gradcam"]["heatmap_output_dir"]))
  out_dir = root / rel
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / f"saliency_{uuid.uuid4().hex}.json"

  payload = get_raw_saliency_json(saliency)
  out_path.write_text(json.dumps(payload), encoding="utf-8")
  return out_path


def run_gradcam(tensor: Tensor, model: DSDBAModel, cfg: dict[str, Any]) -> tuple[Path, dict[str, float]]:
  """
  Full XAI path: Grad-CAM → heatmap PNG → 4-band % (phase2-interface-contracts.md).

  Latency:
    Should stay within ``gradcam.latency_target_ms`` on CPU (FR-CV-015); callers may time externally.
  """
  saliency = compute_gradcam(model, tensor, cfg)
  heatmap_path = create_heatmap_overlay(tensor, saliency, cfg)
  bands = compute_band_attributions(saliency, cfg)
  return heatmap_path, bands


def run_gradcam_timed(
  tensor: Tensor, model: DSDBAModel, cfg: dict[str, Any]
) -> tuple[Path, dict[str, float], float]:
  """Same as ``run_gradcam`` but returns elapsed wall time in ms (tests, FR-CV-015)."""
  t0 = time.perf_counter()
  path, bands = run_gradcam(tensor, model, cfg)
  elapsed_ms = (time.perf_counter() - t0) * 1000.0
  return path, bands, float(elapsed_ms)
