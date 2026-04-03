"""
Module: src.cv.gradcam
SRS Reference: FR-CV-010-016
SDLC Phase: 3 - Environment Setup & MCP Configuration
Sprint: C
Pipeline Stage: XAI
Purpose: Compute Grad-CAM saliency and convert it into 4 frequency-band attributions for XAI.
Dependencies: torch, pytorch-grad-cam, numpy, Pillow.
Interface Contract:
  Input:  torch.Tensor [3, 224, 224] float32 + trained EfficientNet-B4 model handle
  Output: tuple[Path, dict[str, float]] (heatmap PNG path, band_attributions with 4 keys summing to 100.0)
Latency Target: <= 3,000 ms on CPU per FR-CV-015
Open Questions Resolved: Q4/Q5/Q6 resolved in Phase 2 (Q4 layer path, Q5 Mel mapping, Q6 UI)
Open Questions Blocking: None for Sprint C
MCP Tools Used: context7-mcp | huggingface-mcp | stitch-mcp
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-22
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from matplotlib import cm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor, nn

from src.cv.model import DSDBAModel
from src.utils.logger import log_info


def _parse_layer_path(path: str) -> list[str]:
  """Parse dotted/indexed layer path into traversal tokens."""
  tokens: list[str] = []
  i = 0
  while i < len(path):
    if path[i] == ".":
      i += 1
      continue
    if path[i] == "[":
      j = path.index("]", i)
      tokens.append(path[i + 1 : j])
      i = j + 1
      continue
    j = i
    while j < len(path) and path[j] not in ".[":
      j += 1
    tokens.append(path[i:j])
    i = j
  return tokens


def get_target_layer(model: DSDBAModel, cfg: dict[str, Any]) -> nn.Module:
  """
  Resolve target layer from config path.

  Args:
    model: DSDBA model wrapper.
    cfg: Full configuration mapping containing `gradcam.target_layer`.

  Returns:
    Target module object for Grad-CAM.

  Raises:
    AttributeError: If configured path cannot be resolved.
  """

  raw_path = str(cfg["gradcam"]["target_layer"])
  # Support multiple config path conventions:
  # - "model.features[8]" where "model" maps to EfficientNet backbone
  # - "model.backbone.features[8]" where "model.backbone" is the wrapper's backbone
  # Both should resolve to "backbone.features[8]" for traversal starting from DSDBAModel.
  if raw_path.startswith("model.backbone."):
    normalized = raw_path.replace("model.backbone.", "backbone.", 1)
  elif raw_path.startswith("model."):
    normalized = raw_path.replace("model.", "backbone.", 1)
  else:
    normalized = raw_path
  tokens = _parse_layer_path(normalized)

  obj: Any = model
  for tok in tokens:
    if tok.isdigit():
      obj = obj[int(tok)]
    else:
      obj = getattr(obj, tok)
  if not isinstance(obj, nn.Module):
    raise AttributeError(f"Resolved object is not nn.Module for path: {raw_path}")
  return obj


def compute_gradcam(model: DSDBAModel, tensor: Tensor, cfg: dict[str, Any]) -> np.ndarray:
  """
  Compute Grad-CAM saliency map.

  Args:
    model: Trained DSDBA model.
    tensor: Input tensor `[3,224,224]` or `[1,3,224,224]` float32.
    cfg: Config mapping.

  Returns:
    Saliency map shaped `[224, 224]` with values clipped to [0, 1].
  """

  input_tensor = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
  device = next(model.parameters()).device
  input_tensor = input_tensor.to(device=device, dtype=torch.float32)
  target_layer = get_target_layer(model, cfg)
  model.eval()

  with torch.no_grad():
    logits = model(input_tensor)
  cam_target_class = cfg["gradcam"].get("cam_target_class", None)
  if cam_target_class is None:
    target_idx = int(torch.argmax(logits, dim=1).item())
  else:
    target_idx = int(cam_target_class)
    if logits.ndim != 2 or target_idx < 0 or target_idx >= logits.shape[1]:
      raise ValueError(
        f"cam_target_class={target_idx} is out of range for model output shape {tuple(logits.shape)}"
      )
  targets = [ClassifierOutputTarget(target_idx)]

  with GradCAM(model=model, target_layers=[target_layer]) as cam:
    grayscale = cam(input_tensor=input_tensor, targets=targets)

  saliency = np.asarray(grayscale[0], dtype=np.float32)
  return np.clip(saliency, 0.0, 1.0)


def create_heatmap_overlay(tensor: Tensor, saliency: np.ndarray, cfg: dict[str, Any]) -> Path:
  """
  Create heatmap overlay PNG using configured colormap and alpha.

  Args:
    tensor: Input tensor `[3,224,224]` or `[1,3,224,224]`.
    saliency: Saliency map `[224,224]` normalized to [0,1].
    cfg: Config mapping with `gradcam.colormap` and `gradcam.overlay_alpha`.

  Returns:
    Path to generated PNG file.
  """

  grad_cfg = cfg["gradcam"]
  alpha = float(grad_cfg["overlay_alpha"])
  cmap_name = str(grad_cfg["colormap"])
  fmt = str(grad_cfg["output_format"]).lower()
  out_dir_value = grad_cfg.get("output_dir", None) or grad_cfg.get("heatmap_output_dir", None)
  out_dir = Path(str(out_dir_value)) if out_dir_value is not None else Path("artifacts/gradcam")
  if not out_dir.is_absolute():
    out_dir = Path(__file__).resolve().parents[2] / out_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  x = tensor[0] if tensor.ndim == 4 else tensor
  image = x.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
  image = np.clip(image, 0.0, 1.0)

  colormap = cm.get_cmap(cmap_name)
  heatmap = colormap(np.clip(saliency, 0.0, 1.0))[..., :3].astype(np.float32)
  overlay = np.clip((1.0 - alpha) * image + alpha * heatmap, 0.0, 1.0)
  overlay_u8 = (overlay * 255.0).astype(np.uint8)

  filename = f"gradcam_overlay_{int(time.time() * 1000)}.{fmt}"
  output_path = out_dir / filename
  from PIL import Image

  Image.fromarray(overlay_u8).save(output_path)
  return output_path


def get_mel_band_row_indices(cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
  """
  Map configured Hz band boundaries to mel-row index ranges.

  Args:
    cfg: Config mapping with `audio.n_mels` and `gradcam.band_hz`.

  Returns:
    Dict of half-open row ranges `{band: (start_idx, end_idx)}`.
  """

  n_mels = int(cfg["audio"]["n_mels"])
  band_hz: dict[str, list[float]] = cfg["gradcam"]["band_hz"]
  max_hz = float(max(v[1] for v in band_hz.values()))
  mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=max_hz)

  output: dict[str, tuple[int, int]] = {}
  for name in ["low", "low_mid", "high_mid", "high"]:
    low_hz, high_hz = band_hz[name]
    if name == "high":
      idx = np.where((mel_freqs >= low_hz) & (mel_freqs <= high_hz))[0]
    else:
      idx = np.where((mel_freqs >= low_hz) & (mel_freqs < high_hz))[0]
    if idx.size == 0:
      raise ValueError(f"No mel bins found for band '{name}' in range [{low_hz}, {high_hz}]")
    start = int(idx.min())
    end = int(idx.max()) + 1
    output[name] = (start, end)
  return output


def compute_band_attributions(saliency: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
  """
  Aggregate saliency by mel band and normalize to 100%.

  Args:
    saliency: Saliency map `[224,224]`.
    cfg: Config mapping.

  Returns:
    Band attribution percentages summing to 100.0 +/- 0.001.

  Notes:
    `gradcam.band_attribution_method` controls aggregation:

    - **saliency_mass** (default): ``sum(saliency in band) / sum(saliency over full map)``.
      Matches localized hotspots better than row-wise mean (which dilutes small red areas
      across large blue regions).

    - **mean_softmax**: mean saliency per band, then softmax or proportional scaling per
      ``band_normalisation``. Mean fixes bin-count bias for **sum**, but can look
      uniformly ~25% when the map is mostly flat noise with a tiny hotspot.
  """

  ranges = get_mel_band_row_indices(cfg)
  n_mels = int(cfg["audio"]["n_mels"])
  if saliency.shape[0] != n_mels:
    row_indices = np.linspace(0, saliency.shape[0] - 1, num=n_mels).astype(np.int64)
    mel_aligned = saliency[row_indices, :]
  else:
    mel_aligned = saliency

  band_keys = ["low", "low_mid", "high_mid", "high"]
  method = str(cfg["gradcam"].get("band_attribution_method", "saliency_mass")).lower()

  if method in ("saliency_mass", "mass", "mass_fraction"):
    sums = np.array(
      [float(np.sum(mel_aligned[s:e, :])) for (s, e) in [ranges[k] for k in band_keys]],
      dtype=np.float64,
    )
    total_mass = float(np.sum(sums))
    if total_mass <= 1e-18:
      perc = np.ones(4, dtype=np.float64) * 25.0
    else:
      perc = sums / total_mass * 100.0
  else:
    means = np.array(
      [float(np.mean(mel_aligned[s:e, :])) for (s, e) in [ranges[k] for k in band_keys]],
      dtype=np.float64,
    )
    means = np.clip(means, 1e-12, None)
    norm_mode = str(cfg["gradcam"].get("band_normalisation", "softmax")).lower()
    if norm_mode == "softmax":
      exp = np.exp(means - np.max(means))
      probs = exp / np.sum(exp)
      perc = probs * 100.0
    else:
      perc = means / float(np.sum(means)) * 100.0

  out = {band_keys[i]: float(perc[i]) for i in range(4)}
  total = float(sum(out.values()))
  assert abs(total - 100.0) <= 0.001, f"Band attribution sum invalid: {total}"
  return out


def run_gradcam(tensor: Tensor, model: DSDBAModel, cfg: dict[str, Any]) -> tuple[Path, dict[str, float]]:
  """
  Main API to generate heatmap PNG and band attribution percentages.

  Args:
    tensor: Input tensor `[3,224,224]` or `[1,3,224,224]`.
    model: DSDBA model.
    cfg: Config mapping.

  Returns:
    `(heatmap_png_path, band_attributions_dict)`.
  """

  start = time.perf_counter()
  saliency = compute_gradcam(model=model, tensor=tensor, cfg=cfg)
  heatmap_path = create_heatmap_overlay(tensor=tensor, saliency=saliency, cfg=cfg)
  band_attr = compute_band_attributions(saliency=saliency, cfg=cfg)
  latency_ms = (time.perf_counter() - start) * 1000.0

  log_info(
    stage="cv_gradcam",
    message="gradcam_complete",
    data={"latency_ms": round(latency_ms, 3), "heatmap_path": str(heatmap_path), "band_attr": band_attr},
  )
  assert latency_ms <= float(cfg["gradcam"]["latency_target_ms"]), (
    f"Grad-CAM latency exceeded: {latency_ms:.3f} ms > {cfg['gradcam']['latency_target_ms']} ms"
  )
  return heatmap_path, band_attr


def get_raw_saliency_json(saliency: np.ndarray) -> dict[str, Any]:
  """
  Return JSON-serialisable raw saliency payload.

  Args:
    saliency: Saliency map `[H,W]`.

  Returns:
    Dict payload with nested list matrix and shape metadata.
  """

  payload = {"shape": list(saliency.shape), "saliency": saliency.astype(np.float32).tolist()}
  # Validate serialisability for FR-CV-016.
  json.dumps(payload)
  return payload