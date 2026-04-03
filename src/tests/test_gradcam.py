from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from src.cv.gradcam import (
  compute_band_attributions,
  compute_gradcam,
  create_heatmap_overlay,
  get_mel_band_row_indices,
  get_raw_saliency_json,
  get_target_layer,
  run_gradcam,
)
from src.cv.model import DSDBAModel


def _load_cfg() -> dict:
  root = Path(__file__).resolve().parents[2]
  return yaml.safe_load((root / "config.yaml").read_text())


@pytest.fixture(scope="module")
def cfg() -> dict:
  return _load_cfg()


@pytest.fixture(scope="module")
def model(cfg: dict) -> DSDBAModel:
  m = DSDBAModel(cfg=cfg, pretrained=False)
  m.eval()
  return m


@pytest.fixture(scope="module")
def tensor(cfg: dict) -> torch.Tensor:
  shape = tuple(int(v) for v in cfg["audio"]["output_tensor_shape"])
  return torch.rand(*shape, dtype=torch.float32)


def test_target_layer_exists(model: DSDBAModel, cfg: dict) -> None:
  layer = get_target_layer(model, cfg)
  assert isinstance(layer, nn.Module)


def test_saliency_shape(model: DSDBAModel, tensor: torch.Tensor, cfg: dict) -> None:
  saliency = compute_gradcam(model, tensor, cfg)
  assert saliency.shape == (224, 224)


def test_saliency_range(model: DSDBAModel, tensor: torch.Tensor, cfg: dict) -> None:
  saliency = compute_gradcam(model, tensor, cfg)
  assert float(np.min(saliency)) >= 0.0
  assert float(np.max(saliency)) <= 1.0


def test_heatmap_png_created(model: DSDBAModel, tensor: torch.Tensor, cfg: dict, tmp_path: Path) -> None:
  cfg_local = dict(cfg)
  cfg_local["gradcam"] = dict(cfg["gradcam"])
  cfg_local["gradcam"]["output_dir"] = str(tmp_path)
  saliency = compute_gradcam(model, tensor, cfg_local)
  out_path = create_heatmap_overlay(tensor, saliency, cfg_local)
  assert out_path.exists()
  assert out_path.suffix.lower() == ".png"


def test_mel_band_mapping_not_linear(cfg: dict) -> None:
  ranges = get_mel_band_row_indices(cfg)
  lengths = [ranges["low"][1] - ranges["low"][0], ranges["low_mid"][1] - ranges["low_mid"][0], ranges["high_mid"][1] - ranges["high_mid"][0], ranges["high"][1] - ranges["high"][0]]
  # Q5 check: mel spacing should not produce equal linear partitions.
  assert not all(v == lengths[0] for v in lengths)
  assert ranges["low"][1] != 32
  assert ranges["low_mid"][1] != 64
  assert ranges["high_mid"][1] != 96


def test_band_sum_100(cfg: dict) -> None:
  saliency = np.random.rand(224, 224).astype(np.float32)
  attributions = compute_band_attributions(saliency, cfg)
  total = float(sum(attributions.values()))
  assert abs(total - 100.0) <= 0.001


def test_uniform_saliency_mean_softmax_balanced(cfg: dict) -> None:
  """With mean_softmax, uniform saliency gives ~25% per band."""
  cfg_local = dict(cfg)
  cfg_local["gradcam"] = dict(cfg["gradcam"])
  cfg_local["gradcam"]["band_attribution_method"] = "mean_softmax"
  saliency = np.ones((224, 224), dtype=np.float32)
  attributions = compute_band_attributions(saliency, cfg_local)
  for v in attributions.values():
    assert abs(v - 25.0) < 0.05


def test_saliency_mass_sums_to_100(cfg: dict) -> None:
  cfg_local = dict(cfg)
  cfg_local["gradcam"] = dict(cfg["gradcam"])
  cfg_local["gradcam"]["band_attribution_method"] = "saliency_mass"
  saliency = np.random.rand(224, 224).astype(np.float32)
  attributions = compute_band_attributions(saliency, cfg_local)
  assert abs(sum(attributions.values()) - 100.0) <= 0.001


def test_gradcam_latency(model: DSDBAModel, tensor: torch.Tensor, cfg: dict, tmp_path: Path) -> None:
  cfg_local = dict(cfg)
  cfg_local["gradcam"] = dict(cfg["gradcam"])
  cfg_local["gradcam"]["output_dir"] = str(tmp_path)
  start = time.perf_counter()
  _ = run_gradcam(tensor=tensor, model=model, cfg=cfg_local)
  elapsed_ms = (time.perf_counter() - start) * 1000.0
  assert elapsed_ms <= float(cfg_local["gradcam"]["latency_target_ms"])


def test_raw_saliency_json_serialisable() -> None:
  saliency = np.random.rand(224, 224).astype(np.float32)
  payload = get_raw_saliency_json(saliency)
  dumped = json.dumps(payload)
  assert isinstance(dumped, str)
