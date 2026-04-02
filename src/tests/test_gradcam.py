from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest
import torch
import yaml

from src.cv.gradcam import (
  compute_band_attributions,
  compute_gradcam,
  create_heatmap_overlay,
  get_mel_band_row_indices,
  get_raw_saliency_json,
  get_target_layer,
  run_gradcam,
  save_saliency_json,
)
from src.cv.model import DSDBAModel


def _project_root() -> Path:
  return Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
  root = _project_root()
  data = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
  c = copy.deepcopy(data)
  c["gradcam"]["heatmap_output_dir"] = str(tmp_path)
  return c


@pytest.fixture
def model(cfg: dict) -> DSDBAModel:
  m = DSDBAModel(cfg=cfg, pretrained=False)
  m.eval()
  return m


@pytest.fixture
def tensor(cfg: dict) -> torch.Tensor:
  _c, h, w = (int(x) for x in cfg["audio"]["output_tensor_shape"])
  return torch.rand(_c, h, w, dtype=torch.float32)


def test_target_layer_exists(cfg: dict, model: DSDBAModel) -> None:
  layer = get_target_layer(model, cfg)
  assert isinstance(layer, torch.nn.Module)


def test_saliency_shape(cfg: dict, model: DSDBAModel, tensor: torch.Tensor) -> None:
  sal = compute_gradcam(model, tensor, cfg)
  h = int(cfg["audio"]["output_tensor_shape"][1])
  w = int(cfg["audio"]["output_tensor_shape"][2])
  assert sal.shape == (h, w)


def test_saliency_range(cfg: dict, model: DSDBAModel, tensor: torch.Tensor) -> None:
  sal = compute_gradcam(model, tensor, cfg)
  assert float(sal.min()) >= 0.0
  assert float(sal.max()) <= 1.0


def test_heatmap_png_created(cfg: dict, model: DSDBAModel, tensor: torch.Tensor) -> None:
  sal = compute_gradcam(model, tensor, cfg)
  path = create_heatmap_overlay(tensor, sal, cfg)
  assert path.is_file()
  assert path.suffix.lower() == ".png"


def test_mel_band_mapping_not_linear(cfg: dict) -> None:
  actual = get_mel_band_row_indices(cfg)
  h = int(cfg["audio"]["output_tensor_shape"][1])
  n_mels = int(cfg["audio"]["n_mels"])
  chunk = n_mels // 4
  naive: dict[str, tuple[int, int]] = {}
  names = ("low", "low_mid", "high_mid", "high")
  for i, name in enumerate(names):
    j0 = i * chunk
    j1 = (i + 1) * chunk - 1
    r0 = int(j0 * h / n_mels)
    r1 = int((j1 + 1) * h / n_mels) - 1
    r1 = min(max(r1, r0), h - 1)
    naive[name] = (r0, r1)
  assert actual != naive
  # Q5: not the naive quarter-bin row boundaries [0,55],[56,111],...
  even_split_edges = (0, h // 4, h // 2, (3 * h) // 4, h)
  even_bands = {
    names[0]: (even_split_edges[0], even_split_edges[1] - 1),
    names[1]: (even_split_edges[1], even_split_edges[2] - 1),
    names[2]: (even_split_edges[2], even_split_edges[3] - 1),
    names[3]: (even_split_edges[3], even_split_edges[4] - 1),
  }
  assert actual != even_bands


def test_band_sum_100(cfg: dict, model: DSDBAModel, tensor: torch.Tensor) -> None:
  sal = compute_gradcam(model, tensor, cfg)
  bands = compute_band_attributions(sal, cfg)
  assert abs(sum(bands.values()) - 100.0) <= 0.001


def test_gradcam_latency(cfg: dict, model: DSDBAModel, tensor: torch.Tensor) -> None:
  t0 = time.perf_counter()
  run_gradcam(tensor, model, cfg)
  elapsed_ms = (time.perf_counter() - t0) * 1000.0
  assert elapsed_ms <= float(cfg["gradcam"]["latency_target_ms"])


def test_raw_saliency_json_serialisable(cfg: dict, model: DSDBAModel, tensor: torch.Tensor) -> None:
  sal = compute_gradcam(model, tensor, cfg)
  payload = get_raw_saliency_json(sal)
  json.dumps(payload)


def test_saliency_json_file_created(cfg: dict, model: DSDBAModel, tensor: torch.Tensor) -> None:
  sal = compute_gradcam(model, tensor, cfg)
  path = save_saliency_json(sal, cfg)
  assert path.is_file()
  assert path.suffix.lower() == ".json"
  json.loads(path.read_text(encoding="utf-8"))
