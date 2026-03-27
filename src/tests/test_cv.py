from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from src.cv.infer import (
	export_to_onnx,
	load_onnx_session,
	run_onnx_inference,
	timed_onnx_inference,
	verify_onnx_equivalence,
)
from src.cv.model import DSDBAModel
from src.cv.train import get_class_weights


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
def onnx_bundle(cfg: dict, model: DSDBAModel):
	cfg_local = copy.deepcopy(cfg)
	onnx_path = export_to_onnx(model, cfg_local)
	session = load_onnx_session(onnx_path, cfg_local)
	return cfg_local, onnx_path, session


def test_model_output_shape(cfg: dict, model: DSDBAModel) -> None:
	shape = tuple(int(v) for v in cfg["audio"]["output_tensor_shape"])
	x = torch.randn(1, *shape)
	y = model(x)
	assert tuple(y.shape) == (1, 2)


def test_sigmoid_output_range(onnx_bundle) -> None:
	cfg, _, session = onnx_bundle
	shape = tuple(int(v) for v in cfg["audio"]["output_tensor_shape"])
	x = torch.randn(1, *shape)
	_, confidence = run_onnx_inference(session, x, cfg)
	assert 0.0 < confidence < 1.0


def test_freeze_unfreeze(cfg: dict) -> None:
	m = DSDBAModel(cfg=cfg, pretrained=False)
	m.freeze_backbone()

	frozen_ok = all(not p.requires_grad for p in m.backbone.features.parameters())
	head_ok = all(p.requires_grad for p in m.backbone.classifier.parameters())
	assert frozen_ok
	assert head_ok

	m.unfreeze_top_n(2)
	top_blocks = list(m.backbone.features.children())[-2:]
	unfrozen_ok = any(p.requires_grad for b in top_blocks for p in b.parameters())
	assert unfrozen_ok


def test_onnx_export_creates_file(onnx_bundle) -> None:
	_, onnx_path, _ = onnx_bundle
	assert onnx_path.exists()
	assert onnx_path.suffix == ".onnx"


def test_onnx_equivalence(cfg: dict, model: DSDBAModel, onnx_bundle) -> None:
	cfg_local, onnx_path, _ = onnx_bundle
	cfg_local["deployment"]["onnx_equivalence_tolerance"] = 1.0e-5
	assert verify_onnx_equivalence(model, onnx_path, cfg_local)


def test_onnx_latency(onnx_bundle) -> None:
	cfg, _, session = onnx_bundle
	shape = tuple(int(v) for v in cfg["audio"]["output_tensor_shape"])
	x = torch.randn(1, *shape)
	_, latency_ms = timed_onnx_inference(session, x, cfg)
	assert latency_ms <= float(cfg["deployment"]["onnx_latency_target_ms"])


def test_onnx_cpu_provider_only(onnx_bundle) -> None:
	_, _, session = onnx_bundle
	providers = session.get_providers()
	assert providers == ["CPUExecutionProvider"]


def test_class_weights_sum() -> None:
	class DummyDataset:
		labels = [0, 0, 0, 1, 1]

	weights = get_class_weights(DummyDataset())
	assert isinstance(weights, torch.Tensor)
	assert tuple(weights.shape) == (2,)
	assert torch.all(weights > 0)
	assert torch.isfinite(weights).all()
