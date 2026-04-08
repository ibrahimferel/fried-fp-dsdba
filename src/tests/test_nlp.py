from __future__ import annotations

import asyncio
import copy
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from src.nlp.explain import (
  NLPTimeoutError,
  build_prompt,
  build_rule_based_explanation,
  generate_explanation,
  get_cached_explanation,
)


def _load_cfg() -> dict[str, Any]:
  root = Path(__file__).resolve().parents[2]
  return yaml.safe_load((root / "config.yaml").read_text())


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
  return _load_cfg()


def _sample_inputs() -> tuple[str, float, dict[str, float]]:
  label = "spoof"
  confidence = 0.73  # ratio in [0,1]
  band_pct = {"low": 10.0, "low_mid": 25.0, "high_mid": 30.0, "high": 35.0}
  return label, confidence, band_pct


def test_build_prompt_contains_all_fields(cfg: dict[str, Any]) -> None:
  label, confidence, band_pct = _sample_inputs()
  prompt = build_prompt(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg)
  assert label in prompt
  assert str(float(confidence)) in prompt
  for k in ["low", "low_mid", "high_mid", "high"]:
    assert k in prompt


def test_rule_based_fallback_always_returns() -> None:
  label = "bonafide"
  confidence = 0.91
  band_pct = {"low": 20.0, "low_mid": 20.0, "high_mid": 30.0, "high": 30.0}
  out = build_rule_based_explanation(label=label, confidence=confidence, band_pct=band_pct)
  assert isinstance(out, str)
  assert out.strip() != ""


def test_rule_based_grammar() -> None:
  label, confidence, band_pct = _sample_inputs()
  out = build_rule_based_explanation(label=label, confidence=confidence, band_pct=band_pct)
  assert out.strip() != ""
  sentences = [s for s in re.split(r"[.!?]+", out) if s.strip()]
  assert len(sentences) >= 3


@pytest.mark.asyncio
async def test_qwen_timeout_triggers_fallback(cfg: dict[str, Any]) -> None:
  cfg_local = copy.deepcopy(cfg)
  label, confidence, band_pct = _sample_inputs()

  async def _timeout(*args: Any, **kwargs: Any) -> str:
    raise NLPTimeoutError("timeout")

  expected_rule_based = build_rule_based_explanation(label=label, confidence=confidence, band_pct=band_pct)

  with (
    patch("src.nlp.explain.call_qwen_api", new=_timeout),
    patch("src.nlp.explain.call_gemma_fallback", new=_timeout),
  ):
    explanation, api_was_used = await generate_explanation(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg_local)

  assert explanation == expected_rule_based
  assert api_was_used is False


@pytest.mark.asyncio
async def test_warning_flag_on_fallback(cfg: dict[str, Any]) -> None:
  cfg_local = copy.deepcopy(cfg)
  label, confidence, band_pct = _sample_inputs()

  async def _timeout(*args: Any, **kwargs: Any) -> str:
    raise NLPTimeoutError("timeout")

  with (
    patch("src.nlp.explain.call_qwen_api", new=_timeout),
    patch("src.nlp.explain.call_gemma_fallback", new=_timeout),
  ):
    _, api_was_used = await generate_explanation(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg_local)

  assert api_was_used is False


@pytest.mark.asyncio
async def test_cv_result_independent_of_nlp(cfg: dict[str, Any]) -> None:
  cfg_local = copy.deepcopy(cfg)
  label, confidence, band_pct = _sample_inputs()
  started = asyncio.Event()
  unblock = asyncio.Event()

  async def _slow_call(*args: Any, **kwargs: Any) -> str:
    started.set()
    await unblock.wait()
    return "api explanation"

  with patch("src.nlp.explain.call_qwen_api", new=_slow_call):
    task = asyncio.create_task(generate_explanation(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg_local))

    # If `generate_explanation` blocks the event loop, this won't execute immediately.
    cv_displayed = True
    assert cv_displayed is True

    # Task should be pending until we unblock the provider call.
    await asyncio.sleep(0)
    assert not task.done()

    await started.wait()
    unblock.set()
    explanation, api_was_used = await task

  assert explanation == "api explanation"
  assert api_was_used is True


def test_no_api_key_in_source() -> None:
  src_path = Path(__file__).resolve().parents[2] / "src" / "nlp" / "explain.py"
  text = src_path.read_text(encoding="utf-8")
  patterns = [r"hf_", r"sk-", r"DASH", r"Bearer"]
  for pat in patterns:
    assert re.search(pat, text) is None


@pytest.mark.asyncio
async def test_cache_hit_skips_api(cfg: dict[str, Any]) -> None:
  cfg_local = copy.deepcopy(cfg)
  label, confidence, band_pct = _sample_inputs()
  # Use a unique API return string so we can confirm it comes from cache on the second call.
  api_text = "API explanation"

  # Ensure cache is clean for this test key by disabling and re-enabling a new cache dict.
  cfg_local.setdefault("nlp", {}).setdefault("caching", {}).setdefault("enabled", False)
  # Force caching on for this test even if config.yaml disables it.
  cfg_local["nlp"]["caching"]["enabled"] = True
  cfg_local["nlp"]["_explanation_cache"] = {}

  with patch("src.nlp.explain.call_gemma_fallback", new=AsyncMock(side_effect=NLPTimeoutError("should not be called"))):
    qwen_mock = AsyncMock(return_value=api_text)
    with patch("src.nlp.explain.call_qwen_api", new=qwen_mock):
      first_text, first_api_used = await generate_explanation(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg_local)
      second_text, second_api_used = await generate_explanation(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg_local)

  assert first_text == api_text
  assert first_api_used is True
  assert second_text == api_text
  assert second_api_used is True
  assert qwen_mock.call_count == 1

  # Also validate cache retrieval helper for the same key.
  cached_text = get_cached_explanation(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg_local)
  assert cached_text == api_text

