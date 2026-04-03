from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from src.nlp.explain import (
  NLPTimeoutError,
  build_prompt,
  build_rule_based_explanation,
  generate_explanation,
  get_cached_explanation,
)


@pytest.fixture()
def cfg() -> dict:
  # Minimal cfg subset for NLP functions.
  return {
    "nlp": {
      "timeout_sec": 0.01,
      "explanation_min_sentences": 3,
      "explanation_max_sentences": 5,
      "api_key_env_var": "QWEN_API_KEY",
      "hf_token_env_var": "HF_TOKEN",
      "qwen_model": "Qwen/Qwen2.5-7B-Instruct",
      "gemma_model": "google/gemma-2-9b-it",
      "caching": {
        "enabled": True,
        "cache_key_fields": ["label", "confidence_bucket", "top_band"],
        "confidence_buckets": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
      },
    }
  }


@pytest.fixture()
def band_pct() -> dict[str, float]:
  return {"low": 10.0, "low_mid": 20.0, "high_mid": 30.0, "high": 40.0}


def test_build_prompt_contains_all_fields(cfg: dict, band_pct: dict[str, float]) -> None:
  prompt = build_prompt(label="spoof", confidence=0.9123, band_pct=band_pct, cfg=cfg)
  assert "Prediction label: spoof" in prompt
  assert "Confidence:" in prompt
  for k in ("low", "low_mid", "high_mid", "high"):
    assert k in prompt


def test_rule_based_fallback_always_returns(band_pct: dict[str, float]) -> None:
  out = build_rule_based_explanation(label="bonafide", confidence=0.55, band_pct=band_pct)
  assert isinstance(out, str)
  assert out.strip() != ""


def test_rule_based_grammar(band_pct: dict[str, float]) -> None:
  out = build_rule_based_explanation(label="spoof", confidence=0.9, band_pct=band_pct)
  # crude sentence count: split on ".", keep non-empty.
  sentences = [s.strip() for s in out.split(".") if s.strip()]
  assert len(sentences) >= 3


@pytest.mark.asyncio
async def test_qwen_timeout_triggers_fallback(monkeypatch: pytest.MonkeyPatch, cfg: dict, band_pct: dict[str, float]) -> None:
  async def fake_qwen(*_args, **_kwargs) -> str:
    raise NLPTimeoutError(provider="qwen_2_5", timeout_sec=float(cfg["nlp"]["timeout_sec"]))

  async def fake_gemma(*_args, **_kwargs) -> str:
    raise RuntimeError("fallback also fails")

  from src.nlp import explain as mod

  monkeypatch.setattr(mod, "call_qwen_api", fake_qwen)
  monkeypatch.setattr(mod, "call_gemma_fallback", fake_gemma)

  text, api_used = await generate_explanation("spoof", 0.8, band_pct, cfg)
  assert isinstance(text, str) and text.strip()
  assert api_used is False


@pytest.mark.asyncio
async def test_warning_flag_on_fallback(monkeypatch: pytest.MonkeyPatch, cfg: dict, band_pct: dict[str, float]) -> None:
  async def fail_qwen(*_args, **_kwargs) -> str:
    raise RuntimeError("boom")

  from src.nlp import explain as mod

  monkeypatch.setattr(mod, "call_qwen_api", fail_qwen)
  monkeypatch.setattr(mod, "call_gemma_fallback", fail_qwen)

  _, api_used = await generate_explanation("bonafide", 0.51, band_pct, cfg)
  assert api_used is False


@pytest.mark.asyncio
async def test_cv_result_independent_of_nlp(monkeypatch: pytest.MonkeyPatch, cfg: dict, band_pct: dict[str, float]) -> None:
  async def slow_qwen(*_args, **_kwargs) -> str:
    await asyncio.sleep(0.2)
    return "ok"

  from src.nlp import explain as mod

  monkeypatch.setattr(mod, "call_qwen_api", slow_qwen)

  # Mock CV result available immediately.
  cv_result = {"label": "spoof", "confidence": 0.9}
  task = asyncio.create_task(generate_explanation(cv_result["label"], cv_result["confidence"], band_pct, cfg))

  # Ensure UI/CV can display without awaiting NLP.
  assert cv_result["label"] == "spoof"
  assert isinstance(task, asyncio.Task)

  text, api_used = await task
  assert isinstance(text, str)
  assert api_used is True


def test_no_api_key_in_source() -> None:
  path = Path(__file__).resolve().parents[1] / "nlp" / "explain.py"
  src = path.read_text(encoding="utf-8")

  patterns = [
    r"hf_[A-Za-z0-9]+",
    r"sk-[A-Za-z0-9]+",
    r"Bearer\\s+",
  ]
  for pat in patterns:
    assert re.search(pat, src) is None, f"Found forbidden token pattern: {pat}"


@pytest.mark.asyncio
async def test_cache_hit_skips_api(monkeypatch: pytest.MonkeyPatch, cfg: dict, band_pct: dict[str, float]) -> None:
  calls = {"n": 0}

  async def counting_qwen(*_args, **_kwargs) -> str:
    calls["n"] += 1
    return "cached text"

  from src.nlp import explain as mod

  monkeypatch.setattr(mod, "call_qwen_api", counting_qwen)

  text1, api1 = await generate_explanation("spoof", 0.91, band_pct, cfg)
  assert text1 == "cached text"
  assert api1 is True
  assert calls["n"] == 1

  # Second call should hit cache (same label + nearest bucket + top band)
  cached = get_cached_explanation("spoof", 0.91, band_pct, cfg)
  assert cached == "cached text"

  text2, api2 = await generate_explanation("spoof", 0.91, band_pct, cfg)
  assert text2 == "cached text"
  assert api2 is True
  assert calls["n"] == 1

