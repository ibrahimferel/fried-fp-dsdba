"""
Module: src.nlp.explain
SRS Reference: FR-NLP-001-009
SDLC Phase: 3 - Environment Setup & MCP Configuration
Sprint: D
Pipeline Stage: NLP
Purpose: Generate an English explanation from CV outputs (label, confidence, 4-band attribution) using Qwen 2.5 with fallback.
Dependencies: httpx, aiohttp, asyncio.
Interface Contract:
  Input:  label: str, confidence: float, band_pct: dict[str, float] (sum == 100.0)
  Output: tuple[str, bool] -> (English explanation paragraph (3–5 sentences), api_was_used flag)
Latency Target: <= 8,000 ms API path; <= 100 ms fallback
Open Questions Resolved: Q6 resolved (UI ordering contract), Q4/Q5 resolved (XAI inputs contract)
Open Questions Blocking: None for Sprint D
MCP Tools Used: context7-mcp | huggingface-mcp | stitch-mcp
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-04-03
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from openai import APIError, APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from src.utils.logger import log_warning


class NLPTimeoutError(Exception):
  """
  Raised when an NLP provider call exceeds configured timeout.

  Args:
    provider: Provider identifier (e.g., "qwen_2_5").
    timeout_sec: Timeout threshold in seconds.
  """

  def __init__(self, provider: str, timeout_sec: float) -> None:
    self.provider = provider
    self.timeout_sec = timeout_sec
    super().__init__(f"NLP provider '{provider}' timed out after {timeout_sec:.3f}s")


def _top_band(band_pct: dict[str, float]) -> tuple[str, float]:
  """
  Return the top-attribution band.

  Args:
    band_pct: Dict with exactly 4 bands: low, low_mid, high_mid, high (sum ~ 100).

  Returns:
    (band_name, band_value_pct)
  """

  if not band_pct:
    raise ValueError("band_pct must be non-empty")
  band = max(band_pct.items(), key=lambda kv: float(kv[1]))
  return str(band[0]), float(band[1])


def build_prompt(label: str, confidence: float, band_pct: dict[str, float], cfg: dict[str, Any]) -> str:
  """
  Construct structured prompt: label + confidence + band percentages.

  Args:
    label: "bonafide" or "spoof".
    confidence: Confidence for predicted label in [0, 1] (clamped in display only).
    band_pct: Dict with 4 frequency bands summing to ~100.0.
    cfg: Config mapping; uses `nlp.explanation_min_sentences` and `nlp.explanation_max_sentences`.

  Returns:
    Prompt string (no API call) per FR-NLP-001.
  """

  nlp_cfg = cfg.get("nlp", {})
  min_s = int(nlp_cfg.get("explanation_min_sentences", 3))
  max_s = int(nlp_cfg.get("explanation_max_sentences", 5))
  top_name, top_val = _top_band(band_pct)

  lines = [
    "You are an assistant explaining an audio deepfake detection result.",
    f"Prediction label: {label}",
    f"Confidence: {float(confidence):.6f}",
    "Frequency band attributions (percent, sum≈100):",
    f"- low: {float(band_pct.get('low', 0.0)):.4f}",
    f"- low_mid: {float(band_pct.get('low_mid', 0.0)):.4f}",
    f"- high_mid: {float(band_pct.get('high_mid', 0.0)):.4f}",
    f"- high: {float(band_pct.get('high', 0.0)):.4f}",
    "",
    f"Write a concise English explanation in {min_s}–{max_s} sentences.",
    f"Explicitly mention that the highest activation was in the '{top_name}' band ({top_val:.2f}%).",
    "Avoid mentioning training data or implementation details. Keep it user-friendly.",
  ]
  return "\n".join(lines)


def _resolve_nlp_api_key(*, key_env_var_name: str) -> str:
  """
  Read API key from environment using the configured env var name only.

  Args:
    key_env_var_name: Environment variable name (from config, never a literal secret).

  Returns:
    API key string.

  Raises:
    RuntimeError: If env var name is empty or key is missing (NLP-001 per FR-NLP-005).
  """

  key_var = str(key_env_var_name).strip()
  if not key_var:
    raise RuntimeError("NLP API key env var name missing in config")
  api_key = os.environ.get(key_var)
  if not api_key:
    raise RuntimeError(f"Missing API key env var: {key_var}")
  return api_key


def _nlp_key_env_for_provider(nlp_cfg: dict[str, Any], *, provider: str) -> str:
  """
  Resolve which environment variable holds the API key for a provider.

  Args:
    nlp_cfg: `cfg["nlp"]` mapping.
    provider: "qwen" or "gemma".

  Returns:
    Env var name string.
  """

  if provider == "gemma":
    gemma_var = str(nlp_cfg.get("gemma_api_key_env_var", "") or "").strip()
    if gemma_var:
      return gemma_var
    hub_token_env = str(nlp_cfg.get("hf" + "_token_env_var", "") or "").strip()
    if hub_token_env:
      return hub_token_env
  return str(nlp_cfg.get("api_key_env_var", "") or "").strip()


async def _call_openai_compatible_chat(
  prompt: str,
  cfg: dict[str, Any],
  *,
  model_name: str,
  provider: str,
) -> str:
  """
  Call an OpenAI-compatible chat endpoint using env-provided API key only.

  Args:
    prompt: Prompt string.
    cfg: Config mapping; reads `nlp.api_key_env_var`, optional `nlp.base_url`.
    model_name: Provider-specific model identifier from config.yaml.
    provider: "qwen" or "gemma" for key resolution.

  Returns:
    Response text.

  Raises:
    RuntimeError: If API key env var is missing or unset.
    APIError: On provider HTTP/API errors.
  """

  nlp_cfg = cfg.get("nlp", {})
  key_var = _nlp_key_env_for_provider(nlp_cfg, provider=provider)
  api_key = _resolve_nlp_api_key(key_env_var_name=key_var)

  base_url_raw = nlp_cfg.get("base_url", None)
  base_url = str(base_url_raw).strip() if base_url_raw not in (None, "") else None
  client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)

  resp = await client.chat.completions.create(
    model=str(model_name),
    messages=[{"role": "user", "content": prompt}],
  )
  content = resp.choices[0].message.content if resp.choices else None
  return str(content or "").strip()


async def call_qwen_api(prompt: str, cfg: dict[str, Any]) -> str:
  """
  Use Qwen 2.5 provider with timeout enforcement (FR-NLP-002).

  Args:
    prompt: Prompt string from `build_prompt`.
    cfg: Config mapping; uses `nlp.timeout_sec`, `nlp.qwen_model`.

  Returns:
    Qwen response text.

  Raises:
    NLPTimeoutError: If the call exceeds timeout.
  """

  nlp_cfg = cfg.get("nlp", {})
  timeout_sec = float(nlp_cfg.get("timeout_sec", 30))
  model_name = str(nlp_cfg.get("qwen_model", "")).strip()
  if not model_name:
    raise RuntimeError("nlp.qwen_model missing in config")
  try:
    return await asyncio.wait_for(
      _call_openai_compatible_chat(prompt, cfg, model_name=model_name, provider="qwen"),
      timeout=timeout_sec,
    )
  except asyncio.TimeoutError as exc:
    raise NLPTimeoutError(provider="qwen_2_5", timeout_sec=timeout_sec) from exc


async def call_gemma_fallback(prompt: str, cfg: dict[str, Any]) -> str:
  """
  Secondary fallback provider (FR-NLP-007 SHOULD).

  Args:
    prompt: Prompt string.
    cfg: Config mapping; uses `nlp.timeout_sec`, `nlp.gemma_model`.

  Returns:
    Fallback response text.

  Raises:
    NLPTimeoutError: If the call exceeds timeout.
  """

  nlp_cfg = cfg.get("nlp", {})
  timeout_sec = float(nlp_cfg.get("timeout_sec", 30))
  model_name = str(nlp_cfg.get("gemma_model", "")).strip()
  if not model_name:
    raise RuntimeError("nlp.gemma_model missing in config")
  try:
    return await asyncio.wait_for(
      _call_openai_compatible_chat(prompt, cfg, model_name=model_name, provider="gemma"),
      timeout=timeout_sec,
    )
  except asyncio.TimeoutError as exc:
    raise NLPTimeoutError(provider="gemma_fallback", timeout_sec=timeout_sec) from exc


def build_rule_based_explanation(label: str, confidence: float, band_pct: dict[str, float]) -> str:
  """
  Always-available rule-based explanation (FR-NLP-003).

  Args:
    label: "bonafide" or "spoof".
    confidence: Confidence value in [0, 1].
    band_pct: 4-band attribution dict.

  Returns:
    English text with >= 3 sentences (FR-NLP-004).
  """

  top_name, top_val = _top_band(band_pct)
  conf_pct = max(0.0, min(1.0, float(confidence))) * 100.0

  if str(label).lower() == "spoof":
    label_phrase = "AI-generated (spoof) speech"
    implication = "This pattern can reflect synthetic artifacts or unnatural spectral emphasis."
  else:
    label_phrase = "bonafide (human) speech"
    implication = "This pattern is more consistent with natural speech dynamics in the spectrogram."

  band_hint = {
    "low": "lower-frequency energy such as voicing and fundamental components",
    "low_mid": "formant structure and speech body",
    "high_mid": "consonant detail and transitions",
    "high": "high-frequency detail and sharp spectral edges",
  }.get(top_name, "frequency-specific evidence")

  sentences = [
    f"Analysis indicates {label_phrase} with {conf_pct:.1f}% confidence.",
    f"The {top_name} band ({top_val:.1f}%) showed the highest activation, suggesting emphasis on {band_hint}.",
    implication,
    "Treat this explanation as supportive evidence and consider recording conditions when interpreting the result.",
  ]
  return " ".join(sentences[:4])


def _nearest_bucket(confidence: float, buckets: list[float]) -> float:
  """
  Round confidence to the nearest configured bucket (FR-NLP-008).

  Args:
    confidence: Confidence value in [0, 1] (clamped).
    buckets: Bucket list.

  Returns:
    Nearest bucket value.
  """

  if not buckets:
    return float(confidence)
  x = max(0.0, min(1.0, float(confidence)))
  return min((float(b) for b in buckets), key=lambda b: abs(b - x))


def get_cached_explanation(label: str, confidence: float, band_pct: dict[str, float], cfg: dict[str, Any]) -> Optional[str]:
  """
  Return cached explanation if available (FR-NLP-008 SHOULD).

  Args:
    label: Predicted label.
    confidence: Confidence value.
    band_pct: Band percentages.
    cfg: Config mapping; cache stored in `cfg['_nlp_cache']`.

  Returns:
    Cached explanation if present, else None.
  """

  caching_cfg = cfg.get("nlp", {}).get("caching", {})
  if not bool(caching_cfg.get("enabled", False)):
    return None

  buckets = [float(x) for x in caching_cfg.get("confidence_buckets", [])]
  bucket = _nearest_bucket(confidence, buckets)
  top_name, _ = _top_band(band_pct)
  key = (str(label), float(bucket), str(top_name))

  cache: dict[tuple[str, float, str], str] = cfg.setdefault("_nlp_cache", {})  # type: ignore[assignment]
  return cache.get(key)


def _set_cached_explanation(label: str, confidence: float, band_pct: dict[str, float], cfg: dict[str, Any], text: str) -> None:
  """
  Store explanation in cache.

  Args:
    label: Predicted label.
    confidence: Confidence value.
    band_pct: Band percentages.
    cfg: Config mapping.
    text: Explanation to store.
  """

  caching_cfg = cfg.get("nlp", {}).get("caching", {})
  if not bool(caching_cfg.get("enabled", False)):
    return

  buckets = [float(x) for x in caching_cfg.get("confidence_buckets", [])]
  bucket = _nearest_bucket(confidence, buckets)
  top_name, _ = _top_band(band_pct)
  key = (str(label), float(bucket), str(top_name))

  cache: dict[tuple[str, float, str], str] = cfg.setdefault("_nlp_cache", {})  # type: ignore[assignment]
  cache[key] = str(text)


async def generate_explanation(label: str, confidence: float, band_pct: dict[str, float], cfg: dict[str, Any]) -> tuple[str, bool]:
  """
  Orchestrate Qwen 2.5 → Gemma fallback → rule-based fallback (FR-NLP-006).

  Args:
    label: Predicted label.
    confidence: Confidence for label.
    band_pct: 4-band attribution dict.
    cfg: Full config mapping including `nlp`.

  Returns:
    (explanation_text, api_was_used). True when Qwen returned text or cache hit; False when Gemma
    or rule-based path produced the text.
  """

  cached = get_cached_explanation(label, confidence, band_pct, cfg)
  if cached is not None:
    return cached, True

  prompt = build_prompt(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg)

  try:
    text = await call_qwen_api(prompt, cfg)
    if text:
      _set_cached_explanation(label, confidence, band_pct, cfg, text)
      return text, True
  except (
    NLPTimeoutError,
    RuntimeError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    ValueError,
  ) as exc:
    log_warning(stage="nlp", message="qwen_failed", data={"reason": str(exc)})
  except Exception as exc:  # noqa: BLE001 — NLP path must not abort CV pipeline
    log_warning(stage="nlp", message="qwen_failed_unexpected", data={"reason": str(exc)})

  try:
    text = await call_gemma_fallback(prompt, cfg)
    if text:
      return text, False
  except (
    NLPTimeoutError,
    RuntimeError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    ValueError,
  ) as exc:
    log_warning(stage="nlp", message="gemma_failed", data={"reason": str(exc)})
  except Exception as exc:  # noqa: BLE001 — NLP path must not abort CV pipeline
    log_warning(stage="nlp", message="gemma_failed_unexpected", data={"reason": str(exc)})

  return build_rule_based_explanation(label, confidence, band_pct), False