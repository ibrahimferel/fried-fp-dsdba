"""
Module: src.nlp.explain
SRS Reference: FR-NLP-001–009
SDLC Phase: 4 - Implementation (Sprint D)
Sprint: D
Pipeline Stage: NLP
Purpose: Generate an English explanation from CV outputs (label, confidence, 4-band attribution) using Qwen 2.5 with fallback.
Dependencies: asyncio, os, openai (async)
Interface Contract:
  Input:  label: str, confidence: float, band_pct: dict[str, float] (band_pct includes 4 keys; sum ~= 100.0)
  Output: tuple[str, bool] from `generate_explanation` => (explanation_text, api_was_used)
  Note: `build_prompt` returns str prompt only; `build_rule_based_explanation` returns str explanation only.
Latency Target: <= 8,000 ms API path (per `cfg['nlp']['timeout_sec']`), <= 100 ms fallback on CPU (rule-based)
Open Questions Resolved: Q6 (UI ordering) and Q4/Q5 input contract alignment (Sprint D scope)
Open Questions Blocking: None for Sprint D
MCP Tools Used: stitch-mcp (conceptual API orchestration), context7-mcp, huggingface-mcp
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-04-02
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from src.utils.logger import log_info, log_warning


class NLPTimeoutError(TimeoutError):
  """Raised when an NLP provider call times out or fails.

  Args:
    message: Human-readable description of why the provider failed.

  Returns:
    None.

  Raises:
    None (this is an exception type).
  """


@dataclass(frozen=True)
class _CacheRecord:
  """Internal cache record used by the NLP explanation module.

  Args:
    text: Cached explanation text.
    api_was_used: Whether an external API produced the cached text.
  """

  text: str
  api_was_used: bool


def _confidence_to_ratio(confidence: float) -> float:
  """
  Convert confidence into a ratio in [0, 1] for caching bucketing.

  Args:
    confidence: Either a ratio (0..1) or a percent-like value (>1). Must be finite.
      - Per FR-CV-004, model inference confidence is a probability in (0, 1).

  Returns:
    float ratio in [0, 1] (clamped).

  Raises:
    ValueError: If confidence is NaN or infinite.
  """
  x = float(confidence)
  if not (x == x) or x in (float("inf"), float("-inf")):
    raise ValueError("confidence must be finite")
  # If confidence looks like "92.0", treat as percent. Otherwise treat as probability.
  ratio = x / 100.0 if x > 1.0 else x
  return max(0.0, min(1.0, ratio))


def _ratio_to_percent_text(confidence: float) -> float:
  """
  Convert confidence into a percentage value for human-readable text.

  Args:
    confidence: Ratio (0..1) or percent-like value (>1).

  Returns:
    float in [0, 100] representing the displayed percentage.

  Raises:
    ValueError: If confidence is NaN or infinite.
  """
  ratio = _confidence_to_ratio(confidence)
  return ratio * 100.0


def _get_top_band_name(band_pct: dict[str, float]) -> str:
  """
  Get the band with the highest attribution.

  Args:
    band_pct: Frequency-band attribution dict with float percentages.

  Returns:
    Top band name as str.

  Raises:
    ValueError: If band_pct is empty.
  """
  if not band_pct:
    raise ValueError("band_pct must be non-empty")
  top_name, _ = max(band_pct.items(), key=lambda kv: float(kv[1]))
  return str(top_name)


def _confidence_bucket(confidence: float, cfg: dict[str, Any]) -> float:
  """
  Bucket a confidence ratio to the nearest configured confidence bucket.

  Args:
    confidence: Ratio (0..1) or percent-like (>1) confidence.
      - Per FR-NLP-008 cache key uses nearest bucket from `cfg['nlp']['caching']['confidence_buckets']`.

    cfg: Full configuration mapping.

  Returns:
    float bucket value taken from cfg.

  Raises:
    KeyError: If caching buckets are missing from cfg.
  """
  ratio = _confidence_to_ratio(confidence)
  buckets = cfg["nlp"]["caching"]["confidence_buckets"]
  bucket = min((float(b) for b in buckets), key=lambda b: abs(b - ratio))
  return float(bucket)


def _cache_enabled(cfg: dict[str, Any]) -> bool:
  """
  Check whether explanation caching is enabled.

  Args:
    cfg: Full configuration mapping.

  Returns:
    True if caching is enabled for NLP.

  Raises:
    KeyError: If cfg structure lacks expected keys.
  """
  return bool(cfg["nlp"]["caching"].get("enabled", False))


def _cache_key(label: str, confidence: float, band_pct: dict[str, float], cfg: dict[str, Any]) -> tuple[str, float, str]:
  """
  Build the cache key used for NLP explanation caching.

  Args:
    label: Model label (e.g., 'bonafide' or 'spoof').
    confidence: Confidence (ratio or percent-like); bucketed to nearest configured value.
    band_pct: 4-band attribution dict.
    cfg: Full configuration mapping.

  Returns:
    Tuple of (label, confidence_bucket, top_band_name).

  Raises:
    ValueError: If band_pct is empty.
  """
  top_band = _get_top_band_name(band_pct)
  bucket = _confidence_bucket(confidence, cfg)
  return str(label), bucket, top_band


def _get_cache_dict(cfg: dict[str, Any]) -> dict[tuple[str, float, str], _CacheRecord]:
  """
  Return the in-memory cache dict stored inside cfg.

  Args:
    cfg: Full configuration mapping. Cache is stored under `cfg['nlp']['_explanation_cache']`.

  Returns:
    Cache mapping from cache_key => _CacheRecord.

  Raises:
    KeyError: If cfg structure is missing 'nlp'.
  """
  cfg_nlp = cfg["nlp"]
  cache = cfg_nlp.setdefault("_explanation_cache", {})
  # typing: cache is intentionally mutable inside cfg for runtime memoization
  return cache


def build_prompt(label: str, confidence: float, band_pct: dict[str, float], cfg: dict[str, Any]) -> str:
  """
  Construct a structured prompt for the primary Qwen explanation provider.

  Args:
    label: Predicted class label from CV ('bonafide' or 'spoof').
    confidence: CV confidence (probability ratio in (0, 1) per FR-CV-004 or percent-like >1).
      - Used in prompt as a numeric confidence value.
    band_pct: Dict[str, float] with 4 frequency bands; values are percentages summing to ~100.0.
    cfg: Full configuration mapping (includes prompt sentence constraints).

  Returns:
    A prompt string containing: label, confidence, and all 4 band percentages.

  Raises:
    ValueError: If band_pct does not contain 4 bands.
  """
  expected_keys = ["low", "low_mid", "high_mid", "high"]
  missing = [k for k in expected_keys if k not in band_pct]
  if missing:
    raise ValueError(f"band_pct missing required keys: {missing}")

  top_band = _get_top_band_name(band_pct)
  conf_ratio = _confidence_to_ratio(confidence)
  conf_pct = conf_ratio * 100.0

  min_s = int(cfg["nlp"]["explanation_min_sentences"])
  max_s = int(cfg["nlp"]["explanation_max_sentences"])
  if min_s < 1 or max_s < min_s:
    raise ValueError("Invalid sentence constraints in cfg['nlp']")

  bands_block = "\n".join(
    f"- {k}: {float(band_pct[k]):.2f}%"
    for k in expected_keys
  )

  prompt = (
    "You are a reliable explainability assistant for a deepfake speech detector.\n"
    f"Predicted label: {label}\n"
    f"Confidence (probability): {conf_ratio}\n"
    f"Confidence (%): {conf_pct:.2f}%\n"
    "4 frequency-band attribution percentages (sum ~= 100%):\n"
    f"{bands_block}\n"
    f"Highest-attribution band: {top_band}\n\n"
    f"Instruction: Produce {min_s}-{max_s} sentences of clear English explaining the result.\n"
    f"Your explanation MUST cite the highest-attribution band and connect it to the label.\n"
    "Avoid speculation beyond what the attribution implies."
  )
  return prompt


async def call_qwen_api(prompt: str, cfg: dict[str, Any]) -> str:
  """
  Call Qwen 2.5 via an OpenAI-compatible async API client (primary provider).

  Args:
    prompt: User prompt string from `build_prompt`.
    cfg: Full configuration mapping.
      - Uses cfg['nlp']['api_key_env_var'] to look up the API key via os.environ.
      - Uses cfg['nlp']['timeout_sec'] as the asyncio.wait_for timeout.

  Returns:
    Provider response text as str (English explanation).

  Raises:
    NLPTimeoutError: On timeout or any provider exception.
      - Per FR-NLP-002: enforce timeout and use this error type for fallback.
  """
  api_key_env = str(cfg["nlp"]["api_key_env_var"])
  api_key = os.environ.get(api_key_env)
  timeout_sec = float(cfg["nlp"]["timeout_sec"])

  try:
    # Lazy import to keep module import side effects minimal in test runners.
    from openai import AsyncOpenAI  # type: ignore

    # Read OpenAI-compatible base URL from config (HF Inference).
    pythonbase_url = str(cfg["nlp"].get("base_url", "https://api-inference.huggingface.co/v1/"))
    model = str(cfg["nlp"]["primary_provider"])

    client = AsyncOpenAI(api_key=api_key, base_url=pythonbase_url)
    system_msg = "Reply in English. Provide 3-5 sentences. Cite the highest attribution band."
    coro = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
      ],
    )
    resp = await asyncio.wait_for(coro, timeout=timeout_sec)

    content = resp.choices[0].message.content if resp.choices else None
    return str(content or "").strip()
  except asyncio.TimeoutError as exc:
    raise NLPTimeoutError("Qwen API timeout") from exc
  except Exception as exc:
    raise NLPTimeoutError("Qwen API error") from exc


async def call_gemma_fallback(prompt: str, cfg: dict[str, Any]) -> str:
  """
  Call the secondary fallback provider (e.g., Gemma-3) via an OpenAI-compatible async client.

  Args:
    prompt: Prompt string from `build_prompt`.
    cfg: Full configuration mapping.
      - Uses cfg['nlp']['timeout_sec'] as the asyncio.wait_for timeout.

  Returns:
    Provider response text as str.

  Raises:
    NLPTimeoutError: On timeout or any provider exception.
      - Per FR-NLP-007 SHOULD: enforce timeout and use this error type for fallback.
  """
  timeout_sec = float(cfg["nlp"]["timeout_sec"])
  try:
    from openai import AsyncOpenAI  # type: ignore

    # Read OpenAI-compatible base URL from config (HF Inference).
    pythonbase_url = str(cfg["nlp"].get("base_url", "https://api-inference.huggingface.co/v1/"))
    model = str(cfg["nlp"]["fallback_provider"])

    # Prefer a Qwen-proxy key if present; otherwise allow stitch-mcp to handle auth upstream.
    api_key_env = str(cfg["nlp"]["api_key_env_var"])
    api_key = os.environ.get(api_key_env)

    client = AsyncOpenAI(api_key=api_key, base_url=pythonbase_url)
    coro = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": "Reply in English. Provide 3-5 sentences. Cite the highest attribution band."},
        {"role": "user", "content": prompt},
      ],
    )
    resp = await asyncio.wait_for(coro, timeout=timeout_sec)
    content = resp.choices[0].message.content if resp.choices else None
    return str(content or "").strip()
  except asyncio.TimeoutError as exc:
    raise NLPTimeoutError("Gemma fallback timeout") from exc
  except Exception as exc:
    raise NLPTimeoutError("Gemma fallback error") from exc


def build_rule_based_explanation(label: str, confidence: float, band_pct: dict[str, float]) -> str:
  """
  Generate a deterministic, always-available rule-based English explanation.

  Args:
    label: Predicted class label ('bonafide' or 'spoof').
    confidence: CV confidence (ratio in (0, 1) or percent-like >1).
    band_pct: Frequency-band attribution dict with 4 required keys.

  Returns:
    English explanation string with 3-5 sentences.
      - Per FR-NLP-003/FR-NLP-004, must be grammatically correct English.

  Raises:
    ValueError: If required band keys are missing.
  """
  expected_keys = ["low", "low_mid", "high_mid", "high"]
  missing = [k for k in expected_keys if k not in band_pct]
  if missing:
    raise ValueError(f"band_pct missing required keys: {missing}")

  top_band = _get_top_band_name(band_pct)
  top_val = float(band_pct[top_band])
  conf_pct = _ratio_to_percent_text(confidence)

  label_norm = str(label).strip().lower()
  label_phrase = "bonafide" if label_norm == "bonafide" else "spoof" if label_norm == "spoof" else label

  # Minimal, attribution-grounded heuristics.
  if top_band == "low":
    interpret = "dominant low-frequency energy can reflect natural speech cadence"
  elif top_band == "low_mid":
    interpret = "mid-frequency emphasis often aligns with phonetic structure"
  elif top_band == "high_mid":
    interpret = "high-mid activation can indicate texture and consonant-like components"
  else:
    interpret = "high-frequency concentration can be associated with sharper spectral artifacts"

  # FR-NLP-003 template requirement.
  sentence1 = f"Analysis indicates {label_phrase} speech with {conf_pct:.2f}% confidence."
  sentence2 = f"The {top_band} frequency band ({top_val:.2f}%) showed the highest activation, suggesting {interpret}."

  # One extra sentence conditioned on label for stronger rule-based alignment.
  if label_norm == "spoof":
    sentence3 = "Overall, the strongest band suggests the model is leveraging spectral characteristics typical of synthetic or manipulated speech."
  else:
    sentence3 = "Overall, the strongest band suggests the model is leveraging natural spectral patterns consistent with genuine speech."

  sentence4 = (
    f"This attribution summary is based on relative band activation rather than external metadata, "
    f"so it reflects the model's focus during classification."
  )

  # Keep within 3-5 sentences; return 4 sentences for robustness.
  return " ".join([sentence1, sentence2, sentence3, sentence4]).strip()


async def generate_explanation(
  label: str,
  confidence: float,
  band_pct: dict[str, float],
  cfg: dict[str, Any],
) -> tuple[str, bool]:
  """
  Orchestrate explanation generation with caching and a Qwen -> Gemma -> rule-based fallback chain.

  Args:
    label: Predicted class label ('bonafide' or 'spoof').
    confidence: CV confidence (probability ratio in (0, 1) or percent-like >1).
    band_pct: Frequency-band attribution dict with 4 required keys.
    cfg: Full configuration mapping.

  Returns:
    (explanation_text, api_was_used)
      - api_was_used is False only when the final output is rule-based due to API failures (FR-NLP-003).

  Raises:
    ValueError: If inputs are invalid (e.g., missing bands).
  """
  if _cache_enabled(cfg):
    cached_text = get_cached_explanation(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg)
    if cached_text is not None:
      cache_key = _cache_key(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg)
      cache_record = _get_cache_dict(cfg).get(cache_key)
      api_was_used = bool(cache_record.api_was_used) if cache_record is not None else True
      return cached_text, api_was_used

  prompt = build_prompt(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg)
  api_was_used = False
  explanation_text = ""

  # Qwen 2.5 primary.
  try:
    explanation_text = await call_qwen_api(prompt=prompt, cfg=cfg)
    api_was_used = True
    log_info(stage="nlp", message="qwen_explanation_success", data={"label": label})
  except NLPTimeoutError:
    log_warning(stage="nlp", message="qwen_explanation_failed_fallback", data={"label": label})

  if api_was_used:
    if _cache_enabled(cfg):
      cache_key = _cache_key(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg)
      _get_cache_dict(cfg)[cache_key] = _CacheRecord(text=explanation_text, api_was_used=True)
    return explanation_text, True

  # Secondary Gemma-3 fallback (SHOULD).
  try:
    explanation_text = await call_gemma_fallback(prompt=prompt, cfg=cfg)
    api_was_used = True
    log_info(stage="nlp", message="gemma_fallback_success", data={"label": label})
  except NLPTimeoutError:
    log_warning(stage="nlp", message="gemma_explanation_failed_rule_fallback", data={"label": label})

  if api_was_used:
    if _cache_enabled(cfg):
      cache_key = _cache_key(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg)
      _get_cache_dict(cfg)[cache_key] = _CacheRecord(text=explanation_text, api_was_used=True)
    return explanation_text, True

  # Final always-available rule-based explanation.
  explanation_text = build_rule_based_explanation(label=label, confidence=confidence, band_pct=band_pct)
  api_was_used = False

  if _cache_enabled(cfg):
    cache_key = _cache_key(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg)
    _get_cache_dict(cfg)[cache_key] = _CacheRecord(text=explanation_text, api_was_used=False)

  return explanation_text, api_was_used


def get_cached_explanation(label: str, confidence: float, band_pct: dict[str, float], cfg: dict[str, Any]) -> str | None:
  """
  Retrieve a cached NLP explanation from cfg (if caching is enabled).

  Args:
    label: Predicted class label.
    confidence: Confidence used to compute a cache bucket key.
    band_pct: Frequency-band attribution dict used to compute top band name.
    cfg: Full configuration mapping.

  Returns:
    Cached explanation text if present, else None.

  Raises:
    ValueError: If band_pct is empty.
  """
  if not _cache_enabled(cfg):
    return None
  cache_key = _cache_key(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg)
  cache_record = _get_cache_dict(cfg).get(cache_key)
  return cache_record.text if cache_record is not None else None
