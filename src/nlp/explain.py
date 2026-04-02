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
  Output: English explanation paragraph (3-5 sentences)
Latency Target: <= 8,000 ms API path; <= 100 ms fallback
Open Questions Resolved: Q6 resolved (UI ordering contract), Q4/Q5 resolved (XAI inputs contract)
Open Questions Blocking: None for Sprint D
MCP Tools Used: context7-mcp | huggingface-mcp | stitch-mcp
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-22
"""