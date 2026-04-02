"""
Module: app (UI wiring)
SRS Reference: FR-DEP-001-010
SDLC Phase: 3 - Environment Setup & MCP Configuration
Sprint: E
Pipeline Stage: Deployment
Purpose: UI wiring that orchestrates the full feed-forward pipeline and renders CV/XAI/NLP outputs.
Dependencies: gradio, asyncio.
Interface Contract:
  Input: user-uploaded audio file object (WAV/FLAC, <= 20 MB)
  Output: staged UI payload including CV verdict, confidence, Grad-CAM heatmap asset, band_pct[4], and NLP explanation (with fallback)
Latency Target: <= 15,000 ms end-to-end on CPU per FR-DEP-007
Open Questions Resolved: Q3 pending empirical Colab run; Q4/Q5/Q6 resolved for integration contracts
Open Questions Blocking: Q3 may affect training/export viability before deployment
MCP Tools Used: context7-mcp, huggingface-mcp, stitch-mcp
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-22
"""