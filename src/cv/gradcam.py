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