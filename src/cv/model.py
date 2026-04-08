"""
Module: src.cv.model
SRS Reference: FR-CV-001 | FR-CV-002
SDLC Phase: 4 — Implementation (Sprint B)
Sprint: B
Pipeline Stage: CV Inference
Interface Contract:
  Input: torch.Tensor [B, 3, 224, 224] float32
  Output: torch.Tensor [B, 2] float32 (binary logits)
Latency Target: ≤ 3,000 ms forward proxy (NFR-Performance); ONNX ≤ 1,500 ms (FR-DEP-010)
Open Questions Resolved: Q3 VRAM (Colab); Q4–Q6 design locks
Open Questions Blocking: Q7 EER protocol note in compute_eer (train.py)
MCP Tools Used: context7-mcp (torchvision EfficientNet-B4) | huggingface-mcp
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-29
"""

from __future__ import annotations

from typing import Any
from urllib.error import URLError

import torch
from torch import Tensor, nn
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

from src.utils.logger import log_warning


class DSDBAModel(nn.Module):
  """EfficientNet-B4 backbone with 2-class linear head for bonafide/spoof."""

  def __init__(self, cfg: dict[str, Any] | None = None, pretrained: bool = True) -> None:
    super().__init__()
    self.cfg = cfg or {}
    model_cfg = self.cfg.get("model", {})
    num_classes = int(model_cfg.get("num_classes", 2))

    use_imagenet = pretrained and str(model_cfg.get("pretrained_weights", "imagenet1k")).lower() == "imagenet1k"
    weights = EfficientNet_B4_Weights.DEFAULT if use_imagenet else None
    try:
      self.backbone = efficientnet_b4(weights=weights)
    except (OSError, RuntimeError, FileNotFoundError, ValueError, URLError) as exc:
      # Offline or cache-miss: fall back to random init for tests/local runs.
      self.backbone = efficientnet_b4(weights=None)
      log_warning(
        stage="cv_model",
        message="efficientnet_pretrained_unavailable",
        data={"reason": str(exc)},
      )

    in_features = int(self.backbone.classifier[-1].in_features)
    self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

  def freeze_backbone(self) -> None:
    for param in self.backbone.features.parameters():
      param.requires_grad = False
    for param in self.backbone.avgpool.parameters():
      param.requires_grad = False
    for param in self.backbone.classifier.parameters():
      param.requires_grad = True

  def unfreeze_top_n(self, n: int) -> None:
    n = max(0, int(n))
    features = list(self.backbone.features.children())
    if n == 0:
      return

    for block in features[-n:]:
      for param in block.parameters():
        param.requires_grad = True

    for param in self.backbone.classifier.parameters():
      param.requires_grad = True

  def forward(self, x: Tensor) -> Tensor:
    """Return logits with shape [batch, 2]."""
    return self.backbone(x)