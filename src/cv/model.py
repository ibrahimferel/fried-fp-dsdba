"""
Module: src.cv.model
SRS Reference: FR-CV-001-002
SDLC Phase: 3 - Environment Setup & MCP Configuration
Sprint: B
Pipeline Stage: CV Inference
Purpose: Define the EfficientNet-B4 backbone and binary classification head used for bonafide/spoof prediction.
Dependencies: torch, torchvision.
Interface Contract:
  Input:  torch.Tensor [B, 3, 224, 224] float32
  Output: torch.Tensor [B, 2] float32 (logits for bonafide/spoof)
Latency Target: <= 3,000 ms per NFR-Performance (training/infer proxy target; actual timing validated in Sprint B)
Open Questions Resolved: Q4/Q5/Q6 resolved (affects XAI/UI later)
Open Questions Blocking: Q3 - VRAM feasibility affects Sprint B runtime/batch sizing
MCP Tools Used: context7-mcp | huggingface-mcp | stitch-mcp
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-22
"""

from __future__ import annotations

from typing import Any

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
    except Exception as exc:  # noqa: BLE001
      # Offline or cache-miss environments should still be able to run tests/training.
      self.backbone = efficientnet_b4(weights=None)
      log_warning(
        stage="cv_model",
        message="efficientnet_pretrained_unavailable",
        data={"reason": str(exc)},
      )

    in_features = int(self.backbone.classifier[-1].in_features)
    self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

  def freeze_backbone(self) -> None:
    """Freeze all backbone parameters except classifier head."""
    for param in self.backbone.features.parameters():
      param.requires_grad = False
    for param in self.backbone.avgpool.parameters():
      param.requires_grad = False
    for param in self.backbone.classifier.parameters():
      param.requires_grad = True

  def unfreeze_top_n(self, n: int) -> None:
    """Unfreeze top-N EfficientNet feature blocks for fine-tuning."""
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