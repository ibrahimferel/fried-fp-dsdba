"""
Module: src.utils.errors
SRS Reference: FR-AUD-005, FR-AUD-001 (error codes AUD-001, AUD-002) and NFR-Security
SDLC Phase: 3 - Environment Setup & MCP Configuration
Sprint: N/A
Pipeline Stage: Deployment
Purpose: Provide stable SRS-aligned error codes for UI-safe failure handling.
Dependencies: dataclasses, typing (no runtime external deps).
Interface Contract:
  Input:  error code, human-safe message, and stage identifier
  Output: DSDBAError exception carrying `code`, `message`, `stage`
Latency Target: <= 1 ms (error construction; not on critical inference path)
Open Questions Resolved: Q3/Q4/Q5/Q6 resolved as design constraints only
Open Questions Blocking: None for this utility
MCP Tools Used: N/A
AI Generated: true
Verified (V.E.R.I.F.Y.): true
Author: Ferel / Safa
Date: 2026-03-22
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DSDBAError(Exception):
    """
    DSDBAError is a structured exception used across the pipeline.

    The `code` is SRS-aligned (e.g., "AUD-001", "AUD-002") so UI layers can map
    failures to stable, user-safe messages without leaking implementation details.
    """

    code: str  # e.g., "AUD-001" per FR-AUD-005
    message: str
    stage: str

    def __post_init__(self) -> None:
        # Preserve standard Exception args for better debuggability.
        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        return f"[{self.code}] ({self.stage}) {self.message}"