"""
Module: src.utils.logger
SRS Reference: NFR-Security, NFR-Maintainability (structured logging to replace print)
SDLC Phase: 3 - Environment Setup & MCP Configuration
Sprint: N/A
Pipeline Stage: Deployment
Purpose: Provide structured JSON logging for pipeline stages and UI integration (no print()).
Dependencies: logging, json.
Interface Contract:
  Input:  structured event dict (stage, code, message, timing)
  Output: JSON-serialisable log record (stdout/stderr or logger backend)
Latency Target: <= 5 ms per log event (logging not on critical inference path)
Open Questions Resolved: None (utility scaffold)
Open Questions Blocking: None
MCP Tools Used: context7-mcp
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-22
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


_LOGGER_NAME = "dsdba"
_LOGGER = logging.getLogger(_LOGGER_NAME)

if not _LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(_handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


def _emit(level: int, stage: str, message: str, data: dict[str, Any] | None = None) -> None:
    """
    Emit a structured JSON log event.

    Args:
        level: Logging level from `logging` module.
        stage: Pipeline stage identifier (e.g., "audio_dsp").
        message: Human-readable event message.
        data: Optional JSON-serialisable metadata payload.

    Returns:
        None.

    Raises:
        TypeError: If `data` contains values that are not JSON serialisable.
    """

    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": logging.getLevelName(level),
        "stage": stage,
        "message": message,
        "data": data or {},
    }
    _LOGGER.log(level, json.dumps(payload, ensure_ascii=True))


def log_info(stage: str, message: str, data: dict[str, Any] | None = None) -> None:
    """
    Log an INFO-level structured JSON event.

    Args:
        stage: Pipeline stage identifier.
        message: Event message.
        data: Optional JSON-serialisable metadata payload.

    Returns:
        None.
    """

    _emit(logging.INFO, stage=stage, message=message, data=data)


def log_warning(stage: str, message: str, data: dict[str, Any] | None = None) -> None:
    """
    Log a WARNING-level structured JSON event.

    Args:
        stage: Pipeline stage identifier.
        message: Event message.
        data: Optional JSON-serialisable metadata payload.

    Returns:
        None.
    """

    _emit(logging.WARNING, stage=stage, message=message, data=data)


def log_error(stage: str, message: str, data: dict[str, Any] | None = None) -> None:
    """
    Log an ERROR-level structured JSON event.

    Args:
        stage: Pipeline stage identifier.
        message: Event message.
        data: Optional JSON-serialisable metadata payload.

    Returns:
        None.
    """

    _emit(logging.ERROR, stage=stage, message=message, data=data)