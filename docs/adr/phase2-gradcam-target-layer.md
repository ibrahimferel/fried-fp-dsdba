# ADR — Phase 2 Q4 Resolution: Grad-CAM Target Layer (EfficientNet-B4)

**Document:** DSDBA-SRS-2026-002 v2.1  
**Phase:** 2 — System Design & Technical Specification  
**SRS refs:** FR-CV-010, FR-CV-011, FR-CV-015  
**Label:** [Phase 2 | v1 | Q4-RESOLVED]

## Decision

Q4 is resolved with the locked target layer path:

- `gradcam.target_layer = "model.features[8]"`
- Canonical module reference for pytorch-grad-cam: `target_layers = [model.features[8]]`

## Evidence

Terminal introspection result (local `.venv`, `torchvision==0.16.0`):

- `type(model.features[-1]) == torchvision.ops.misc.Conv2dNormActivation`
- Named modules include:
  - `features.8` (`Conv2dNormActivation`)
  - `features.8.0` (`Conv2d`)
  - `features.8.1` (`BatchNorm2d`)
  - `features.8.2` (`SiLU`)

This confirms that the final feature block is addressable as `features.8` and is equivalent to `features[-1]` for the current torchvision graph.

## Alternatives considered

1. Keep `model.features[-1]` literal in config.
2. Use explicit index `model.features[8]`.

**Selected:** option 2 for deterministic, serialization-friendly path matching `named_modules()` output.

## Consequences

- Sprint C implementation SHALL use config-driven layer selection.
- If torchvision graph version changes, Sprint C smoke tests must validate the path and update ADR if needed.

## MCP note

- `context7-mcp` used for API alignment context; runtime confirmation used local introspection.
