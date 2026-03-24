# DSDBA — Phase 2 Module Interface Contracts

**Document:** DSDBA-SRS-2026-002 v2.1  
**Phase:** 2 — System Design & Technical Specification  
**Label:** [Phase 2 | v1 | Interface Contracts]

## Contract 1 — `src/audio/dsp.py`

`preprocess_audio(file_path: Path) -> torch.Tensor`

- **Input:** `Path` to WAV/FLAC file.
- **Output:** `torch.Tensor` shape `[3, 224, 224]`, dtype `torch.float32`.
- **Raises:**
  - `ValueError("AUD-001")` if duration `< 0.5 s` (FR-AUD-005)
  - `ValueError("AUD-002")` if format unsupported (FR-AUD-001)
- **Latency target:** `<= 500 ms` CPU (NFR-Performance)

## Contract 2 — `src/cv/infer.py`

`run_inference(tensor: torch.Tensor) -> tuple[str, float]`

- **Input:** tensor `[3, 224, 224]` float32.
- **Output:** `(label, confidence)` where label in `{bonafide, spoof}`, confidence in `(0, 1)`.
- **Latency target:** `<= 1,500 ms` CPU via ONNX Runtime (FR-DEP-010).

## Contract 3 — `src/cv/gradcam.py`

`run_gradcam(tensor: torch.Tensor, model) -> tuple[Path, dict[str, float]]`

- **Input:** tensor `[3, 224, 224]` float32 + trained EfficientNet-B4 model.
- **Output:** `(heatmap_png_path, band_attributions)` with exactly 4 keys (`low`, `low_mid`, `high_mid`, `high`) summing to `100.0 ± 0.001`.
- **Latency target:** `<= 3,000 ms` CPU (FR-CV-015).

## Contract 4 — `src/nlp/explain.py`

`generate_explanation(label: str, confidence: float, band_pct: dict[str, float]) -> str`

- **Input:** label, confidence, 4-band attribution dict (`sum == 100.0 ± 0.001`).
- **Output:** English paragraph (3–5 sentences) (FR-NLP-001, FR-NLP-004).
- **Fallback:** rule-based template when Qwen API fails or times out (FR-NLP-003).
- **Latency target:** API path `<= 8,000 ms`; fallback `<= 100 ms` (NFR-Performance).

## Contract 5 — `app.py`

`run_pipeline_ui(file_obj) -> dict`

- **Input:** user upload (`WAV/FLAC`, max 20 MB).
- **Output:** staged UI payload containing CV result, Grad-CAM asset, band percentages, NLP explanation/fallback.
- **Ordering rule:** CV panel MUST be visible before NLP resolution (FR-NLP-006).
- **Latency target:** end-to-end `<= 15,000 ms` on CPU (FR-DEP-007).

## Config lock statement

This contract locks Phase 2 decisions. Sprints A–E SHALL implement these signatures and constraints without redesign.
