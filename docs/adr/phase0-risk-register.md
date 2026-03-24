# DSDBA — Phase 0 Architectural Risk Register

**Document:** DSDBA-SRS-2026-002 v2.1  
**SDLC Phase:** 0 — Project Inception & Architecture Design  
**SRS Traceability:** NFR-Reliability, FR-AUD-008, FR-CV-001, FR-CV-010, FR-NLP-003, FR-NLP-006  
**Label:** [Phase 0 | v1 | SRS-ref: FR-AUD-008, FR-CV-001, FR-CV-010, FR-NLP-003, FR-NLP-006]

---

## [EXPLORE] → [DECISION]

| Topic | Option A | Option B | Decision |
|-------|----------|----------|----------|
| Tensor validation | Validate only in `dsp.py` | Validate at **CV module entry** + tests | **B** — fail-fast at consumer boundary (FR-CV-001) |
| Grad-CAM layer | Guess from paper | **Lock in `config.yaml` + introspection in Sprint C** | **B** — Q4 must close before trusting saliency |
| NLP vs UI | Block UI until explanation returns | **Show CV first; NLP async** (FR-NLP-006) | **B** — matches SRS coupling |

---

## Risk 1 — Tensor shape contract violation `[3, 224, 224]`

| Field | Detail |
|--------|--------|
| **Description** | A malformed or drifted tensor from Audio DSP can propagate undetected into CV/XAI, causing silent misclassification or meaningless Grad-CAM. |
| **Coupling point** | **FR-AUD-008** → **FR-CV-001** (Audio tensor output → CV input contract). |
| **Likelihood** | Medium (integration errors, dtype/device mismatch). |
| **Impact** | High — invalid scientific and user-facing conclusions. |
| **Mitigation** | Validate `shape == (3, 224, 224)` and `dtype == float32` at **CV module entry** (`src/cv/infer.py`); duplicate assertion in `tests/test_cv.py` with golden vectors. |
| **Residual risk** | Low after tests and ONNX export checks. |

---

## Risk 2 — Grad-CAM target layer misidentification (Q4)

| Field | Detail |
|--------|--------|
| **Description** | If the wrong layer is attributed, heatmaps and **FR-CV-010–016** frequency-band mapping become misleading. |
| **Coupling point** | **FR-CV-010** → `model.features[-1]` (EfficientNet-B4) must match actual forward graph. |
| **Likelihood** | Medium until empirical confirmation. |
| **Impact** | High — XAI and FoR review credibility. |
| **Mitigation** | Keep `gradcam.target_layer` in `config.yaml`; in Sprint C, confirm via PyTorch module introspection and **pytorch-grad-cam** smoke tests; update ADR if path differs. |
| **Residual risk** | Tracked under Open Question **Q4** until closed. |

---

## Risk 3 — Qwen 2.5 latency blocking CV result display

| Field | Detail |
|--------|--------|
| **Description** | Slow or stalled LLM calls could delay or obscure primary detection results. |
| **Coupling point** | **FR-NLP-006** — UI SHALL show Stage 2 (CV) before Stage 4 (NLP) completes. |
| **Likelihood** | Medium on CPU-only Spaces and variable API latency. |
| **Impact** | Medium — usability and perceived reliability. |
| **Mitigation** | Isolate NLP in `asyncio` tasks; always render CV panel first; **FR-NLP-003** rule-based fallback + warning badge if timeout \> 30 s. |
| **Residual risk** | Low if UI contract is enforced in `app.py` integration tests. |

---

## Sign-off (Phase 0)

| Role | Action |
|------|--------|
| Architecture | Risks accepted for Phase 1–2 design freeze pending Q3–Q7 gates. |

**MCP tools used (documentation):** context7-mcp (library semantics for librosa / pytorch-grad-cam); huggingface-mcp (Spaces constraints); stitch-mcp (orchestration narrative only).
