# DSDBA — Phase 1 Prioritised Backlog

**Document:** DSDBA-SRS-2026-002 v2.1  
**SDLC Phase:** 1 — Requirements & Backlog Definition  
**Label:** [Phase 1 | v1 | SRS-ref: FR-AUD, FR-CV, FR-NLP, FR-DEP]

---

## Priority legend

| Tag | Meaning | Planning rule |
|-----|---------|----------------|
| **SHALL** | Sprint critical — minimum releasable contract | Must complete in listed sprint |
| **SHOULD** | Sprint target — SRS desirable | Defer only with recorded justification + no SHALL break |
| **MAY** | If time / optional | Safe to defer without violating core SHALL set |

---

## Sprint A — Audio DSP (`src/audio/dsp.py`)

| Priority | FR-ID | Notes |
|----------|-------|--------|
| SHALL | FR-AUD-001, 002, 003, 004, 005, 006, 007, 008 | Core ingest → Mel → `[3,224,224]` float32 pipeline |
| SHOULD | FR-AUD-009 | Batch / vectorised API for throughput |
| SHOULD | FR-AUD-010 | Structured JSON logging of DSP stages |
| MAY | FR-AUD-011 | Optional MP3/OGG decode (torchaudio) |

**Blocking Open Question:** none.

---

## Sprint B — CV training + ONNX (`src/cv/model.py`, `train.py`, export + `infer.py`)

| Priority | FR-ID | Notes |
|----------|-------|--------|
| SHALL | FR-CV-001, 002, 003, 004, 005, 007, 008 | Architecture, head, training, metrics, Hub, acceptance criteria |
| SHALL | FR-DEP-010 | ONNX export, numerical equivalence, CPU inference path |
| SHOULD | FR-CV-006 | SpecAugment + augmentation policy |
| MAY | FR-CV-009 | EfficientNet-B0 baseline ablation |

**Blocking Open Question:** **Q3** (Colab VRAM / training feasibility) — **must be resolved before Sprint B starts.**

---

## Sprint C — XAI / Grad-CAM (`src/cv/gradcam.py`)

| Priority | FR-ID | Notes |
|----------|-------|--------|
| SHALL | FR-CV-010, 011, 012, 013, 014, 015 | Layer lock, library, heatmap, Hz bands, Softmax 100%, latency |
| SHOULD | FR-CV-016 | Raw saliency JSON (e.g. dev endpoint) |

**Blocking Open Questions:** **Q4** (Grad-CAM layer) **and** **Q5** (Mel bin ↔ Hz mapping) — **both must be resolved before Sprint C starts.**

---

## Sprint D — NLP / Qwen 2.5 (`src/nlp/explain.py`)

| Priority | FR-ID | Notes |
|----------|-------|--------|
| SHALL | FR-NLP-001, 002, 003, 004, 005, 006 | Explanation spec, async API, fallback, English, secrets, UI ordering |
| SHOULD | FR-NLP-007 | Secondary LLM (e.g. Gemma-3) fallback chain |
| SHOULD | FR-NLP-008 | Explanation caching |
| MAY | FR-NLP-009 | Language toggle (non-English) |

**Blocking Open Question:** none for sprint start (Q1 resolved).

---

## Sprint E — Deployment UI (`app.py`)

| Priority | FR-ID | Notes |
|----------|-------|--------|
| SHALL | FR-DEP-001, 002, 003, 004, 005, 006, 007 | Framework, upload limits, public flow, results UX, ONNX path in app, async NLP UX, E2E latency budget |
| SHOULD | FR-DEP-008 | Bundled demo samples (2 bonafide + 2 spoof) |
| SHOULD | FR-DEP-009 | About / dataset citation panel |
| SHALL | FR-DEP-010 | Already satisfied in Sprint B for model artefact; **verify** wiring + secrets + latency in UI integration |

**Note:** FR-DEP-010 is implemented primarily in Sprint B (ONNX); Sprint E **shall** integrate and verify end-to-end.

**Blocking Open Question:** **Q6** (Gradio vs Streamlit) — **must be resolved before Sprint E starts.**

---

## SHOULD / MAY deferral (without violating SHALL)

| FR-ID | Priority | Deferral condition |
|-------|----------|---------------------|
| FR-AUD-009, FR-AUD-010 | SHOULD | Sprint A still delivers single-file SHALL path; batch/logging post-MVP |
| FR-AUD-011 | MAY | WAV/FLAC only until MP3/OGG bandwidth allows |
| FR-CV-006 | SHOULD | Train without SpecAugment first; add when stability confirmed |
| FR-CV-009 | MAY | Skip B0 baseline if B4 track meets schedule |
| FR-CV-016 | SHOULD | Heatmap + JSON bands sufficient; raw saliency endpoint optional |
| FR-NLP-007, FR-NLP-008 | SHOULD | Rule-only + Qwen path acceptable if secondary LLM / cache deferred |
| FR-NLP-009 | MAY | English-only UI per FR-NLP-004 |
| FR-DEP-008, FR-DEP-009 | SHOULD | Ship with upload-only UI; demo clips + About in follow-up |

**Rule:** No deferral may remove a **SHALL** item from its sprint without Change Request to SRS.

---

## Q6 recommendation (Phase 1 — not a formal close)

**Recommendation:** Lock **Gradio 4.x** for Sprint E pending Phase 2 design sign-off.

| Criterion | Gradio 4.x | Notes |
|-----------|------------|--------|
| Audio widget | Native `Audio` I/O | Matches FR-DEP upload + playback for FoR clips |
| HF Spaces | First-class `gradio` + `requirements.txt` pattern | Lower ops risk vs custom Streamlit Dockerfile |
| Async NLP + CV ordering | `async` handlers + progressive UI patterns | Supports **FR-NLP-006** (show CV result before explanation completes) and **FR-DEP-006** (non-blocking explanation slot) |

**Status:** **Q6 remains OPEN** until Phase 2 gate / ADR update — this section is evidence for backlog prioritisation only.

---

## Traceability

Full **Requirement Traceability Matrix:** `docs/adr/phase1-rtm.md`.

**Count note:** Enumerated FR IDs in SRS ranges total **46** IDs (11 + 16 + 9 + 10). If SRS v2.1 document counts **37 parent requirements**, map many-to-one at Phase 2 traceability review.
