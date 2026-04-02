# DSDBA — Master Implementation Plan (MIP) Draft

**Document:** DSDBA-SRS-2026-002 v2.1  
**SDLC:** Phases 0–7  
**Label:** [Phase 0 | v1 | MIP-draft | SRS-ref: all FR buckets]

---

## Legend

- **Blocking Q:** Open Question that must be resolved before exiting the phase gate (see `docs/context/session-cheatsheet.md`).
- Checkbox items are **exit criteria** for planning; implementation verification follows Chain 00 gate checks.

---

## Phase 0 — Project Inception & Architecture

| ID | Task | SRS FR / NFR owned | Blocking Open Question |
|----|------|---------------------|-------------------------|
| 0.1 | Confirm sequential multimodal pipeline (Audio → CV → XAI → NLP → UI) | NFR-Maintainability, Section 0.1 | — |
| 0.2 | Tech stack rationale (PyTorch train, ONNX deploy, Gradio default) | FR-DEP-010, FR-CV-007 | Q6 (UI framework lock) |
| 0.3 | ADR MCP selection | ADR-0001 | — |
| 0.4 | Pipeline diagram + risk register | FR-AUD-008, FR-CV-*, coupling | — |
| 0.5 | Initial Q3/Q4 assessment | Sprint B (Q3), Sprint C (Q4) | Q3, Q4 |

- [x] 0.1 Pipeline architecture baseline documented
- [x] 0.2 Stack rationale documented (diagram + ADR-0001 + risk register)
- [x] 0.3 ADR-0001 accepted (`docs/adr/phase0-mcp-selection.md`)
- [x] 0.4 Risk register + diagram in `docs/adr/`
- [x] 0.5 Q3/Q4 assessment recorded in `docs/context/session-cheatsheet.md`

---

## Phase 1 — Requirements & Backlog

| ID | Task | SRS FR / NFR owned | Blocking Open Question |
|----|------|---------------------|-------------------------|
| 1.1 | Backlog from SRS FR tables | All FR-* | — |
| 1.2 | SHALL / SHOULD / MAY prioritisation | All | — |
| 1.3 | Traceability matrix (FR → module → test) | NFR-Maintainability | — |
| 1.4 | Q5 Mel bin mapping; Q6 UI framework | FR-CV-013, FR-DEP-001 | Q5, Q6 |

- [ ] 1.1–1.3 Backlog + traceability
- [ ] 1.4 Resolve or defer Q5/Q6 with documented rationale

---

## Phase 2 — System Design & Technical Specification

| ID | Task | SRS FR / NFR owned | Blocking Open Question |
|----|------|---------------------|-------------------------|
| 2.1 | Audio DSP contract | FR-AUD-001–008 | — |
| 2.2 | CV interface + ONNX | FR-CV-001–009, FR-DEP-010 | Q3 |
| 2.3 | Grad-CAM + 4-band Softmax | FR-CV-010–016 | Q4, Q5 |
| 2.4 | NLP async interface | FR-NLP-001–009 | — |
| 2.5 | Deployment + UI spec | FR-DEP-001–010 | Q6 |
| 2.6 | `config.yaml` spec (no magic numbers) | NFR-Maintainability | — |
| 2.7 | `.cursorrules` alignment | NFR-Maintainability | — |

---

## Phase 3 — Environment Setup & MCP Configuration

| ID | Task | SRS FR / NFR owned | Blocking Open Question |
|----|------|---------------------|-------------------------|
| 3.1 | Colab: PyTorch, librosa, torchaudio | FR-CV-003 | Q3 |
| 3.2 | Cursor workspace: `src/`, requirements pinned | NFR-Security | — |
| 3.3 | context7-mcp configuration | ADR-0001 | — |
| 3.4 | huggingface-mcp + secrets pattern | FR-DEP, FR-CV-007 | — |
| 3.5 | stitch-mcp + async NLP | FR-NLP-002, FR-NLP-003 | — |
| 3.6 | GitHub scaffold: README, requirements, config | FR-DEP | — |

---

## Phase 4 — Feature Development (Sprints)

| Sprint | Module | SRS FR IDs | Blocking Open Question |
|--------|--------|------------|-------------------------|
| A | `src/audio/dsp.py` | FR-AUD-001–011 | — |
| B | `src/cv/model.py`, `train.py`, ONNX export | FR-CV-001–009, FR-DEP-010 | **Q3** |
| C | `src/cv/gradcam.py` | FR-CV-010–016 | **Q4**, **Q5** |
| D | `src/nlp/explain.py` | FR-NLP-001–009 | — |
| E | `app.py` UI | FR-DEP-001–010 | **Q6** |

---

## Phase 5 — Integration & End-to-End Testing

| ID | Task | SRS FR / NFR owned | Blocking Open Question |
|----|------|---------------------|-------------------------|
| 5.1 | E2E WAV → verdict + explanation | FR-DEP-007 | — |
| 5.2 | Accuracy: EER, AUC-ROC on FoR | FR-CV-008 | **Q7** |
| 5.3 | Latency benchmark per stage | NFR-Performance | — |
| 5.4 | NLP resilience (timeout → fallback) | FR-NLP-003 | — |
| 5.5 | Security audit | NFR-Security | — |

---

## Phase 6 — Build, Release & Hugging Face Spaces

| ID | Task | SRS FR / NFR owned | Blocking Open Question |
|----|------|---------------------|-------------------------|
| 6.1 | ONNX verification ≤ 1,500 ms CPU | FR-DEP-010 | — |
| 6.2 | Spaces deploy + secrets | FR-DEP-001 | — |
| 6.3 | Demo samples (2 bonafide + 2 spoof) | FR-DEP-008 | — |
| 6.4 | About section | FR-DEP-009 | — |
| 6.5 | Release freeze (SRS review cycle) | NFR | — |

---

## Phase 7 — Monitoring, Maintenance & Retrospective

| ID | Task | SRS FR / NFR owned | Blocking Open Question |
|----|------|---------------------|-------------------------|
| 7.1 | Cold-start &lt; 30 s on Spaces | NFR-Reliability | — |
| 7.2 | EER protocol finalisation | FR-CV-008 | **Q7** |
| 7.3 | Retrospective (accuracy, latency, XAI) | NFR | — |

---

## Phase gate summary

| Phase | Cannot proceed until |
|-------|----------------------|
| 2 → 3 | Q3 plan accepted (VRAM) |
| 3 → 4 | Environment reproducible |
| 4 Sprint B | Q3 resolved or waived with ADR |
| 4 Sprint C | Q4, Q5 resolved |
| 4 Sprint E | Q6 resolved |
| 5 → 6 | **Q7** protocol for EER scoring agreed or deferred with justification |

---

*This MIP is a living document; update after each sprint closeout in `docs/context/session-cheatsheet.md`.*
