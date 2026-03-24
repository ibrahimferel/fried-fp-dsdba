# DSDBA — Session Context Cheatsheet
**Document:** DSDBA-SRS-2026-002 v2.1 | Context Carry v1.0
**Rule:** READ THIS FILE at the start of every Cursor session (@file docs/context/session-cheatsheet.md)
**Rule:** UPDATE THIS FILE at the end of every Cursor session before closing.

---

## 📍 CURRENT STATUS

| Field | Value |
|-------|-------|
| **Active SDLC Phase** | Phase 3 — Environment Setup & MCP Configuration |
| **Active Sprint** | N/A |
| **Last Completed** | Chain 04 — Phase 3 scaffold: Colab notebook, README, src stubs, `src/utils/errors.py` |
| **Next Action** | Run Q3 Cell 4 in `notebooks/dsdba_training.ipynb` (Colab) and send peak VRAM value |
| **Gate Status** | 🟡 **Phase 3 IN PROGRESS** — blocked by Q3 empirical VRAM measurement |

---

## ✅ COMPLETED SRS REQUIREMENTS
*(none yet — update after each sprint)*

---

## 🔴 OPEN QUESTIONS STATUS

| Q# | Question | Status | Blocks |
|----|----------|--------|--------|
| Q1 | Qwen 2.5 API adoption | ✅ RESOLVED — Qwen 2.5 (Alibaba Cloud) confirmed | — |
| Q2 | FoR dataset variant | ✅ RESOLVED — for-2sec (2.0 s clips, 32,000 samples) | — |
| Q3 | EfficientNet-B4 VRAM on Colab | 🟡 PENDING — run notebook Cell 4 for actual peak VRAM | Sprint B |
| Q4 | Grad-CAM target layer path | ✅ RESOLVED — `model.features[8]` locked from torchvision introspection | Sprint C |
| Q5 | Mel bin-to-Hz mapping validation | ✅ RESOLVED — `librosa.mel_frequencies` mapping contract locked | Sprint C |
| Q6 | Gradio vs Streamlit framework lock | ✅ RESOLVED — Gradio 4.x locked | Sprint E |
| Q7 | EER scoring protocol (scikit vs ASVspoof) | 🔴 OPEN | Phase 5 |

### Phase 0 assessments (Chain 01)

**Q3 — VRAM (Colab T4, batch_size=16):** Training EfficientNet-B4 at `batch_size: 16` is estimated at **~8 GB VRAM**. Colab Free Tier **T4 ≈ 15 GB** — feasible **without** gradient checkpointing for this batch size. `config.yaml` keeps `training.gradient_checkpointing: true` and `mixed_precision: true` as **precaution** (NFR-Reliability). **Exit Sprint B:** empirical step-one training run must confirm headroom.

**Q4 — Grad-CAM layer:** In `torchvision` EfficientNet-B4, **`model.features[-1]`** (final MBConv block in the `Sequential` features) is the **candidate** target for pytorch-grad-cam (**FR-CV-010**). **Exit Sprint C:** confirm exact module path via PyTorch introspection + Grad-CAM smoke test; update `gradcam.target_layer` in `config.yaml` if the graph differs.

---

## 🗂️ FILES CREATED THIS PROJECT

| Artifact | Path |
|----------|------|
| Risk register | `docs/adr/phase0-risk-register.md` |
| Pipeline diagram | `docs/adr/phase0-pipeline-diagram.md` |
| ADR-0001 MCP selection | `docs/adr/phase0-mcp-selection.md` |
| MIP draft | `docs/adr/phase0-mip.md` |
| Phase 1 backlog | `docs/adr/phase1-backlog.md` |
| Phase 1 RTM | `docs/adr/phase1-rtm.md` |
| Q4 ADR | `docs/adr/phase2-gradcam-target-layer.md` |
| Q5 ADR | `docs/adr/phase2-mel-band-mapping.md` |
| Q6 ADR | `docs/adr/phase2-ui-framework.md` |
| Phase 2 interface contracts | `docs/adr/phase2-interface-contracts.md` |
| Phase 3 Q3 VRAM result | `docs/adr/phase3-colab-vram.md` |

---

## 🏗️ PIPELINE CONTRACT (Reference — Do Not Change)

| Stage | Module | Input | Output | Latency |
|-------|--------|-------|--------|---------|
| Audio DSP | src/audio/dsp.py | WAV/FLAC file | [3,224,224] float32 tensor | ≤ 500 ms |
| CV Inference | src/cv/infer.py | [3,224,224] float32 tensor | (label, confidence float) | ≤ 1,500 ms (ONNX) |
| XAI Grad-CAM | src/cv/gradcam.py | [3,224,224] tensor + model | heatmap PNG + band_pct[4] | ≤ 3,000 ms |
| NLP Explanation | src/nlp/explain.py | label + confidence + band_pct[4] | English paragraph (3–5 sentences) | ≤ 8,000 ms |

---

## 💡 TECH DEBT LOG

*(populate with [TECH DEBT: description | SRS-ref: FR-###] items as they arise)*

---

## 📝 SESSION HISTORY

### Session 00 — Project Bootstrap
**Date:** [DATE]
**Status:** ✅ COMPLETE — Setup only
**Actions:** Created .cursorrules, config.yaml, requirements.txt, session-cheatsheet.md
**Next:** Run Chain 01 in Cursor Agent Mode

### Session 01 — Architecture approval
**Date:** 2026-03-22
**Status:** ✅ COMPLETE — Human approved recommended architecture (modular `src/`, ONNX on HF Spaces CPU, phase gates Q3–Q7)
**Actions:** Session init (EXPLORE/DECIDE); cheatsheet updated
**Next:** Phase 0.1 documentation subtask (ADR + diagram in `docs/`)

### Session 02 — Chain 01 Phase 0 inception
**Date:** 2026-03-22
**Status:** ✅ COMPLETE — Documentation-only chain
**Actions:** Created `docs/adr/phase0-risk-register.md`, `phase0-pipeline-diagram.md`, `phase0-mcp-selection.md` (ADR-0001), `phase0-mip.md`; updated Q3/Q4 assessments; Phase 0 gate closed
**Next:** Chain 02 — Phase 1 Requirements & Backlog

### Session 03 — Chain 02 Phase 1 backlog
**Date:** 2026-03-22
**Status:** ✅ COMPLETE — Documentation-only chain
**Actions:** Created `docs/adr/phase1-backlog.md`, `docs/adr/phase1-rtm.md`; Q6 recommendation recorded (Gradio 4.x — still OPEN until Phase 2 gate)
**Next:** Chain 03 — Phase 2 System Design

### Session 04 — Chain 03 Phase 2 specification lock
**Date:** 2026-03-22
**Status:** ✅ COMPLETE — Documentation + config lock chain
**Actions:** Created `docs/adr/phase2-gradcam-target-layer.md`, `phase2-mel-band-mapping.md`, `phase2-ui-framework.md`, `phase2-interface-contracts.md`; locked `config.yaml` (`gradcam.target_layer: model.features[8]`); resolved Q4/Q5/Q6
**Next:** Chain 04 — Phase 3 Environment Setup & MCP Configuration

### Session 05 — Chain 04 Phase 3 scaffold (Q3 pending)
**Date:** 2026-03-22
**Status:** 🟡 PARTIAL — notebook/README/stubs/errors scaffolded; Q3 needs Colab Cell 4 measurement
**Actions:** Created `notebooks/dsdba_training.ipynb`, updated `README.md`, scaffolded module docstrings for `src/*`, implemented `src/utils/errors.py`, created pending `docs/adr/phase3-colab-vram.md`
**Next:** Provide peak VRAM from Cell 4; I will lock `config.yaml -> training.gradient_checkpointing` and mark Q3 resolved