## CHAIN 02 — PHASE 1: REQUIREMENTS & BACKLOG DEFINITION
> **Mode:** Cursor Composer  
> **Blocking Q:** None  
> **Produces:** Prioritised backlog, SHALL/SHOULD/MAY breakdown, traceability matrix

```
# ════════════════════════════════════════════════════════════════════════
# DSDBA Chain 02 — Phase 1: Requirements & Backlog Definition
# SRS: DSDBA-SRS-2026-002 v2.1 | Phase: 1 | Mode: Composer
# ════════════════════════════════════════════════════════════════════════

@file .cursorrules
@file config.yaml
@file docs/context/session-cheatsheet.md

[S] SITUATION: Phase 0 architecture is complete. Now decomposing all SRS v2.1
    functional requirements into an implementation backlog, prioritised by
    SHALL/SHOULD/MAY and ordered by pipeline dependency.

[C] CHALLENGE: Produce Phase 1 deliverables:
    (a) Complete FR backlog decomposed from all 4 SRS sections (AUD/CV/NLP/DEP)
    (b) Priority classification (SHALL = Sprint Critical, SHOULD = Sprint Target, MAY = If Time)
    (c) Full Requirement Traceability Matrix (RTM): FR-ID → module → sprint → blocking Q
    (d) Identify SHOULD and MAY requirements that could be deferred without SRS violation

[F] FORMAT: Markdown tables. One RTM table per pipeline stage. Save to docs/.

TASK 1: Prioritised Backlog by Sprint
  Create docs/adr/phase1-backlog.md:

  SPRINT A — Audio DSP (src/audio/dsp.py):
  SHALL: FR-AUD-001, 002, 003, 004, 005, 006, 007, 008
  SHOULD: FR-AUD-009 (batch API), FR-AUD-010 (JSON logging)
  MAY: FR-AUD-011 (MP3/OGG formats)

  SPRINT B — CV Training + ONNX (src/cv/):
  SHALL: FR-CV-001, 002, 003, 004, 005, 007, 008 + FR-DEP-010
  SHOULD: FR-CV-006 (SpecAugment)
  MAY: FR-CV-009 (EfficientNet-B0 baseline)
  BLOCKER: Q3 must be resolved before this sprint starts

  SPRINT C — XAI / Grad-CAM (src/cv/gradcam.py):
  SHALL: FR-CV-010, 011, 012, 013, 014, 015
  SHOULD: FR-CV-016 (raw saliency JSON endpoint)
  BLOCKER: Q4 AND Q5 must be resolved before this sprint starts

  SPRINT D — NLP / Qwen 2.5 (src/nlp/explain.py):
  SHALL: FR-NLP-001, 002, 003, 004, 005, 006
  SHOULD: FR-NLP-007 (Gemma-3 fallback), FR-NLP-008 (caching)
  MAY: FR-NLP-009 (language toggle)

  SPRINT E — Deployment UI (app.py):
  SHALL: FR-DEP-001, 002, 003, 004, 005, 006, 007
  SHOULD: FR-DEP-008 (demo samples), FR-DEP-009 (About section)
  BLOCKER: Q6 must be resolved before this sprint starts

TASK 2: Requirement Traceability Matrix
  Create docs/adr/phase1-rtm.md with a full table:
  Columns: FR-ID | Description | Priority | Sprint | Module File | Blocking Q | Status
  All 37 functional requirements must appear (FR-AUD-001–011, FR-CV-001–016,
  FR-NLP-001–009, FR-DEP-001–010).

TASK 3: Q6 Decision Recommendation
  In docs/adr/phase1-backlog.md, add a section recommending Gradio 4.x for Sprint E
  based on: native audio widget support, simpler deployment to HF Spaces, better
  component streaming API for async NLP display (FR-NLP-006, FR-DEP-006).
  This is a recommendation — Q6 is not officially closed until Phase 2 gate.

Update session-cheatsheet.md with Phase 1 complete status.
```

---
