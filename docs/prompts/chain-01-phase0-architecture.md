---

## CHAIN 01 — PHASE 0: PROJECT INCEPTION & ARCHITECTURE
> **Mode:** Cursor Agent Mode  
> **Blocking Q:** None (first chain)  
> **Produces:** Pipeline diagram, ADR-0001, architectural risk register, MIP draft

```
# ════════════════════════════════════════════════════════════════════════
# DSDBA Chain 01 — Phase 0: Project Inception & Architecture Design
# SRS: DSDBA-SRS-2026-002 v2.1 | Phase: 0 | Mode: Agent
# ════════════════════════════════════════════════════════════════════════

@file .cursorrules
@file config.yaml
@file docs/context/session-cheatsheet.md

[S] SITUATION: Starting Phase 0 of the DSDBA project. The SRS v2.1 defines a
    4-stage sequential multimodal pipeline: Audio DSP → CV Inference → XAI →
    NLP Explanation. No code exists yet. This chain produces the architectural
    foundation that all subsequent chains depend on.

[C] CHALLENGE: Produce the complete Phase 0 deliverables as defined in the
    DSDBA Master Plan. Specifically:
    (a) Identify and document top 3 architectural risks with SRS coupling points
    (b) Produce a Mermaid pipeline architecture diagram
    (c) Create ADR-0001 for MCP tool selection rationale
    (d) Draft the Master Implementation Plan (MIP) checklist for Phases 0–7
    (e) Provide initial assessment of Open Questions Q3 and Q4

[A] AUDIENCE: Ferel and Safa (ITS Informatics), future KCVanguard reviewers,
    and the Cursor AI that will implement this project chain-by-chain.

[F] FORMAT: Markdown documentation. Mermaid diagrams. ADR template from
    .cursorrules. All output saved to docs/ folder. IEEE 830-aligned language.

[F] FOUNDATIONS: Pipeline is strictly feed-forward (Section 0.1 SRS v2.1).
    Critical coupling points: tensor shape contract [3,224,224], Grad-CAM layer
    identity, Mel filter bank mapping, Qwen 2.5 async isolation (Section 0.2).

TASKS — execute in this order:

TASK 1: Architectural Risk Register
  Create docs/adr/phase0-risk-register.md with these 3 risks:
  Risk 1: Tensor shape contract violation [3,224,224] → silent downstream corruption
    - Coupling point: FR-AUD-008 → FR-CV-001 interface
    - Mitigation: validate shape at CV module entry; assertion in test_cv.py
  Risk 2: Grad-CAM target layer misidentification → wrong saliency (Q4)
    - Coupling point: FR-CV-010 → model.features[-1] must be confirmed via PyTorch introspection
    - Mitigation: lock in config.yaml; verify via jacobgil pytorch-grad-cam tests
  Risk 3: Qwen 2.5 API latency blocks CV result display
    - Coupling point: FR-NLP-006 → UI must show Stage 2 before Stage 3 completes
    - Mitigation: asyncio.Task isolation; rule-based fallback always ready (FR-NLP-003)

TASK 2: Pipeline Architecture Diagram
  Create docs/adr/phase0-pipeline-diagram.md containing a Mermaid flowchart showing:
  - Stage 1: WAV/FLAC → librosa DSP → [3,224,224] float32 tensor (FR-AUD-001–008)
  - Stage 2: tensor → EfficientNet-B4 ONNX → (label, confidence) (FR-CV-001–009)
  - Stage 3: tensor + model → Grad-CAM → heatmap PNG + band_pct[4] (FR-CV-010–016)
  - Stage 4: label + confidence + band_pct[4] → Qwen 2.5 async → English paragraph (FR-NLP-001–006)
  - Stage 5: all results → Gradio UI → user display (FR-DEP-001–010)
  Label each arrow with the exact data type crossing the boundary.

TASK 3: MCP Tool ADR
  Create docs/adr/phase0-mcp-selection.md using the ADR template:
  - Decision: context7-mcp for library docs, huggingface-mcp for Hub/Spaces, stitch-mcp for orchestration
  - Alternatives rejected: manual documentation lookup, direct API calls without orchestration
  - Rationale: each MCP maps to a specific pipeline stage and dependency type
  - SRS Reference: all FRs across Audio/CV/NLP/Deploy modules

TASK 4: Open Questions Q3 and Q4 Initial Assessment
  Add to docs/context/session-cheatsheet.md:
  Q3 Assessment: EfficientNet-B4 with batch_size=16 on Colab T4 requires ~8 GB VRAM.
    Colab Free Tier T4 has 15 GB. Feasible WITHOUT gradient checkpointing for batch=16.
    Gradient checkpointing in config.yaml set to true as precaution. Status: 🟡 LIKELY FEASIBLE
  Q4 Assessment: In torchvision EfficientNet-B4, the final MBConv block is accessible as
    model.features[-1] (EfficientNet features Sequential, last element).
    Requires PyTorch introspection in Sprint C to confirm exact path.
    Status: 🟡 CANDIDATE CONFIRMED — needs empirical validation

TASK 5: Master Implementation Plan
  Create docs/adr/phase0-mip.md with a checkbox list of ALL phases and sprints,
  listing the SRS FR IDs owned by each phase and the blocking Open Question if any.

After all tasks, update docs/context/session-cheatsheet.md:
  - Current Phase: 0 — COMPLETE
  - Next Action: Run Chain 02 (Phase 1 Backlog)
  - Q3 and Q4 status updated per Task 4 assessment above
```

---