# ADR-0001 — MCP Tool Selection for DSDBA Development

**Status:** Accepted  
**Date:** 2026-03-22  
**Deciders:** Ferel, Safa (ITS Informatics)  
**Technical Story:** DSDBA-SRS-2026-002 v2.1 — Phase 0 architecture  
**SRS Reference:** Cross-cutting — FR-AUD, FR-CV, FR-NLP, FR-DEP, NFR-Maintainability  
**Label:** [Phase 0 | v1 | ADR-0001]

---

## Context

DSDBA requires accurate **library API usage** (librosa, PyTorch, ONNX Runtime, pytorch-grad-cam, Gradio), **Hugging Face Hub / Spaces** integration, and **async orchestration** for Qwen 2.5 and fallback chains. Cursor-based development must reduce hallucinated APIs and keep deployment constraints (2 vCPU CPU-only) visible.

---

## Decision

Adopt three MCP servers for the Cursor workflow:

| MCP | Primary use | Pipeline mapping |
|-----|-------------|------------------|
| **context7-mcp** | Up-to-date docs and examples for PyTorch, librosa, pytorch-grad-cam, Gradio, ONNX Runtime | Stages: Audio DSP, CV, XAI, Deployment |
| **huggingface-mcp** | Model Hub artefacts, dataset (FoR), Spaces deployment patterns | Stages: CV (checkpoints), data, FR-DEP |
| **stitch-mcp** | Multi-stage orchestration, Qwen 2.5 async workflows, fallback chains | Stages: NLP, end-to-end glue |

---

## Alternatives considered

| Alternative | Rejected because |
|-------------|-------------------|
| Manual browser documentation lookup | Inconsistent; violates D.O.C.S. traceability and slows Agent cycles. |
| Direct REST calls to Qwen without orchestration layer | Harder to test **FR-NLP-003** / **FR-NLP-006** (timeout + UI ordering). |
| Single “generic” MCP for all libraries | Does not map to dependency types (docs vs Hub vs async orchestration). |

---

## Consequences

**Positive:**

- Clear **one-tool-per-concern** mapping for Cursor prompts and Chain instructions.
- Easier verification of **NFR-Performance** and **FR-DEP-010** against official ONNX Runtime and Gradio docs.

**Negative:**

- Three MCP endpoints must remain configured and authenticated where required (HF).
- Developers must **name the MCP** in each chain (S.C.A.F.F. / governance).

---

## Compliance

- **S.H.I.E.L.D.:** API keys remain in HF Spaces secrets only (**FR-NLP-005**); MCP usage does not imply secrets in repo.
- **SRS:** All functional modules remain traceable to FRs listed in **Master Implementation Plan** (`docs/adr/phase0-mip.md`).

---

## [EXPLORE] → [DECISION]

**Option A:** Docs-only + manual HF. **Option B:** context7 + huggingface + stitch as above.  
**Selected: B** — aligns with sequential pipeline dependencies and reduces API drift in Cursor-generated code.
