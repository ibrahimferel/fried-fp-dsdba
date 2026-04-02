# ADR — Phase 2 Q6 Resolution: UI Framework Lock

**Document:** DSDBA-SRS-2026-002 v2.1  
**Phase:** 2 — System Design & Technical Specification  
**SRS refs:** FR-DEP-001, FR-DEP-002, FR-DEP-003, FR-DEP-006, FR-NLP-006  
**Label:** [Phase 2 | v1 | Q6-RESOLVED]

## Decision

Q6 is resolved: **Gradio 4.x** is the locked deployment UI framework for DSDBA.

## Rationale

- Native `gr.Audio` upload/playback supports audio-first workflow (FR-DEP-001/002).
- Native image components support Grad-CAM display (FR-DEP-005/006).
- Async-friendly interaction model supports showing CV result before NLP finishes (FR-NLP-006).
- Hugging Face Spaces deployment path is direct (`app.py` + `requirements.txt`) with minimal operational overhead.

## Alternatives rejected

- **Streamlit 1.x**: acceptable for dashboards, but less direct for staged async UX and audio-first interaction in this project contract.

## Consequences

- `config.yaml` `deployment.framework` is now binding: `gradio`.
- Sprint E will implement only Gradio path; no dual-framework abstraction is planned.
