## CHAIN 04 — PHASE 3: ENVIRONMENT SETUP & MCP CONFIGURATION
> **Mode:** Cursor Agent Mode  
> **Must Close:** Q3 (EfficientNet-B4 VRAM feasibility)  
> **Produces:** Colab notebook scaffold, GitHub repo setup, MCP verification, Q3 resolution

```
# ════════════════════════════════════════════════════════════════════════
# DSDBA Chain 04 — Phase 3: Environment Setup & MCP Configuration
# SRS: DSDBA-SRS-2026-002 v2.1 | Phase: 3 | Mode: Agent
# GATE: Q3 MUST be resolved before this chain closes
# ════════════════════════════════════════════════════════════════════════

@file .cursorrules
@file config.yaml
@file docs/context/session-cheatsheet.md
@file docs/adr/phase2-interface-contracts.md

[C] CHALLENGE:
    (a) Create the Google Colab training notebook scaffold
    (b) Verify all 3 MCP servers connect correctly
    (c) Set up GitHub repository with correct README
    (d) Resolve Q3: empirically verify EfficientNet-B4 VRAM on Colab T4
    (e) Scaffold all empty src/ stub files with correct module docstrings

TASK 1: Colab Training Notebook Scaffold
  Create notebooks/dsdba_training.ipynb with cells (stubs only, no training logic yet):
  Cell 1 — Environment setup: pip install commands for requirements.txt pinned versions
  Cell 2 — GPU verification: assert torch.cuda.is_available(), print VRAM
  Cell 3 — Q3 VRAM Test: create a dummy EfficientNet-B4, run 1 forward pass with
    batch_size=16, dummy tensor [16, 3, 224, 224], measure peak VRAM usage via
    torch.cuda.max_memory_allocated(). If > 11 GB → set gradient_checkpointing=true.
  Cell 4 — HuggingFace login: using huggingface-skills MCP (mcp.json: "huggingface-skills"), authenticate and verify FoR dataset access
  Cell 5 — Dataset download: FoR for-2sec split via huggingface-mcp datasets API
  Cell 6 — (placeholder) Training loop — populated in Chain 06

TASK 2: GitHub Repository Setup
  Create README.md with:
  - Project title: DSDBA — Deepfake Speech Detection & Biometric Authentication System
  - Pipeline diagram (copy from docs/adr/phase0-pipeline-diagram.md)
  - Installation instructions using requirements.txt
  - Training instructions (Colab notebook link)
  - Demo instructions (HF Spaces link placeholder)
  - Dataset citation: Abdel-Dayem, M. (2023). Fake-or-Real (FoR) Dataset. Kaggle.
  - Architecture: EfficientNet-B4 + Grad-CAM XAI + Qwen 2.5 NLP
  - Team credits: Ferel, Safa — ITS Informatics, KCVanguard ML Workshop

TASK 3: Stub All Source Files with Module Docstrings
  For each of these files, generate ONLY the module docstring (no implementation yet):
  - src/audio/dsp.py        → SRS FR-AUD-001–011, Sprint A
  - src/cv/model.py         → SRS FR-CV-001–002, Sprint B
  - src/cv/train.py         → SRS FR-CV-003–008, Sprint B
  - src/cv/infer.py         → SRS FR-DEP-010, Sprint B
  - src/cv/gradcam.py       → SRS FR-CV-010–016, Sprint C
  - src/nlp/explain.py      → SRS FR-NLP-001–009, Sprint D
  - src/utils/config.py     → config.yaml Pydantic loader
  - src/utils/logger.py     → structured JSON logger
  - src/utils/errors.py     → SRS error codes AUD-001, AUD-002
  - app.py                  → SRS FR-DEP-001–010, Sprint E
  Use the module docstring template from .cursorrules exactly.

TASK 4: Resolve Q3
  Record the result from Colab Cell 3 in docs/adr/phase3-colab-vram.md:
  - Actual peak VRAM for batch_size=16 with EfficientNet-B4: [X] GB
  - Decision: gradient_checkpointing = [true/false based on result]
  - config.yaml training.batch_size and gradient_checkpointing confirmed
  Update session-cheatsheet.md: Q3 → ✅ RESOLVED — [actual VRAM value].

TASK 5: src/utils/errors.py Implementation
  Implement this file fully (it has no dependencies):
  Define a Python dataclass or Enum for SRS error codes:
    AUD_001 = "AUD-001"  # Audio < 0.5 s [FR-AUD-005]
    AUD_002 = "AUD-002"  # Unsupported format [FR-AUD-001]
  Define DSDBAError(Exception) with fields: code, message, stage

Update session-cheatsheet.md: Phase 3 complete, Q3 resolved, all stubs created.
```

---