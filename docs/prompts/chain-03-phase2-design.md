## CHAIN 03 — PHASE 2: SYSTEM DESIGN & TECHNICAL SPECIFICATION
> **Mode:** Cursor Composer  
> **Must Close:** Q4, Q5, Q6 (before this chain ends)  
> **Produces:** Module interface contracts, config.yaml confirmed, ADRs for Q4/Q5/Q6

```
# ════════════════════════════════════════════════════════════════════════
# DSDBA Chain 03 — Phase 2: System Design & Technical Specification
# SRS: DSDBA-SRS-2026-002 v2.1 | Phase: 2 | Mode: Composer
# GATE: Q4, Q5, Q6 MUST be resolved before this chain closes
# ════════════════════════════════════════════════════════════════════════

@file .cursorrules
@file config.yaml
@file docs/context/session-cheatsheet.md
@file docs/adr/phase1-backlog.md

[S] SITUATION: Backlog is prioritised. Now specifying exact technical contracts
    for all module interfaces and locking the three design decisions (Q4/Q5/Q6)
    that block downstream sprints.

[C] CHALLENGE: 
    (a) Write complete interface contracts for all 5 pipeline modules
    (b) Resolve Q4: confirm Grad-CAM target layer attribute path in EfficientNet-B4
    (c) Resolve Q5: validate Mel filter bank frequency-to-bin mapping logic
    (d) Resolve Q6: lock Gradio as UI framework
    (e) Create all mandatory ADR files
    (f) Confirm and finalise config.yaml (no changes allowed after this phase)

[F] FOUNDATIONS: All decisions made here are locked — downstream sprints implement
    them, not redesign them. Treat this as a binding technical contract.

TASK 1: Resolve Q4 — Grad-CAM Target Layer
  Perform PyTorch model introspection to confirm the exact attribute path.
  In Cursor terminal, run:
    python -c "
    import torchvision
    model = torchvision.models.efficientnet_b4(weights=None)
    print(type(model.features[-1]))
    print(model.features[-1])
    for name, module in model.named_modules():
        if 'features.8' in name or 'features.-1' in name:
            print(name, type(module))
    "
  Document result in docs/adr/phase2-gradcam-target-layer.md.
  Update config.yaml gradcam.target_layer with confirmed value.
  Update session-cheatsheet.md: Q4 → ✅ RESOLVED — [confirmed layer path].

TASK 2: Resolve Q5 — Mel Filter Bank Frequency-to-Bin Mapping
  Create docs/adr/phase2-mel-band-mapping.md with:
  - Implementation: use librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)
    to get Hz for each of the 128 Mel bins, then compute row ranges for each of the
    4 bands in config.yaml gradcam.band_hz (NOT a naive linear slice of 128 rows)
  - Validation plan: test on a known spectrogram and verify band boundaries match
    expected Hz ranges
  - Code sketch of the mapping function (pure pseudocode, not final code):
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)
    band_rows = {band_name: where(mel_freqs in [band_low, band_high])}
  Update session-cheatsheet.md: Q5 → ✅ RESOLVED — Mel filter bank mapping confirmed.

TASK 3: Resolve Q6 — UI Framework Lock
  Create docs/adr/phase2-ui-framework.md:
  - Decision: Gradio 4.x
  - Rationale: native gr.Audio upload widget, gr.Image for heatmap, streaming support
    for async NLP display (FR-NLP-006), simpler HF Spaces deployment (FR-DEP-001)
  - Alternatives rejected: Streamlit (less native audio support, no streaming components)
  - SRS References: FR-DEP-001, FR-DEP-002, FR-DEP-003, FR-DEP-006, FR-NLP-006
  Update config.yaml: deployment.framework = gradio (already set, now officially locked).
  Update session-cheatsheet.md: Q6 → ✅ RESOLVED — Gradio 4.x locked.

TASK 4: Module Interface Contracts
  Create docs/adr/phase2-interface-contracts.md with exact signatures:

  dsp.py: preprocess_audio(file_path: Path) -> torch.Tensor
    Input:  Path to WAV or FLAC file
    Output: torch.Tensor shape=[3, 224, 224], dtype=torch.float32
    Raises: ValueError("AUD-001") if duration < 0.5 s [FR-AUD-005]
    Raises: ValueError("AUD-002") if format unsupported [FR-AUD-001]
    Latency target: ≤ 500 ms on CPU [NFR-Performance]

  infer.py: run_inference(tensor: torch.Tensor) -> tuple[str, float]
    Input:  torch.Tensor shape=[3, 224, 224], dtype=torch.float32
    Output: (label: "bonafide"|"spoof", confidence: float in (0,1))
    Latency target: ≤ 1,500 ms on CPU via ONNX Runtime [FR-DEP-010]

  gradcam.py: run_gradcam(tensor: torch.Tensor, model) -> tuple[Path, dict[str, float]]
    Input:  tensor [3,224,224] float32, trained EfficientNet-B4 model
    Output: (heatmap_png_path: Path, band_attributions: dict with 4 keys summing to 100.0)
    Latency target: ≤ 3,000 ms on CPU [FR-CV-015]

  explain.py: generate_explanation(label: str, confidence: float, band_pct: dict) -> str
    Input:  label, confidence score, 4-band attribution dict (values sum to 100.0)
    Output: English explanation paragraph (3–5 sentences) [FR-NLP-004]
    Fallback: rule-based template if Qwen 2.5 API fails [FR-NLP-003]
    Latency target: ≤ 8,000 ms (API) or ≤ 100 ms (fallback) [NFR-Performance]

TASK 5: Finalise config.yaml
  Using context7-mcp, verify that all config.yaml parameter names match the exact
  API parameter names in librosa 0.10.x and pytorch-grad-cam 1.5.0.
  Make any corrections. After this task, config.yaml is LOCKED for the project.

Update session-cheatsheet.md: Phase 2 complete, Q4/Q5/Q6 all resolved.
```
