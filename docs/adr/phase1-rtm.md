# DSDBA — Requirement Traceability Matrix (RTM)

**Document:** DSDBA-SRS-2026-002 v2.1  
**SDLC Phase:** 1 — Requirements & Backlog Definition  
**Label:** [Phase 1 | v1 | RTM | SRS-ref: all FR buckets]

**Scope:** One row per enumerated FR-ID in ranges **FR-AUD-001–011**, **FR-CV-001–016**, **FR-NLP-001–009**, **FR-DEP-001–010** (**46** IDs). Descriptions are aligned with `config.yaml` and pipeline contracts.

---

## Stage 1 — Audio DSP (FR-AUD)

| FR-ID | Description (concise) | Priority | Sprint | Module file(s) | Blocking Q | Status |
|-------|------------------------|----------|--------|----------------|--------------|--------|
| FR-AUD-001 | Accept SHALL-level formats (WAV/FLAC); reject unsupported with AUD-002 | SHALL | A | `src/audio/dsp.py` | — | Pending |
| FR-AUD-002 | Resample to target sample rate (16 kHz) | SHALL | A | `src/audio/dsp.py` | — | Pending |
| FR-AUD-003 | Multi-channel → mono (mean) before processing | SHALL | A | `src/audio/dsp.py` | — | Pending |
| FR-AUD-004 | Fixed clip duration 2.0 s; centre-crop / pad per edge rules | SHALL | A | `src/audio/dsp.py` | — | Pending |
| FR-AUD-005 | Reject clips &lt; min duration; surface AUD-001 JSON | SHALL | A | `src/audio/dsp.py` | — | Pending |
| FR-AUD-006 | STFT → Mel (128) with configured n_fft, hop, window | SHALL | A | `src/audio/dsp.py` | — | Pending |
| FR-AUD-007 | Numerical pipeline (normalise, resize) to model input grid | SHALL | A | `src/audio/dsp.py` | — | Pending |
| FR-AUD-008 | Output `torch.Tensor [3,224,224] float32` | SHALL | A | `src/audio/dsp.py` | — | Pending |
| FR-AUD-009 | Batch / multi-file DSP API | SHOULD | A | `src/audio/dsp.py` | — | Pending |
| FR-AUD-010 | Structured JSON logging of DSP stages | SHOULD | A | `src/audio/dsp.py`, `src/utils/logger.py` | — | Pending |
| FR-AUD-011 | MAY formats MP3/OGG via torchaudio | MAY | A | `src/audio/dsp.py` | — | Pending |

---

## Stage 2 — Computer Vision & XAI (FR-CV)

| FR-ID | Description (concise) | Priority | Sprint | Module file(s) | Blocking Q | Status |
|-------|------------------------|----------|--------|----------------|--------------|--------|
| FR-CV-001 | EfficientNet-B4 backbone; ImageNet-pretrained | SHALL | B | `src/cv/model.py` | Q3 | Pending |
| FR-CV-002 | Binary head; bonafide/spoof logits | SHALL | B | `src/cv/model.py` | Q3 | Pending |
| FR-CV-003 | Training schedule (frozen → fine-tune); optimiser | SHALL | B | `src/cv/train.py` | Q3 | Pending |
| FR-CV-004 | Sigmoid + threshold; calibrated prediction | SHALL | B | `src/cv/infer.py` | Q3 | Pending |
| FR-CV-005 | Weighted BCE / class imbalance handling | SHALL | B | `src/cv/train.py` | Q3 | Pending |
| FR-CV-006 | SpecAugment + augmentation limits | SHOULD | B | `src/cv/train.py` | Q3 | Pending |
| FR-CV-007 | Checkpointing; HF Hub push/pull | SHALL | B | `src/cv/train.py` | Q3 | Pending |
| FR-CV-008 | FoR test metrics: EER/AUC-ROC vs acceptance criteria | SHALL | B | `src/cv/train.py`, `tests/` | Q7 | Pending |
| FR-CV-009 | EfficientNet-B0 baseline ablation | MAY | B | `src/cv/model.py` | Q3 | Pending |
| FR-CV-010 | Grad-CAM target layer locked in config | SHALL | C | `src/cv/gradcam.py` | Q4 | Pending |
| FR-CV-011 | pytorch-grad-cam integration | SHALL | C | `src/cv/gradcam.py` | Q4 | Pending |
| FR-CV-012 | Heatmap PNG + jet overlay | SHALL | C | `src/cv/gradcam.py` | Q4 | Pending |
| FR-CV-013 | 4-band Hz attribution (Mel ↔ Hz) | SHALL | C | `src/cv/gradcam.py` | Q4, Q5 | Pending |
| FR-CV-014 | Softmax band weights; sum = 100% ± ε | SHALL | C | `src/cv/gradcam.py` | Q5 | Pending |
| FR-CV-015 | Grad-CAM CPU latency ≤ 3,000 ms | SHALL | C | `src/cv/gradcam.py` | Q4 | Pending |
| FR-CV-016 | Raw saliency JSON exposure (dev) | SHOULD | C | `src/cv/gradcam.py` | Q4 | Pending |

---

## Stage 3 — NLP (FR-NLP)

| FR-ID | Description (concise) | Priority | Sprint | Module file(s) | Blocking Q | Status |
|-------|------------------------|----------|--------|----------------|--------------|--------|
| FR-NLP-001 | 3–5 sentence English explanation | SHALL | D | `src/nlp/explain.py` | — | Pending |
| FR-NLP-002 | Qwen 2.5 async API; timeout | SHALL | D | `src/nlp/explain.py` | — | Pending |
| FR-NLP-003 | Rule-based fallback + warning badge on failure | SHALL | D | `src/nlp/explain.py` | — | Pending |
| FR-NLP-004 | English-only output | SHALL | D | `src/nlp/explain.py` | — | Pending |
| FR-NLP-005 | No API keys in repo/logs; Spaces secrets | SHALL | D | `src/nlp/explain.py`, `app.py` | — | Pending |
| FR-NLP-006 | CV panel visible before NLP completes | SHALL | D | `src/nlp/explain.py`, `app.py` | — | Pending |
| FR-NLP-007 | Secondary LLM (e.g. Gemma-3) fallback | SHOULD | D | `src/nlp/explain.py` | — | Pending |
| FR-NLP-008 | Explanation caching | SHOULD | D | `src/nlp/explain.py` | — | Pending |
| FR-NLP-009 | Optional language toggle | MAY | D | `src/nlp/explain.py` | — | Pending |

---

## Stage 4 — Deployment (FR-DEP)

| FR-ID | Description (concise) | Priority | Sprint | Module file(s) | Blocking Q | Status |
|-------|------------------------|----------|--------|----------------|--------------|--------|
| FR-DEP-001 | Web UI framework (Gradio/Streamlit per Q6) | SHALL | E | `app.py` | Q6 | Pending |
| FR-DEP-002 | Max upload size; reject before DSP | SHALL | E | `app.py` | Q6 | Pending |
| FR-DEP-003 | Public access path (no auth) | SHALL | E | `app.py` | Q6 | Pending |
| FR-DEP-004 | Display label + confidence clearly | SHALL | E | `app.py` | Q6 | Pending |
| FR-DEP-005 | Present heatmap + band summary in UI | SHALL | E | `app.py` | Q6 | Pending |
| FR-DEP-006 | Async NLP slot; non-blocking layout | SHALL | E | `app.py` | Q6 | Pending |
| FR-DEP-007 | E2E wall time ≤ 15 s (CPU) | SHALL | E | `app.py`, `tests/` | Q6 | Pending |
| FR-DEP-008 | Bundled demo samples (2+2) | SHOULD | E | `app.py`, `assets/` | Q6 | Pending |
| FR-DEP-009 | About / citation panel | SHOULD | E | `app.py` | Q6 | Pending |
| FR-DEP-010 | ONNX Runtime; \|Δ\| &lt; 1e-5 vs PyTorch; CPU infer ≤ 1,500 ms | SHALL | B (+ E verify) | `src/cv/infer.py`, `app.py`, `tests/` | Q3 | Pending |

**Note:** FR-DEP-010 primary implementation is **Sprint B**; Sprint E **verifies** integration and UI-facing behaviour.

---

## Matrix coverage

| Bucket | Count |
|--------|-------|
| FR-AUD | 11 |
| FR-CV | 16 |
| FR-NLP | 9 |
| FR-DEP | 10 |
| **Total** | **46** |
