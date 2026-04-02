# DSDBA — Phase 0 Pipeline Architecture Diagram

**Document:** DSDBA-SRS-2026-002 v2.1  
**SDLC Phase:** 0 — Project Inception & Architecture Design  
**SRS Traceability:** FR-AUD-001–008, FR-CV-001–009, FR-CV-010–016, FR-NLP-001–006, FR-DEP-001–010  
**Label:** [Phase 0 | v1 | SRS-ref: multimodal pipeline]

---

## Feed-forward contract (Section 0.1 SRS)

The pipeline is **strictly sequential**: no feedback from NLP to CV, and no disk persistence of raw audio in production paths (NFR-Security).

---

## Mermaid — end-to-end data flow

Boundary labels use **exact conceptual types** crossing each interface.

```mermaid
flowchart TB
  subgraph S1["Stage 1 — Audio DSP (FR-AUD-001–008)"]
    W["WAV/FLAC bytes\n(pathlib / upload)"]
    DSP["librosa STFT → Mel (128) → resize/normalise\nsrc/audio/dsp.py"]
    T1["torch.Tensor [3,224,224] float32\nCPU"]
    W --> DSP --> T1
  end

  subgraph S2["Stage 2 — CV Inference (FR-CV-001–009)"]
    ONNX["ONNX Runtime EfficientNet-B4\nsrc/cv/infer.py"]
    OUT2["tuple: label int ∈ {0,1}\nconfidence float ∈ [0,1]"]
    T1 -->|"torch.Tensor [3,224,224] float32"| ONNX
    ONNX --> OUT2
  end

  subgraph S3["Stage 3 — XAI Grad-CAM (FR-CV-010–016)"]
    GC["pytorch-grad-cam + band attribution\nsrc/cv/gradcam.py"]
    OUT3["tuple: heatmap PNG bytes\nband_pct float[4] Softmax-normalised"]
    T1 -->|"same tensor + model handle"| GC
    GC --> OUT3
  end

  subgraph S4["Stage 4 — NLP (FR-NLP-001–006)"]
    N["Qwen 2.5 async + rule fallback\nsrc/nlp/explain.py"]
    OUT4["str English paragraph\n(3–5 sentences)"]
    OUT2 --> N
    OUT3 --> N
    N --> OUT4
  end

  subgraph S5["Stage 5 — Deployment UI (FR-DEP-001–010)"]
    UI["Gradio UI\napp.py"]
    DISP["structured JSON + components\nto browser"]
    OUT2 --> UI
    OUT3 --> UI
    OUT4 --> UI
    UI --> DISP
  end
```

---

## Arrow reference (boundary types)

| From → To | Payload | SRS |
|-----------|---------|-----|
| Upload → DSP | `bytes` / file-like, ≤ 20 MB | FR-DEP-002, FR-AUD-001 |
| DSP → CV | `Tensor [3,224,224] float32` | FR-AUD-008 → FR-CV-001 |
| CV → UI | `(label: int, confidence: float)` | FR-CV-004 |
| DSP + model → Grad-CAM | `Tensor` + `nn.Module` | FR-CV-010 |
| Grad-CAM → NLP | `band_pct: float[4]` + heatmap | FR-CV-014, FR-NLP-001 |
| NLP → UI | `str` explanation | FR-NLP-001 |
| CV → UI (ordering) | CV panel **before** NLP completes | FR-NLP-006 |

---

## Notes

- **ONNX** is mandatory for CPU inference on Hugging Face Spaces (**FR-DEP-010**); training remains PyTorch (**FR-CV-007**).
- **Q4** and **Q5** remain open until Sprint C; diagram assumes `config.yaml` locks until validated.
