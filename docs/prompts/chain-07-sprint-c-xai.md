## CHAIN 07 — SPRINT C: XAI / GRAD-CAM MODULE
> **Mode:** Cursor Agent Mode  
> **Blocking Q:** Q4 AND Q5 MUST be resolved (from Chain 03)  
> **Produces:** `src/cv/gradcam.py` complete + `src/tests/test_gradcam.py`

```
# ════════════════════════════════════════════════════════════════════════
# DSDBA Chain 07 — Sprint C: XAI / Grad-CAM Module
# SRS: FR-CV-010–016 | Phase: 4-C | Mode: Agent
# V.E.R.I.F.Y. Level: 3 | BLOCKERS: Q4 AND Q5 MUST be resolved
# ════════════════════════════════════════════════════════════════════════

@file .cursorrules
@file config.yaml
@file docs/context/session-cheatsheet.md
@file docs/adr/phase2-gradcam-target-layer.md
@file docs/adr/phase2-mel-band-mapping.md
@file src/cv/model.py
@file src/cv/infer.py

CONFIRM BEFORE STARTING:
  Q4 in session-cheatsheet.md shows ✅ RESOLVED? If not → STOP.
  Q5 in session-cheatsheet.md shows ✅ RESOLVED? If not → STOP.

Use context7-mcp to verify pytorch-grad-cam 1.5.0 GradCAM class API.

IMPLEMENT: src/cv/gradcam.py  [FR-CV-010–016]

  1. get_target_layer(model: DSDBAModel, cfg) -> nn.Module:
     - Return the layer at config path cfg.gradcam.target_layer [FR-CV-010, FR-CV-011]
     - Use Python eval() or manual path resolution — document in ADR
     - This layer must match the confirmed Q4 resolution

  2. compute_gradcam(model, tensor: Tensor, cfg) -> np.ndarray:
     - Use pytorch-grad-cam GradCAM(model, [target_layer]) [FR-CV-011]
     - Return raw saliency matrix shape [224, 224], values in [0, 1]
     - Latency: must complete in ≤ 3,000 ms on CPU [FR-CV-015]

  3. create_heatmap_overlay(tensor: Tensor, saliency: np.ndarray, cfg) -> Path:
     - Apply jet colormap, alpha blend at cfg.gradcam.overlay_alpha=0.5 [FR-CV-012]
     - Save as PNG, return Path [FR-CV-012]

  4. get_mel_band_row_indices(cfg) -> dict[str, tuple[int, int]]:
     - Use librosa.mel_frequencies(n_mels=128, fmin=0, fmax=8000) [FR-CV-013, Q5]
     - Map cfg.gradcam.band_hz boundaries to actual Mel bin indices
     - NOT a naive linear slice — uses mel_freqs array [Q5 RESOLVED]
     - Returns: {"low": (r1,r2), "low_mid": (r3,r4), "high_mid": (r5,r6), "high": (r7,r8)}

  5. compute_band_attributions(saliency: np.ndarray, cfg) -> dict[str, float]:
     - Sum saliency values within each band's row range [FR-CV-013]
     - Apply Softmax normalisation → values sum to exactly 100.0 [FR-CV-014]
     - Returns: {"low": float, "low_mid": float, "high_mid": float, "high": float}
     - Assert: sum(values) == 100.0 ± 0.001

  6. run_gradcam(tensor: Tensor, model, cfg) -> tuple[Path, dict[str, float]]:  [MAIN API]
     - Compose steps 1–5
     - Return (heatmap_png_path, band_attributions_dict)
     - Matches interface contract from phase2-interface-contracts.md

  7. get_raw_saliency_json(saliency: np.ndarray) -> dict:  [FR-CV-016 SHOULD]
     - Return saliency matrix as nested list in JSON-serialisable dict

IMPLEMENT: src/tests/test_gradcam.py  [V.E.R.I.F.Y. L3]
  1. test_target_layer_exists: get_target_layer() returns nn.Module without AttributeError
  2. test_saliency_shape: compute_gradcam() returns array shape (224, 224)
  3. test_saliency_range: all values in saliency in [0.0, 1.0]
  4. test_heatmap_png_created: heatmap PNG file written to disk
  5. test_mel_band_mapping_not_linear: verify band row indices use mel_frequencies,
     not evenly spaced [32, 64, 96, 128] (Q5 validation)
  6. test_band_sum_100: sum(band_attributions.values()) == 100.0 ± 0.001 [FR-CV-014]
  7. test_gradcam_latency: run_gradcam() completes in ≤ 3,000 ms on CPU [FR-CV-015]
  8. test_raw_saliency_json_serialisable: result is JSON-serialisable

All 8 tests must pass before Gate Check.
```

---