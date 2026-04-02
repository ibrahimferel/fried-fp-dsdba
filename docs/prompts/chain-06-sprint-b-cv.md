## CHAIN 06 — SPRINT B: CV TRAINING & ONNX EXPORT
> **Mode:** Cursor Agent Mode  
> **Blocking Q:** Q3 MUST be resolved (from Chain 04)  
> **Produces:** `src/cv/model.py`, `src/cv/train.py`, `src/cv/infer.py`, `src/tests/test_cv.py`

```
# ════════════════════════════════════════════════════════════════════════
# DSDBA Chain 06 — Sprint B: CV Training & ONNX Export
# SRS: FR-CV-001–009 + FR-DEP-010 | Phase: 4-B | Mode: Agent
# V.E.R.I.F.Y. Level: 3 | BLOCKER: Q3 MUST be resolved
# ════════════════════════════════════════════════════════════════════════

@file .cursorrules
@file config.yaml
@file docs/context/session-cheatsheet.md
@file docs/adr/phase2-interface-contracts.md
@file docs/adr/phase3-colab-vram.md
@file src/audio/dsp.py

CONFIRM BEFORE STARTING: Is Q3 resolved in session-cheatsheet.md?
If Q3 shows 🔴 OPEN → STOP. Do not generate any code. Return to Chain 04 Task 4.

Use context7-mcp to verify torchvision EfficientNet-B4 API before implementing.
Use huggingface-mcp to configure HF Model Hub upload in train.py.

IMPLEMENT: src/cv/model.py  [FR-CV-001–002]
  class DSDBAModel(nn.Module):
    - Load EfficientNet-B4 pretrained on ImageNet-1k [FR-CV-001]
    - Replace classifier head with linear(in_features, 2) [FR-CV-002]
    - freeze_backbone(): freeze all layers except new head
    - unfreeze_top_n(n: int): unfreeze top N layers for fine-tuning
    - forward(x: Tensor) -> Tensor: return logits shape [batch, 2]

IMPLEMENT: src/cv/train.py  [FR-CV-003–008]
  Functions:
  - get_class_weights(dataset) -> Tensor: inverse frequency weights [FR-CV-005]
  - build_augmentations(cfg) -> transforms: SpecAugment + noise + shift [FR-CV-006]
  - train_epoch(model, loader, optimizer, criterion, cfg) -> dict
  - validate_epoch(model, loader, cfg) -> dict[str, float]: returns EER, AUC-ROC
  - compute_eer(y_true, y_scores) -> float: Equal Error Rate [FR-CV-008, Q7 note]
  - run_training(cfg) -> DSDBAModel:
      Phase 1: freeze backbone, train head 5 epochs [FR-CV-003]
      Phase 2: unfreeze top-N layers, finetune at lr ≤ 1e-4 [FR-CV-003]
      Save best checkpoint (by AUC-ROC) every epoch [FR-CV-007, NFR-Reliability]
      Upload checkpoint to HF Model Hub via huggingface-mcp [FR-CV-007]
      Target: EER ≤ 10%, AUC-ROC ≥ 0.90 [FR-CV-008]

IMPLEMENT: src/cv/infer.py  [FR-DEP-010]
  - export_to_onnx(model: DSDBAModel, cfg) -> Path: export and save .onnx
  - verify_onnx_equivalence(model, onnx_path, cfg) -> bool:
      Run same input through PyTorch and ONNX
      Assert |onnx_out - pytorch_out| < cfg.deployment.onnx_equivalence_tolerance
  - load_onnx_session(onnx_path: Path, cfg) -> ort.InferenceSession:
      Use CPUExecutionProvider only [FR-DEP-010]
  - run_onnx_inference(session, tensor: Tensor, cfg) -> tuple[str, float]:
      Returns (label, confidence) [FR-CV-004: sigmoid, threshold 0.5]
      Latency target: ≤ 1,500 ms [NFR-Scalability]

IMPLEMENT: src/tests/test_cv.py  [V.E.R.I.F.Y. L3]
  1. test_model_output_shape: forward pass → logits shape [1, 2]
  2. test_sigmoid_output_range: confidence score in (0.0, 1.0)
  3. test_freeze_unfreeze: verify backbone frozen/unfrozen correctly
  4. test_onnx_export_creates_file: .onnx file exists after export
  5. test_onnx_equivalence: |onnx - pytorch| < 1e-5 for same input
  6. test_onnx_latency: single inference ≤ 1,500 ms on CPU
  7. test_onnx_cpu_provider_only: session uses CPUExecutionProvider
  8. test_class_weights_sum: class weights are valid tensors

ALSO ADD to notebooks/dsdba_training.ipynb:
  Cell 6: Training loop using train.py run_training(cfg)
  Cell 7: ONNX export and equivalence verification
  Cell 8: Upload checkpoint to HF Hub via huggingface-mcp
  Cell 9: Print EER and AUC-ROC results — verify ≤ 10% and ≥ 0.90

After implementation, run: pytest src/tests/test_cv.py -v
All 8 tests must pass before running Gate Check.
```

---