## ⚡ CHAIN 00 — UNIVERSAL GATE CHECK
> **When:** After EVERY chain. This is your gatekeeper.  
> **Mode:** Cursor Agent Mode  
> **Paste into:** New Cursor chat (do not continue previous chain chat)

```
# ════════════════════════════════════════════════════════════════════════
# DSDBA — UNIVERSAL GATE CHECK (Chain 00)
# Run this after completing ANY chain before advancing to the next.
# ════════════════════════════════════════════════════════════════════════

Read all project context before evaluating:
  @file .cursorrules
  @file config.yaml
  @file docs/context/session-cheatsheet.md

Chain being evaluated: [FILL IN: e.g., "Chain 05 — Sprint A Audio DSP"]
Files produced in this chain: [FILL IN: list all files created or modified]
SRS requirements targeted: [FILL IN: e.g., "FR-AUD-001–011"]

Perform all 6 checks below. Do not pass any check unless criteria are fully met.
If ANY check fails → output [GATE: BLOCKED] with specific remediation instructions.
Do NOT suggest advancing until all 6 checks return PASS.

── CHECK 1: S.C.A.F.F. ─────────────────────────────────────────────────
[ ] Module header present with SDLC Phase, Sprint, SRS FR IDs, Interface Contract?
[ ] [EXPLORE] → [DECISION] reasoning documented as a comment or ADR?
[ ] All 5 S.C.A.F.F. elements (Situation/Challenge/Audience/Format/Foundations) satisfied?
[ ] MCP tool usage (context7/huggingface/stitch) explicitly stated where applicable?
[ ] No scope creep — code addresses only the FRs declared for this chain?
Output: S.C.A.F.F. → [PASS | FAIL: specify]

── CHECK 2: V.E.R.I.F.Y. ───────────────────────────────────────────────
[ ] Code logic explainable without reading comments (Verbalise)?
[ ] All imports version-pinned and compatible with both Colab T4 and HF Spaces CPU?
[ ] DSDBA edge cases present in test file for this module?

  Audio DSP edge cases (if applicable):
  [ ] < 0.5 s audio → JSON error AUD-001 (not a crash) [FR-AUD-005]
  [ ] Exactly 2.0 s → no crop, no pad [FR-AUD-004]
  [ ] > 2.0 s → centre-crop confirmed [FR-AUD-004]
  [ ] 0.5–2.0 s → zero-pad at right boundary [FR-AUD-004]
  [ ] Multi-channel → averaged to mono [FR-AUD-003]
  [ ] Output shape [3, 224, 224] float32 asserted [FR-AUD-008]

  CV / ONNX edge cases (if applicable):
  [ ] Tensor shape validation at module entry
  [ ] ONNX equivalence: |diff| < 1e-5 [FR-DEP-010]
  [ ] ONNX inference ≤ 1,500 ms on CPU [FR-DEP-010]

  Grad-CAM edge cases (if applicable):
  [ ] Target layer confirmed (Q4 resolved) [FR-CV-010]
  [ ] Mel Hz mapping used (not row-slice) — Q5 resolved [FR-CV-013]
  [ ] band_sum == 100.0 ± 0.001 [FR-CV-014]
  [ ] Grad-CAM ≤ 3,000 ms on CPU [FR-CV-015]

  NLP edge cases (if applicable):
  [ ] API timeout > 30 s → fallback + warning badge [FR-NLP-003]
  [ ] CV panel visible BEFORE NLP async resolves [FR-NLP-006]
  [ ] API key never appears in code or logs [FR-NLP-005]

  Deployment edge cases (if applicable):
  [ ] File > 20 MB → rejected before processing [FR-DEP-002]
  [ ] E2E ≤ 15,000 ms on CPU [FR-DEP-007]

[ ] Open Questions: are all blocking Qs for this chain resolved?
Output: V.E.R.I.F.Y. → [PASS | FAIL: specify edge case and FR ID]

── CHECK 3: D.O.C.S. ───────────────────────────────────────────────────
[ ] Module docstring complete (all required fields)?
[ ] Every function has a docstring with Args/Returns/Raises/Latency?
[ ] ADR created for any non-obvious design decision?
[ ] config.yaml updated with any new hyperparameters (zero magic numbers)?
[ ] session-cheatsheet.md has been updated with this session's status?
Output: D.O.C.S. → [PASS | FAIL: specify missing field]

── CHECK 4: S.H.I.E.L.D. ───────────────────────────────────────────────
[ ] No API keys in any .py file, notebook, or log output?
[ ] Audio processed in-memory only (BytesIO) — zero disk writes?
[ ] Error messages: generic to UI, verbose to structured log?
[ ] All dependencies pinned in requirements.txt?
[ ] ONNX session created once at startup (not per-request)?
[ ] Tensor contract [3, 224, 224] float32 validated at module boundary?
Output: S.H.I.E.L.D. → [PASS | FAIL: specify violation and NFR ref]

── CHECK 5: R.E.F.A.C.T. ───────────────────────────────────────────────
[ ] All function signatures have complete type hints?
[ ] Zero magic numbers — all constants reference config.yaml?
[ ] Zero print() statements — structured logger used?
[ ] No module exceeds 300 lines?
[ ] McCabe complexity ≤ 10 per function?
[ ] Latency assertions present in test file for all applicable targets?
[ ] All torch.Tensor ops specify dtype and device?
Output: R.E.F.A.C.T. → [PASS | FAIL: specify file and line]

── CHECK 6: SRS COMPLIANCE ─────────────────────────────────────────────
[ ] All SHALL requirements for this chain: 100% implemented?
[ ] SHOULD requirements: implemented or explicitly deferred with justification?
[ ] Every implemented function cites its FR ID in its docstring?
[ ] Acceptance criteria verified (latency/accuracy targets from NFR table)?
Output: SRS COMPLIANCE → [PASS | FAIL: list unmet FRs]

── FINAL VERDICT ────────────────────────────────────────────────────────
IF ALL 6 PASS:
  → Output: ✅ GATE OPEN — Chain [N] complete. Proceed to Chain [N+1].
  → Generate the session-cheatsheet.md update block for me to append.

IF ANY FAIL:
  → Output: 🔴 GATE: BLOCKED — list each failure with file, FR ID, and fix.
  → Do NOT suggest the next chain until all issues are resolved.
```

---
