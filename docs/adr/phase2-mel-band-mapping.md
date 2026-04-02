# ADR — Phase 2 Q5 Resolution: Mel Filter Bank Frequency-to-Bin Mapping

**Document:** DSDBA-SRS-2026-002 v2.1  
**Phase:** 2 — System Design & Technical Specification  
**SRS refs:** FR-CV-013, FR-CV-014  
**Label:** [Phase 2 | v1 | Q5-RESOLVED]

## Decision

Q5 is resolved: band attribution rows SHALL be computed from Mel-bin center frequencies, not naive linear slicing.

### Locked mapping method

1. Generate Mel bin centers: `librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)`.
2. For each band in `config.yaml` (`gradcam.band_hz`), select bin indices where frequency belongs to `[low_hz, high_hz)` (top band includes upper bound).
3. Aggregate saliency over those row indices, then Softmax normalize to 100%.

This follows the Mel axis spacing and preserves physically meaningful boundaries.

## Context7 verification

From `context7-mcp` (librosa docs):

- `librosa.mel_frequencies` signature supports `n_mels`, `fmin`, `fmax`, `htk`.
- Default `n_mels=128`, `fmin=0.0`; `fmax` explicitly set to `8000.0` for DSDBA contract.

## Validation plan (Sprint C test)

- Build known synthetic spectrogram with energy concentrated in each target Hz band.
- Convert bins to Hz using locked mapping.
- Assert each test tone maps to expected band rows.
- Assert normalized `band_pct` sums to `100.0 ± 0.001`.

## Pseudocode sketch (contract-level)

```python
mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)
band_rows = {}
for band_name, (lo, hi) in band_hz.items():
    if band_name == "high":
        idx = np.where((mel_freqs >= lo) & (mel_freqs <= hi))[0]
    else:
        idx = np.where((mel_freqs >= lo) & (mel_freqs < hi))[0]
    band_rows[band_name] = idx
```
