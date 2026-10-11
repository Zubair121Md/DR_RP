# Research Pipeline — Phase Flowchart & Status

**Updated:** 2026-09-28  
**Rule:** Cite only protocol-valid runs. Old MV `*_stable` batch=1 = REQUIRES_RERUN.

```
 BASIC ✓   TRAINING ✓    ANALYSIS ✓         OPTIONAL ✓              PAPER ✓ (draft wired)
 A–F       G/H s42–44    I–N + paired M     O Swin ×3 DONE          Q tables · R figures 01–22
                         LODO P ×3 seeds    (green×CE)              Grad-CAM · S integrity PASS
                                                                    DRAFT.md skeleton
```

## Status board

| Phase | Status | Deliverable |
|------:|--------|-------------|
| A–N, Q–S | **DONE** | core hybrid paper pack |
| O | **DONE** | Swin-Tiny ×3 · `phase_o_arch_comparison.md` · figs 21–22 · `table_phase_o_arch` |
| P | **DONE** | LODO ×3 holdouts × seeds 42/43/44 · `phase_p_lodo_seed_summary.md` · figs 19–20 · `table_lodo_vs_h2` |
| Draft | **DONE** | `paper/DRAFT.md` (Intro→Discussion wired to tables/figures) |

## LODO headline (mean±std, n=3)

| Holdout | LODO | H2 APTOS-only | Δ |
|---------|-----:|--------------:|--:|
| IDRiD | **0.678±0.071** | 0.364±0.148 | +0.314±0.078 |
| Messidor-2 | **0.223±0.066** | 0.050±0.046 | +0.173±0.102 |
| DeepDRiD | 0.454±0.023 | 0.417±0.112 | +0.037±0.089 |

## Phase O headline (green_clahe×CE, mean±std)

| Model | APTOS | IDRiD |
|-------|------:|------:|
| B0 | 0.896±0.010 | 0.711±0.010 |
| MobileViT H2 | 0.876±0.013 | 0.364±0.148 |
| Swin-Tiny | **0.904±0.016** | **0.634±0.041** |

## What’s left (before journal upload)

1. Author names, affiliations, acknowledgements, competing interests, CRediT  
2. Journal template (IEEE / Elsevier / Springer)  
3. Optional: more seeds / higher-resolution re-runs (called out as limitations)  
4. Final proofread of `paper/MANUSCRIPT.docx`

**Manuscript:** `paper/MANUSCRIPT.md` · figures in `paper/figures/main/`
