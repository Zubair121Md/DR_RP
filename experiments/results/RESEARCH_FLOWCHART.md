# Master research flowchart — basic → training → analysis → paper

```
═══════════════════════════════════════════════════════════════════════════════
 LAYER 0 — FOUNDATION
═══════════════════════════════════════════════════════════════════════════════
  ✓ ICDR 0–4, APTOS splits, IDRiD / Messidor-2 / DeepDRiD
  ✓ Preprocess caches: rgb + green_clahe
  ✓ B0 2×2 + mild-CORAL baseline
  ✓ Efficiency snapshot
  ✗ Prior MV *_stable @ batch=1  →  REQUIRES RERUN (do not cite)

═══════════════════════════════════════════════════════════════════════════════
 LAYER 1 — BASIC / INTEGRITY          [Phases A–F]  ✓ DONE
═══════════════════════════════════════════════════════════════════════════════
 LAYER 2 — TRAINING                   [G–H, O–P]    ✓ DONE
═══════════════════════════════════════════════════════════════════════════════
  G/H  Hybrid 2×2 seeds 42/43/44
  O    Swin-Tiny green×CE ×3
  P    LODO MobileViT green×CE ×3 holdouts × seeds 42/43/44

═══════════════════════════════════════════════════════════════════════════════
 LAYER 3 — ANALYSIS                   [I–N]         ✓ DONE
═══════════════════════════════════════════════════════════════════════════════
 LAYER 4 — PAPER                      [Q–S + draft] ✓ ARTIFACTS + DRAFT
═══════════════════════════════════════════════════════════════════════════════
  Q tables (hybrid + O/P) · R figures 01–22 + Grad-CAM · S integrity PASS
  Draft: paper/DRAFT.md  (human prose / citations remaining)
```

## Scientific claims gate

- Architecture claims about MobileViT → only `hybrid_h*_s{42,43,44}_v1`
- Preprocess / mild-CORAL → hybrid + B0 evidence; caveat B0 CORAL ≠ mild-CORAL
- Do not mix batch=1 MV numbers into any table
- LODO / Swin = optional ablations; LODO trains on non-held-out externals
