# Phase G complete — Hybrid 2×2 seed 42

**Status:** `HYBRID_2X2_ALL_OK`  
**Protocol:** frozen_protocol_v1 · CPU · batch_size=4 · freeze_bn=True · chunked max_steps=60  
**Date finished:** 2026-09-23 (~03:44 IST for H4)

| Cell | Preprocess | Loss | Run ID | Best val QWK | Best epoch | Checkpoint |
|------|------------|------|--------|-------------:|-----------:|------------|
| H1 | rgb | CE | `hybrid_h1_s42_v1` | **0.856** | 20 | `checkpoints/hybrid_h1_s42_v1/best.pt` |
| H2 | green_clahe | CE | `hybrid_h2_s42_v1` | **0.859** | 20 | `checkpoints/hybrid_h2_s42_v1/best.pt` |
| H3 | rgb | mild_aware_coral | `hybrid_h3_s42_v1` | **0.806** | 20 | `checkpoints/hybrid_h3_s42_v1/best.pt` |
| H4 | green_clahe | mild_aware_coral | `hybrid_h4_s42_v1` | **0.848** | 20 | `checkpoints/hybrid_h4_s42_v1/best.pt` |

Selection metric: APTOS **validation** quadratic weighted kappa only.

### Early read (val only — not for architecture claims until external eval)

- Best val cell: **H2** (green_clahe × CE, QWK 0.859).  
- Mild-CORAL hybrids (H3/H4) trail CE on val QWK under this recipe; H4 recovers vs H3 with green_clahe.  
- Prior MobileViT `*_stable` batch=1 runs remain **REQUIRES_RERUN / do not cite**.

### Next

1. Phase I — external eval (APTOS test, IDRiD, Messidor-2, DeepDRiD) on H1–H4.  
2. Phase H — seeds 43/44 same 2×2.  
3. Then mild-DR / calibration / stats vs B0.
