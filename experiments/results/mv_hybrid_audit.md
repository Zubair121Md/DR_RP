# PHASE B — MV / Hybrid Model Audit

**Status:** PASS with CRITICAL protocol finding (implementation OK; historical runs invalid)  
**Artifact:** `experiments/results/mv_hybrid_audit.json`  
**Script:** `src/scripts/audit_mv_hybrid.py`

---

## Definition of “MV”

In this repository, **MV = MobileViT-S via `timm`** (`src/models.py::MobileViTS`).

Forward path:

```
image [B,3,224,224]
  → timm mobilevit_s (CNN blocks + MobileViT transformer blocks)
  → global average pool
  → feature [B, 640]
  → Linear (CE) or CoralLayer (CORAL/mild-CORAL)
  → logits [B,5] or [B,4]
  → decode → grade 0–4
```

`cnn_transformer_hybrid` is an **alias** to the same class. `SwinTiny` is **not implemented**.

---

## Synthetic checks (pretrained=False)

| Model | Head | Batch | Logits | Loss finite | Grad ≠0 | Params update |
|-------|------|------:|--------|:-----------:|:-------:|:-------------:|
| mobilevit_s | CE | 1/8/32 | [B,5] | ✔ | ✔ | ✔ |
| mobilevit_s | CORAL | 1/8/32 | [B,4] | ✔ | ✔ | ✔ |
| efficientnet_b0 | CE/CORAL | 1/8/32 | OK | ✔ | ✔ | ✔ |

- No NaN/Inf  
- Probabilities sum ≈ 1  
- All parameters trainable by default (0 frozen)  
- Module counts (MV): **32 BatchNorm2d**, 21 LayerNorm, 37 Dropout  

---

## CRITICAL finding (not a code bug — a protocol confound)

| Field | Detail |
|-------|--------|
| **ID** | `MV_BN_BATCH1` |
| **File** | `src/models.py :: MobileViTS` |
| **Problem** | Historical `hybrid_*_s42_stable` runs used `batch_size=1` on CPU. BatchNorm with N=1 is ill-defined and corrupts running stats. |
| **Evidence** | Completed histories show train loss ↓ while val QWK collapses (e.g. RGB-CE never >0.06; mild-CORAL best@ep1 then ≈0). Configs on disk: `batch_size: 1`, `device: cpu`. |
| **Scientific consequence** | All 6 MV external rows in `external_eval_sweep.json` are **REQUIRES RERUN**. Do not use for architecture claims. |
| **Proposed correction** | Retrain hybrid grid with **batch_size ≥ 16** (prefer **32** to match B0). Keep architecture fixed. Do not silently swap BN for GroupNorm without documenting an ablation. |
| **Code change now?** | **None.** Implementation forward/backward is correct. |

---

## Trace checklist

| Item | Status |
|------|--------|
| Input 224×224×3 | ✔ |
| Feature dim 640 | ✔ |
| CE out [B,5] | ✔ |
| CORAL out [B,4] | ✔ |
| Pretrained ImageNet (timm) | available when `pretrained=True` |
| Gradient flow | ✔ nonzero |
| Accidental detach | none observed |

---

## Phase status

**PASS / WARNING** — proceed to Phase C (done in parallel) then D–F; **do not start Phase G until protocol freezes batch≥16**.
