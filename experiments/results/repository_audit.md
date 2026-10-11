# PHASE A — Repository Audit

**Date:** 2026-09-22  
**Status:** WARNING (pipeline sound; MobileViT seed-42 results protocol-confounded)  
**Mode:** Inspection only — no production code modified during this phase.

---

## 1. Current architecture

| Layer | Implementation | Notes |
|-------|----------------|-------|
| CNN baseline | `EfficientNetB0` (torchvision IMAGENET1K_V1) | Linear or CoralLayer head |
| Hybrid / “MV” | `timm.create_model("mobilevit_s")` + CE/CORAL head | Alias `cnn_transformer_hybrid` → same |
| Transformer-only | `SwinTiny` stub | `NotImplementedError` |
| Ordinal head | `CoralLayer` (shared weight + K−1 biases) | Cao et al. style |
| Loss | CE / CORAL / MildAwareCORAL | Focal = stub |

**What “MV” means here:** MobileViT-S via `timm`, not a custom CNN→tokenize→Transformer reimplementation.

---

## 2. Current data pipeline

| Dataset | n | Role |
|---------|--:|------|
| APTOS train | 2930 | train only |
| APTOS val | 366 | selection + temperature fit |
| APTOS test | 366 | internal test |
| IDRiD (train+test merged) | 516 | external only |
| Messidor-2 (gradable) | 1744 | external only |
| DeepDRiD regular fundus | 1589 | external only |

Verified: zero ID overlap train∩val, train∩test, val∩test. Externals never constructed in `train.py`.

---

## 3. Current preprocessing

- **rgb:** RGB → Resize(224) → [train augs] → ImageNet normalize  
- **green_clahe:** green + CLAHE(clip=2.0, tiles=8×8) → 3-ch stack → Resize → [augs] → normalize  
- Train augs only: HFlip, Rotation±10°, ColorJitter. Eval: no augs.  
- Lossless PNG cache for APTOS/IDRiD/Messidor. DeepDRiD online.  
- **Gap vs stated protocol:** no retinal FOV crop.

---

## 4. Current training pipeline

- AdamW 1e−4, wd 1e−4, cosine, 20 epochs  
- Selection: APTOS val QWK → `best.pt`  
- Resume via `last.pt` (interrupt + epoch)  
- Device override (`cpu`/`mps`/`cuda`); AMP CUDA only  

---

## 5. Current losses

- CORAL targets: `1{y > k}` for k=0..3  
- MildAwareCORAL: threshold weights default `[2,2,1,1]`, mild sample weight 2.0  
- Decode: `sum(sigmoid > 0.5)`; probs via cumulative differences + renormalize  

---

## 6. Current evaluation

- Metrics: Acc, Macro-F1, QWK, MAE, per-class P/R/F1, grade-1 F1 (one-vs-rest ✔), 0↔1 / 1↔2, ECE, temp-scaled ECE, bootstrap CIs  
- Missing: weighted-F1, reliability diagrams, CSV predictions, Grad-CAM  

---

## 7. Current checkpoint system

- Unique run dirs under `checkpoints/`, `results/`, `experiments/`  
- Never overwrites prior timestamped runs  
- Hybrid “stable” IDs used for interrupt resume  

---

## 8. Current experiment matrix

| Cell | seed 42 | 43 | 44 | 4-dataset eval |
|------|---------|----|----|----------------|
| B0 RGB/G+CLAHE × CE/CORAL | ✔ | ✔ | ✔ | mostly APTOS+IDRiD |
| B0 RGB/G+CLAHE × mild-CORAL | ✔ | – | – | ✔ all 4 |
| MV 2×3 (RGB/G+CLAHE × CE/CORAL/mild) | ✔ (`*_stable`) | – | – | ✔ all 4 |

**CRITICAL confound:** all 6 MV `*_stable` runs used `batch_size=1`, `device=cpu` during memory recovery. MobileViT-S has 32 BatchNorm2d layers → batch-1 training is scientifically invalid for architecture comparison. **Mark REQUIRES RERUN.**

---

## 9. Known limitations

1. CRITICAL: MV batch-1/CPU BN confound  
2. No Transformer-only baseline  
3. No FOV crop in P1  
4. Not a git repo / no environment.json per run  
5. mild-CORAL weights not validation-tuned  
6. decode vs argmax(probs) can disagree for non-monotonic CORAL  
7. B0 seed replicates lack Messidor/DeepDRiD in older reports  

---

## 10. Recommended next files to modify / add

| Phase | File |
|-------|------|
| B | `src/scripts/audit_mv_hybrid.py`, `experiments/results/mv_hybrid_audit.md` |
| C | `src/scripts/audit_ordinal.py`, `experiments/results/ordinal_loss_audit.md` |
| D | `src/scripts/visualize_preprocessing.py`, `experiments/results/preprocessing_*` |
| E | `src/scripts/audit_splits.py` |
| F | `configs/research/frozen_protocol.yaml` |
| G+ | `src/scripts/run_hybrid_2x2.py` with **batch≥16**, matched to B0 recipe |

---

## Phase status

**PASS with WARNING** — proceed to Phase B (MV audit) before any hybrid retrain.
