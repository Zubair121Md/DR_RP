# PHASE D — Preprocessing Audit

**Status:** PASS with WARNING (documented FOV-crop gap)  
**Artifacts:** `experiments/results/preprocessing_diagnostics/` + `manifest.json`  
**Script:** `src/scripts/visualize_preprocessing.py`

---

## Pipelines (code truth)

| Name | Steps | Notes |
|------|-------|-------|
| **rgb (P0)** | RGB convert → Resize 224 → (train augs) → ImageNet normalize | No CLAHE |
| **green_clahe (P1)** | RGB → green channel → CLAHE(clip=2.0, tile=8×8) → stack 3ch → Resize 224 → augs → normalize | **No FOV crop** |

Train augs (both): HFlip 0.5, Rotate ±10°, ColorJitter B/C 0.1. Eval: none.

Cache stores deterministic frontend+resize; augs stay online (`CachedPreprocess`).

---

## Visual diagnostics (mild DR, grade 1)

Panels written for one mild sample per dataset:

| Dataset | Panel |
|---------|-------|
| APTOS train | `aptos_train_1b329a127307_y1.png` |
| APTOS val | `aptos_val_0024cdab0c1e_y1.png` |
| APTOS test | `aptos_test_e55188915f9d_y1.png` |
| IDRiD | `idrid_IDRiD_021_y1.png` |
| Messidor-2 | `messidor2_20051020_43832_0100_PP_y1.png` |
| DeepDRiD | `deepdrid_3_l1_y1.png` |

Automated checks: green_clahe output shape `(224,224,3)` and **all three channels equal** after stack — ✔ on all six samples.

---

## WARNING

| ID | Detail |
|----|--------|
| `NO_FOV_CROP` | Some protocol texts mention retinal FOV crop before CLAHE. `src/preprocessing.py` implements green+CLAHE only. Paper must describe **exactly** this pipeline; do not claim FOV crop. |

---

## Cross-dataset consistency

Same preprocess code path is used for APTOS / IDRiD / Messidor-2 / DeepDRiD. Domain shift is therefore camera/population, not a different frontend.

---

## Phase status

**PASS / WARNING** — proceed; frozen protocol records `green_clahe` without FOV crop.
