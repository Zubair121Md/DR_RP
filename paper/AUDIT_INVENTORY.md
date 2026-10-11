# Paper audit inventory (read-only)

**Date:** 2026-09-28  
**Scope:** Explore only — no files modified when this inventory was first written.  
**Rule reminder:** Numbers for the paper must come from scripts reading these artifacts; never hand-typed.

---

## 1. Where artifacts live

| Kind | Location pattern |
|------|------------------|
| Per-split metrics | `results/<run_id>/<split>_metrics.json` |
| Per-image predictions | `results/<run_id>/<split>_predictions.json` |
| Checkpoints (best val QWK) | `checkpoints/<run_id>/best.pt` |
| Run configs / train summaries | `experiments/<run_id>/config.json`, `train_summary.json` (often mirrored under `results/<run_id>/`) |
| Frozen protocol (do not edit) | `configs/research/frozen_protocol.yaml` |
| Seed aggregates (derived) | `results/b0_2x2_seed_summary.json`, `experiments/results/hybrid_seed_summary.json` |
| Efficiency snapshot | `results/efficiency_models.json` (B0 + MobileViT only; no Swin) |

**Splits used in hybrid / Swin / LODO eval:**

- `aptos_test`
- `idrid_external` (IDRiD disease-grading **train+test**, `split=all`, n=516)
- `messidor2_external` (gradable adjudicated, n=1744)
- `deepdrid_external` (regular-fundus train+val eye-level, n=1589)

**Metrics JSON typically includes:** `n_samples`, `accuracy`, `macro_f1`, `quadratic_weighted_kappa`, `mae`, mild-DR P/R/F1, `confusion_matrix`, `ece`, `ece_temperature_scaled`, `temperature`, and a `bootstrap_cis` block (existing bootstrap; **n_boot / RNG not guaranteed to match the new 2000-resample spec** — recompute for paper tables).

**Predictions JSON keys (hybrids / Swin / LODO):** `ids`, `y_true`, `y_pred`, `probs`, `logits`.

**APTOS val predictions:** **MISSING** under `results/**` (no `aptos_val_*`). Temperature scalars are stored per evaluated metrics file (fitted on val at eval time); raw val logits are not archived.

---

## 2. Hybrid MobileViT-S H1–H4 × seeds 42/43/44 (`*_v1`)

| Run ID | Checkpoint | All 4 domain metrics | All 4 domain predictions | Logits on all 4 domains | Logit width |
|--------|:----------:|:--------------------:|:------------------------:|:-----------------------:|:-----------:|
| `hybrid_h1_s{42,43,44}_v1` | yes | yes | yes | **yes** | 5 (CE) |
| `hybrid_h2_s{42,43,44}_v1` | yes | yes | yes | **yes** | 5 (CE) |
| `hybrid_h3_s{42,43,44}_v1` | yes | yes | yes | **yes** | 4 (CORAL) |
| `hybrid_h4_s{42,43,44}_v1` | yes | yes | yes | **yes** | 4 (CORAL) |

**Training regime (from `experiments/hybrid_h2_s42_v1/config.json` — same pattern for H1–H4):**

- device: `cpu`
- batch_size: `4`
- freeze_bn: `True`
- epochs: 20, AdamW lr 1e-4, wd 1e-4, cosine
- selection: APTOS val QWK only
- H3/H4 loss: `mild_aware_coral` with `mild_sample_weight=2.0`, `threshold_weights=[2,2,1,1]`

**Per-image logits:** present for **every** hybrid run × **every** domain listed above. No re-inference required for hybrid analyses A–E.

---

## 3. Swin-Tiny (Phase O) × seeds 42/43/44

| Run ID | Checkpoint | 4-domain metrics/preds | Logits |
|--------|:----------:|:----------------------:|:------:|
| `phase_o_swin_tiny_s{42,43,44}_v1` | yes | yes | yes (dim 5, CE) |

Recipe: green_clahe × CE (configs under `configs/phase_o/`).

---

## 4. LODO (Phase P) × holdouts × seeds 42/43/44

| Holdout | Run ID pattern | Checkpoint | 4-domain metrics/preds/logits | Train size (from `train_summary.json`) |
|---------|----------------|:----------:|:-----------------------------:|----------------------------------------:|
| idrid | `phase_p_lodo_holdout_idrid_s{42,43,44}_v1` | yes | yes | 6263 |
| messidor2 | `phase_p_lodo_holdout_messidor2_s{42,43,44}_v1` | yes | yes | 5035 |
| deepdrid | `phase_p_lodo_holdout_deepdrid_s{42,43,44}_v1` | yes | yes | 5190 |

Val size always 366 (APTOS val). Model: MobileViT-S green_clahe × CE.

---

## 5. EfficientNet-B0 (archived “standard fine-tune”)

Seed table source: `results/b0_2x2_seed_summary.json` cells `rgb_ce`, `green_clahe_ce`, `rgb_coral`, `green_clahe_coral` (seeds 42/43/44).

### 5.1 Training / eval regime (archived B0)

From configs + `train_summary.json` (example: `baseline_efficientnet_b0_green_clahe_20260918_092447`):

| Setting | Archived B0 | Hybrid H1–H4 (for contrast) |
|---------|-------------|-----------------------------|
| batch_size | **32** | 4 |
| device | **mps** | cpu |
| freeze_bn | **not set** (BN trains normally) | True |
| epochs / optim / lr / wd / cosine | 20 / AdamW / 1e-4 / 1e-4 / cosine | same |
| Augmentations (config list) | HFlip 0.5, Rotate ±10°, ColorJitter brightness/contrast 0.1 (+ preprocess-specific frontend) | same family via `src/preprocessing.py` |
| Eval domains in seed table | **APTOS test + IDRiD only** for almost all CE/CORAL seed runs | all four |

### 5.2 Messidor-2 / DeepDRiD metrics for B0

| Run | Messidor-2 | DeepDRiD | Notes |
|-----|:----------:|:--------:|-------|
| Seed-table CE runs (rgb / green_clahe × 42–44) | **no** | **no** | Only aptos + idrid |
| Seed-table CORAL runs s43/s44 | **no** | **no** | Only aptos + idrid |
| `..._green_clahe_coral_20260918_203508` (s42) | **yes** | **yes** | Has logits |
| `..._green_clahe_mild_coral_20260919_182333` | **yes** | **yes** | mild-CORAL; **not** in primary B0 seed 2×2 (that uses standard CORAL) |
| `..._rgb_mild_coral_20260919_190352` | **yes** | **yes** | mild-CORAL; not in primary seed 2×2 |

**Gap:** There is **no complete 3-seed × 4-domain** Messidor/DeepDRiD coverage for B0 CE or B0 standard-CORAL matching the hybrid grid.

### 5.3 Logits for B0

| Situation | Logits in `*_predictions.json` |
|-----------|--------------------------------|
| Most B0 CE / CORAL seed runs | **MISSING** (often `ids`, `y_true`, `y_pred`, `probs` only) |
| A minority of runs (e.g. green_clahe_coral s42 full eval, mild_coral runs) | present |

**Implication:** Referable AUC / image-level bootstrap from logits for archived B0 CE seed runs requires **re-inference** from `checkpoints/.../best.pt` with a verify-against-archived-metrics≤1e-6 gate before use. Do not invent probs from labels alone.

---

## 6. Answer summary (prompt questions)

### Are per-image logits saved for every run and domain?

| Family | Answer |
|--------|--------|
| Hybrid H1–H4 × 42–44 × 4 domains | **Yes** |
| Swin-Tiny × 42–44 × 4 domains | **Yes** |
| LODO × 3 holdouts × 42–44 × 4 domains | **Yes** |
| Archived B0 seed-table runs × all domains | **No** — logits often missing; Messidor/DeepDRiD often missing |
| APTOS validation (any family) | **No** dedicated val prediction dumps |

### What batch size / device / freeze_bn / aug did EfficientNet-B0 use?

- **batch_size = 32**, **device = mps**, **freeze_bn unset/false**  
- Augmentations: random horizontal flip 0.5, rotation ±10°, color jitter brightness/contrast 0.1 (after frontend), ImageNet normalize, 224 input  

This is **not** matched to hybrid CPU / bs=4 / freeze_bn=True (hence the requested B0 `*_v2` matched rerun).

### Which Messidor-2 / DeepDRiD metrics exist for B0?

Only sparse runs (notably one green_clahe **standard CORAL** s42 full eval, plus **mild-CORAL** one-offs). Seed-table CE B0 and most CORAL seeds: **APTOS + IDRiD only**.

---

## 7. Missing inputs for paper tables / figures (blocking or partial)

| Need | Status |
|------|--------|
| Hybrid logits for A–E | OK |
| Temperature `T` per hybrid run | OK (in metrics JSON) |
| APTOS val logits (to refit T from scratch) | Missing — use stored `temperature` from metrics for ECE@T consistency with archived eval |
| B0 logits for referable/bootstrap on CE seeds | Missing — needs gated re-inference |
| B0 Messidor/DeepDRiD for CE × 3 seeds | Missing — needs new eval or matched `*_v2` training |
| Matched B0 (CPU, bs=4, freeze_bn, mild-CORAL 2×2) | **Does not exist** — training requested, not started |
| Seeds 45–47 | **Do not exist** |
| Swin efficiency (params/FLOPs/latency) | Missing from `efficiency_models.json` |
| `paper/tables/dataset_manifest.csv` | **Does not exist yet** — must be generated from CSVs/metrics `n_samples` before Fig 1 can read counts |

---

## 8. Safe next steps (no training until runtime confirmed)

1. Generate hybrid-only tables A–E from existing `results/hybrid_h*_s*_v1/*`.  
2. Optionally write gated B0 logit dump for runs that have checkpoints + archived metrics (verify ≤1e-6) — **does not overwrite** existing prediction files; write to new paths e.g. `results/<run_id>/reinfer_v2/`.  
3. Print estimated wall-clock for matched B0 `*_v2` + seeds 45–47 + efficiency; **wait for user confirm** before training.  
4. Grad-CAM CORAL fix + figure pipeline + manuscript checker.

---

## 9. Assumptions recorded during this audit

1. IDRiD external n=516 means train+test union (`split=all`), as implemented in `src/evaluate.py`.  
2. DeepDRiD external n=1589 is regular-fundus train+val with eye-level labels (`src/dataset.py`).  
3. B0 “seed table” membership is defined by `results/b0_2x2_seed_summary.json`, not every folder under `results/baseline_efficientnet_b0_*`.  
4. Existing `bootstrap_cis` in metrics may use a different bootstrap count than the new paper spec (2000, fixed RNG); treat as non-authoritative for new tables.
