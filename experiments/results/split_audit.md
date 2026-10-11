# PHASE E — Dataset / Split Audit

**Status:** PASS_WITH_WARNINGS  
**Artifact:** `experiments/results/split_audit.json`  
**Script:** `src/scripts/audit_splits.py`

---

## APTOS splits (CSV)

| Split | N | Unique IDs | Dup IDs |
|-------|--:|----------:|--------:|
| train | 2930 | 2930 | 0 |
| val | 366 | 366 | 0 |
| test | 366 | 366 | 0 |

**ID overlaps:** train∩val = 0, train∩test = 0, val∩test = 0.

Class counts are imbalanced (as expected for ICDR); mild (class 1) is scarce on all splits.

---

## External sets

| Dataset | N |
|---------|--:|
| IDRiD | 516 |
| Messidor-2 | 1744 |
| DeepDRiD | 1589 |

---

## Leakage probes

- Sampled content hashes (80/split APTOS; 80/external): no duplicate-hash groups within splits.  
- Cross-dataset MD5 probe (50 images each): **0 overlaps** among APTOS train/test × IDRiD × Messidor-2 × DeepDRiD pairs checked.  
- Code: checkpoint selection = **APTOS val QWK only**; temperature fit = **APTOS val only**; externals never used in training/selection.

---

## WARNINGS (document in paper)

1. DeepDRiD has `patient_id`; metrics are **image-level** — patient clustering not applied.  
2. APTOS / IDRiD / Messidor-2 lack patient IDs in provided labels — patient-level leakage **cannot** be fully audited.

---

## Phase status

**PASS_WITH_WARNINGS** — safe to freeze protocol and start Phase G hybrid retrain.
