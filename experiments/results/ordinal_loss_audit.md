# PHASE C — CE / CORAL / Mild-CORAL Audit

**Status:** PASS (1 WARNING)  
**Artifact:** `experiments/results/ordinal_loss_audit.json`  
**Script:** `src/scripts/audit_ordinal.py`

---

## CORAL targets (K=5)

Tasks correspond to \(P(y \ge 1),\ldots,P(y \ge 4)\) via targets \(1\{y > k\}\) for \(k=0..3\).

| Grade | Expected | Got | OK |
|------:|----------|-----|:--:|
| 0 | [0,0,0,0] | [0,0,0,0] | ✔ |
| 1 | [1,0,0,0] | [1,0,0,0] | ✔ |
| 2 | [1,1,0,0] | [1,1,0,0] | ✔ |
| 3 | [1,1,1,0] | [1,1,1,0] | ✔ |
| 4 | [1,1,1,1] | [1,1,1,1] | ✔ |

dtype float, shape (B,4), `binary_cross_entropy_with_logits`, reduction **mean**.

---

## Decode (monotonic)

Threshold-sum decode with extreme logits (±20) maps all five patterns to grades 0–4 correctly.

---

## MildAwareCORAL vs CORAL

\[
L = \mathrm{mean}\big(\mathrm{BCE}_{\mathrm{none}}(\ell, 1\{y>k\}) \cdot w_k^{\mathrm{thr}} \cdot w_i^{\mathrm{sample}}\big)
\]

- Default \(w^{\mathrm{thr}} = [2,2,1,1]\) (boosts NoDR/Mild/Moderate cuts)  
- \(w^{\mathrm{sample}} = 2\) when \(y=1\), else 1  
- Manual expansion matches implementation ✔  
- Same `CoralLayer` + decode path as CORAL; **only training loss differs** ✔  

---

## WARNING

**Non-monotonic ordinal logits:** `decode_predictions` (count of \(\sigma(\ell_k)>0.5\)) can disagree with `argmax(class_probs)`.

Example pattern [1,0,1,0] → decode grade 2, argmax(probs) grade 1.

Inference / QWK uses threshold-sum. ECE uses converted probs. Document both; do **not** silently change decode without re-evaluating all CORAL runs.

---

## Experiments potentially affected

- None blocked by a bug.  
- Mild-CORAL B0 results remain valid *if* same decode was used throughout (it was).  
- MV CORAL cells still **REQUIRES RERUN** due to Phase B batch-1 issue, not due to ordinal math.

---

## Phase status

**PASS** — proceed to Phase D.
