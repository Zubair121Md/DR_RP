# Protocol addendum: research question and novelty (no new experiments)

**Status:** framing-only addendum. Does **not** modify `configs/research/frozen_protocol.yaml`.  
**Date:** 2026-09-28  
**Decision:** Existing archived H1–H4 × seeds 42–44 (+ B0 / Swin / LODO as already reported) are sufficient under this wording. The optional `configs/paper_v2` 36-job queue remains deferred.

---

## Research question (canonical — use verbatim)

Under APTOS-only training, with model selection based exclusively on APTOS validation QWK, does green-channel CLAHE improve the cross-dataset QWK of MobileViT-S relative to RGB input, and what are the associated trade-offs in mild-stage recall and expected calibration error (ECE)? Further, how does a mild-aware CORAL objective (ordinal loss with fixed mild-stage weighting) alter these trade-offs?

## Novelty (canonical — use verbatim)

This study presents a controlled evaluation of preprocessing and ordinal-loss choices for cross-dataset diabetic retinopathy grading. Whereas much prior work reports in-domain performance [4–6,20–22], we isolate the effects of green-channel CLAHE and a mild-aware CORAL objective (as a combined objective, relative to cross-entropy) under a fixed MobileViT-S architecture, APTOS-only training, and APTOS-validation-QWK model selection. We evaluate cross-dataset QWK on three unseen datasets together with the trade-off between QWK, mild-stage recall, and ECE, providing indicative evidence on when these choices improve or degrade performance across domains.

## Explicit non-claims

- Not plural pipelines or plural hybrids.
- Mild-aware CORAL is not deconfounded into “ordinal” vs “mild weights”.
- CPU batch-4 / freeze_bn is a training resource constraint, not deployment evidence.
- “Consistent external improvement” is not claimed; Messidor-2 failure is part of the answer.

## Original frozen_protocol_v1 wording (unchanged on disk)

> Under a fixed preprocessing budget and deployable compute constraints, do lightweight retina-specific preprocessing pipelines consistently improve cross-dataset generalization of CNN–Transformer hybrids for DR grading, and does a mild-stage-aware ordinal head preserve these gains and calibration across unseen domains?

That wording overclaims the design; the canonical RQ above is the manuscript source of truth.
