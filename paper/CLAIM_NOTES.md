# Claim notes (research integrity)

Canonical RQ / novelty: see `paper/RESEARCH_QUESTION.md`.

## Research question (verbatim)

Under APTOS-only training, with model selection based exclusively on APTOS validation QWK, does green-channel CLAHE improve the cross-dataset QWK of MobileViT-S relative to RGB input, and what are the associated trade-offs in mild-stage recall and expected calibration error (ECE)? Further, how does a mild-aware CORAL objective (ordinal loss with fixed mild-stage weighting) alter these trade-offs?

## Novelty (verbatim)

This study presents a controlled evaluation of preprocessing and ordinal-loss choices for cross-dataset diabetic retinopathy grading. Whereas much prior work reports in-domain performance [4–6,20–22], we isolate the effects of green-channel CLAHE and a mild-aware CORAL objective (as a combined objective, relative to cross-entropy) under a fixed MobileViT-S architecture, APTOS-only training, and APTOS-validation-QWK model selection. We evaluate cross-dataset QWK on three unseen datasets together with the trade-off between QWK, mild-stage recall, and ECE, providing indicative evidence on when these choices improve or degrade performance across domains.

## Safe to claim (numbers only from paper/tables CSVs)
- H2 leads APTOS test and DeepDRiD QWK among hybrids.
- H4 leads IDRiD mean QWK among hybrids (high seed variance).
- CE hybrids have lower APTOS ECE than mild-CORAL hybrids.
- Green-CLAHE does **not** consistently improve all externals (Messidor-2 near zero).
- Selection = APTOS val QWK only; T fitted on APTOS val only.

## Do not claim
- Plural pipelines / hybrids; preservation of gains+calibration; ordinal vs mild deconfound; deployment readiness from CPU bs=4; BN-invalid batch-size-1 runs; selection on test/externals.

## B0 caveats
- Archived B0: different regime (e.g. larger batch / MPS) vs hybrids (CPU bs=4 freeze_bn).
- B0 ordinal cells = standard CORAL; hybrid H3/H4 = mild-CORAL.
- B0 Messidor/DeepDRiD incomplete for some cells — withhold those claims.

## Training deferred
- `configs/paper_v2` 36-job matched-B0 / extra-seed queue is **not required**.
