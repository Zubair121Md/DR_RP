# Threshold-head implementation audit

## Intended construction

Shared feature weight with four threshold biases (CORAL-inspired shared-weight form). Mild-weighted (MW) objective uses threshold weights (2, 2, 1, 1) and sample weight 2 for grade-1 images. Decoding: count σ(z) > 0.5.

## Observed failure

After training, biases were **not** non-increasing in any of 21 threshold-head runs. Cumulative probabilities were non-monotonic for ≥97.3% of test images in every run/domain.

## Reporting rules (Version B)

1. Do **not** call H3/H4/D1 valid CORAL or rank-consistent CORAL.
2. Report decoded-grade metrics (QWK, recall).
3. Do **not** report ECE / probability calibration for threshold-head models.
4. Referable-DR score for threshold heads uses σ(z₁), disclosed in Methods.

## Artifacts

- `paper/tables/coral_verification_runs.csv`
- `paper/tables/coral_verification.md`
- Manuscript §3.5 and Figure 3
