# Protocol deviation: learning-rate schedule stratification

## Summary

Primary MobileViT-S runs labelled with seeds **42–44** did not receive the intended full cosine learning-rate schedule. Scheduler state was not retained across mid-epoch checkpoint-resume cycles (≈60 optimiser steps per launch; 239–240 resumes per run). Logged final learning rate for these runs was **8.84 × 10⁻⁵**, not 0.

Runs labelled **45–47** received the intended **full cosine** schedule (final LR 0; 800 steps per launch).

## Consequences for analysis

- Seeds 42–44 and 45–47 are **not** exchangeable replicates of one protocol.
- All primary QWK, mild-recall, and ECE reporting is **schedule-stratified**.
- Pooled six-seed confirmatory hypothesis tests are **not** used as headline evidence.
- EfficientNet-B0 matched runs used a full cosine schedule with seeds 42–44 (see Supplementary Table S2); comparisons to MobileViT-S must respect schedule matching (prefer MobileViT-S 45–47 when claiming schedule-matched contrasts).

## Supporting artifacts

- `paper/tables/training_schedule_audit.csv`
- `paper/tables/environment.csv`
- Manuscript transparency note and §4.1
