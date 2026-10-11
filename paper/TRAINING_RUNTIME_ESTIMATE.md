# Estimated wall-clock BEFORE any new training

**Decision (2026-09-28):** Under the reworded research question in `paper/RESEARCH_QUESTION.md`, **no new training is required** for the locked claim set. Existing H1–H4 × seeds 42–44 (+ archived B0 / Swin / LODO) are enough.

**Update (2026-09-29):** Optional strengthening package is scripted as **paper_v3** (matched B0, hybrid seeds 45–47, mild-CORAL decomposition, optional 448). See `paper/TRAINING_RUNTIME_ESTIMATE_V3.md`. **Do not launch until you confirm the printed estimate.**

The older `configs/paper_v2` 36-job queue remains deferred (superseded by the tighter paper_v3 grouping).

---

## Optional deferred estimate (if launched later)

Assumptions (stated explicitly):
- Hybrid CPU training throughput ≈ 1.7 it/s at batch size 4 (observed in LODO/hybrid logs).
- APTOS train steps/epoch ≈ ceil(2930/4) = 733; 20 epochs ≈ 14,660 steps ≈ 2.4 h pure train
  plus val/eval overhead → use **~4–6 h per MobileViT-S APTOS-only run** on this machine (CPU).
- Matched B0 on CPU bs=4 is typically faster per step than MobileViT but still CPU-bound;
  use **~3–5 h per B0 matched run** (conservative band).
- Archived B0 on MPS bs=32 was ~6 h; that is NOT the matched regime.
- LODO not requested for new seeds.
- Efficiency benchmark: <1 h.

Optional trainings (all *_v2, new dirs only):
1. Matched B0: {RGB, green-CLAHE} × {CE, mild-CORAL} × seeds {42,43,44} = **12 runs**
2. Extra seeds 45–47 for H1–H4 = **12 hybrid runs**
3. Extra seeds 45–47 for matched B0 = **12 B0 runs**
Total trainings: **36**

Point estimate:
- 12 hybrid × 5 h + 24 B0-matched × 4 h ≈ 60 + 96 = **156 h ≈ 6.5 days** continuous CPU
- Range: **~5–9 days** depending on thermal throttling / contention

Plus:
- Full 4-domain eval after each run: ~0.5–1.5 h each → **+18–54 h**
- Efficiency (B0, MobileViT-S, Swin; CPU+MPS; warmup + 50 reps): **~0.5–1 h**

**Grand total estimate if launched: ~7–11 days** wall-clock on this Mac if run serially with caffeinate.

**NOT REQUIRED for the reworded RQ — awaiting separate explicit confirmation if fairness/power work is wanted later.**
