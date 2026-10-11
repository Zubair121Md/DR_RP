# Paper v3 training runtime estimate (prioritized package)

**Status:** scripts ready; **training NOT launched**. Confirm this estimate before running
`bash scripts/run_paper_v3_caffeinated.sh`.

## Job groups

| Group | Jobs | What |
|-------|-----:|------|
| `core` | 24 | Matched EfficientNet-B0 H1–H4 × seeds 42–44 (12) + hybrid MobileViT-S H1–H4 × seeds 45–47 (12) |
| `decomp` | 6 | Under CLAHE: D1 unweighted CORAL × 42–44 (3) + D2 mild-weighted CE × 42–44 (3) |
| `res448` | 6 | Hybrid H1/H2 at 448×448 × seeds 42–44 (optional) |
| optional GPU BN | 1 | Separate config only (`configs/paper_v3_optional/`); not in the serial queue |

Default launch groups: `core,decomp` (30 jobs).

All run IDs end in `_v3` (or `_v3_opt`). Existing `*_v1`, archived B0, and `paper_v2` dirs are never overwritten.

## Assumptions

- Hybrid CPU throughput ≈ same as primary hybrids: **~4.5–6.5 h** train+eval per MobileViT-S APTOS-only run (bs=4, freeze_bn, chunked).
- Matched B0 under the same CPU regime: **~3.5–5.0 h** train + **~0.5–1.5 h** eval.
- 448×448 without cache: **~14–22 h** per run on CPU (roughly 3–4× vs 224).
- Serial queue only (parallel would jetsam on this machine).
- Optional GPU BN (mps, bs=16, freeze_bn=False): **~1–3 h** for one H2 seed-42 check.

## Point estimates

### Recommended default (`PAPER_V3_GROUPS=core,decomp` → 30 jobs)

- 12 matched B0 × ~4–6.5 h ≈ 48–78 h
- 12 hybrid seeds 45–47 × ~5–8 h ≈ 60–96 h
- 6 decomp × ~5–8 h ≈ 30–48 h
- **Total ≈ 138–222 h ≈ 5.8–9.3 days** serial CPU

### With optional 448 (`PAPER_V3_GROUPS=core,decomp,res448` → 36 jobs)

- Add 6 × ~14.5–23.5 h ≈ 87–141 h
- **Grand total ≈ 225–363 h ≈ 9.4–15.1 days** serial CPU

### Optional GPU BN only

- Generate: `python3 scripts/generate_paper_v3_optional_gpu_bn.py`
- Run manually after confirmation (~1–3 h on MPS).

## How to launch (only after you confirm)

```bash
# Generate configs + print estimate (no training)
python3 scripts/generate_paper_v3_configs.py
PAPER_V3_GROUPS=core,decomp python3 scripts/print_paper_v3_runtime_estimate.py

# After explicit confirmation:
PAPER_V3_GROUPS=core,decomp bash scripts/run_paper_v3_caffeinated.sh

# Include 448 later:
PAPER_V3_GROUPS=core,decomp,res448 bash scripts/run_paper_v3_caffeinated.sh
```

## Decomposition cells (what they answer)

| Cell | Input | Loss | vs H2 / H4 |
|------|-------|------|------------|
| D1 | green_clahe | unweighted CORAL | ordinal head without mild weights |
| D2 | green_clahe | mild-weighted CE | CE head with mild sample weight 2, no ordinal head |
| H2 | green_clahe | CE | primary |
| H4 | green_clahe | mild-aware CORAL | ordinal + threshold weights + mild sample weight |

Compare D1 vs H2 (ordinal alone), D2 vs H2 (mild weight alone), H4 vs both (full recipe).
