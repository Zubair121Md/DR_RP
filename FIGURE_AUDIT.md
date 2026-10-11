# Figure-by-figure audit

**Audit date:** 2026-10-11  
**Builders:** `paper/figures/build_version_b_main.py`, `fig1_pipeline.py`, `build_supplementary.py`

| Figure | Caption matches Version B | Axes / legend | Schedule markers | Source | PDF+PNG | Issues | Status |
|---|---|---|---|---|---|---|---|
| Fig 1 | Protocol deviation disclosed | Study flowchart | Seed groups labelled | Design figure | Yes | — | Pass |
| Fig 2 | CLAHE preprocess | Steps a–g | N/A | Example APTOS mild | Yes | — | Pass |
| Fig 3 | Threshold vs CE + rank consistency | Schematic | N/A | Conceptual + audit | Yes | — | Pass |
| Fig 4 | Schedule-stratified QWK | Datasets × H1–H4 | △ 42–44 / ○ 45–47; group means only | `table_v3_hybrid_qwk_n6_per_seed.csv` | Yes | No pooled six-seed mean | Pass |
| Fig 5 | Grade-1 recall companion | Datasets + n | Same markers | mild recall CSVs | Yes | High variance disclosed | Pass |
| Fig 6 | H1/H2 ECE only | Before/after T | Schedule + fill for T | `ece_ce_cells_n6_per_seed.csv` | Yes | No threshold-head panels | Pass |
| S1 | Grade distributions | Fractions + counts | N/A | Manifest | Yes | — | Pass |
| S2 | Val QWK curves | Epoch | Solid/dashed schedule | histories | Yes | — | Pass |
| S3 | Efficiency | Params/FLOPs/latency | N/A | efficiency table | Yes | — | Pass |
| S4 | Augmentation | Examples | N/A | preprocess cache | Yes | — | Pass |
| S5 | Architecture | Heads | N/A | schematic | Yes | — | Pass |
| S6 | Dataset examples | Grades 0/2 | N/A | fixed seed | Yes | Stale alias file may exist | Pass* |
| S7 | Bootstrap CI | Seeds 42–44 | — | bootstrap CSV | Yes | Conditional on image sampling | Pass |
| S8 | Domain-shift drop | Externals | Schedule | derived QWK | Yes | — | Pass |
| S9 | APTOS metrics | Multi-metric | Schedule | APTOS metrics | Yes | — | Pass |
| S10 | Temperature ECE pairs | Domains | Schedule | ECE CSV | Yes | CE only | Pass |
| S11–S14 | Confusion matrices | H1–H4 | One row per schedule group (summed within group) | metrics JSON | Yes | Schedule-stratified | Pass |
| S15 | Messidor pred shares | Per run | Per seed bars | predictions | Yes | — | Pass |
| S16 | Referable DR | Seeds 42–44 | — | table_B | Yes | — | Pass |
| S17 | Matched B0 | Secondary | — | B0 tables | Yes | — | Pass |
| S18 | LODO | Secondary | — | LODO tables | Yes | — | Pass |
| S19 | Grad-CAM | Qualitative | Seed 42 | panel builder | Yes | Main-text removed (Version B) | Pass |
| S20 | Decomp | Secondary | 42–44 | decomp CSV | Yes | — | Pass |
| S21 | Resolution | Secondary | Caveat vs 45–47 | res448 CSV | Yes | — | Pass |

## Design checks

| Check | Result |
|---|---|
| Color-blind / Okabe–Ito intent | Main Version B builders use distinct markers + colors |
| Grayscale | Markers (△/○) preserve schedule encoding without color |
| Column-width readability | Dense but acceptable; Fig 6 is 2×4 — monitor PDF crop |
| Misleading pooled error bars | Avoided on Figs 4–6 (group means only) |
| Duplicated main content | Old main aliases (e.g. fig05_qwk_dots) may remain on disk; docx embeds Version B names |

## Actions

1. Optionally delete stale `paper/figures/main` aliases not cited by `build_manuscript_docx.py`.  
2. S11–S14 rebuilt as schedule-stratified panels (done).
