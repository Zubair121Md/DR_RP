# Paired statistics (Phase M)

Seeds: [42, 43, 44]. Exact Wilcoxon + sign-flip permutation on paired deltas.

**Caveat:** n=3 → even if every seed favors the same side, two-sided p is typically 0.25. Report mean±std of paired deltas; do not claim p<0.05 from these tests alone. B0 coral ≠ hybrid mild-CORAL — interpret H3/H4 vs B0 coral cautiously.

| Comparison | mean Δ (A−B) | std Δ | Wilcoxon p | sign-flip p | same-sign? |
|------------|-------------:|------:|-----------:|------------:|:----------:|
| B0 rgb_ce vs H1 APTOS QWK | +0.0526 | 0.0097 | 0.250 | 0.250 | Y |
| B0 rgb_ce vs H1 IDRiD QWK | +0.2741 | 0.1691 | 0.250 | 0.250 | Y |
| B0 green_clahe_ce vs H2 APTOS QWK | +0.0197 | 0.0029 | 0.250 | 0.250 | Y |
| B0 green_clahe_ce vs H2 IDRiD QWK | +0.3469 | 0.1448 | 0.250 | 0.250 | Y |
| B0 rgb_coral vs H3 APTOS QWK | +0.0843 | 0.0056 | 0.250 | 0.250 | Y |
| B0 rgb_coral vs H3 IDRiD QWK | +0.3675 | 0.2427 | 0.250 | 0.250 | Y |
| B0 green_clahe_coral vs H4 APTOS QWK | +0.0669 | 0.0149 | 0.250 | 0.250 | Y |
| B0 green_clahe_coral vs H4 IDRiD QWK | +0.1896 | 0.1806 | 0.250 | 0.250 | Y |
| H2−H1 APTOS (CLAHE under CE) | +0.0456 | 0.0185 | 0.250 | 0.250 | Y |
| H2−H1 IDRiD (CLAHE under CE) | -0.0304 | 0.2706 | 1.000 | 1.000 | N |
| H2−H1 DeepDRiD (CLAHE under CE) | +0.1796 | 0.1099 | 0.250 | 0.250 | Y |
| H4−H3 APTOS (CLAHE under mildCORAL) | +0.0165 | 0.0174 | 0.250 | 0.250 | Y |
| H4−H3 IDRiD (CLAHE under mildCORAL) | +0.2119 | 0.1645 | 0.250 | 0.250 | Y |
| H4−H2 APTOS (mildCORAL under CLAHE) | -0.0381 | 0.0148 | 0.250 | 0.250 | Y |
| H4−H2 IDRiD (mildCORAL under CLAHE) | +0.2010 | 0.1162 | 0.250 | 0.250 | Y |
| H3−H1 APTOS (mildCORAL under RGB) | -0.0091 | 0.0015 | 0.250 | 0.250 | Y |
| H2−H3 DeepDRiD | +0.3668 | 0.1024 | 0.250 | 0.250 | Y |
| H2−H4 Messidor-2 | +0.0593 | 0.1282 | 0.500 | 0.500 | N |
| H3−H1 APTOS mild recall | +0.1444 | 0.3501 | 0.750 | 0.750 | N |
| H4−H2 APTOS mild recall | +0.3111 | 0.2988 | 0.500 | 0.500 | N |
| H3−H1 APTOS ECE | +0.1254 | 0.0174 | 0.250 | 0.250 | Y |
| H4−H2 APTOS ECE | +0.1326 | 0.0044 | 0.250 | 0.250 | Y |

JSON: `experiments/results/paired_stats.json`
