# Phase O — CNN vs Hybrid vs Transformer

Matched recipe: **green_clahe × CE**. Seeds 42/43/44 where available.

| Model | Split | QWK mean±std |
|-------|-------|-------------:|
| EfficientNet-B0 | APTOS | 0.896±0.010 |
| EfficientNet-B0 | IDRiD | 0.711±0.010 |
| EfficientNet-B0 | Messidor-2 | n/a in B0 seed table |
| EfficientNet-B0 | DeepDRiD | n/a in B0 seed table |
| MobileViT-S H2 | aptos_test | 0.876±0.013 |
| MobileViT-S H2 | idrid_external | 0.364±0.148 |
| MobileViT-S H2 | messidor2_external | 0.050±0.046 |
| MobileViT-S H2 | deepdrid_external | 0.417±0.112 |
| Swin-Tiny | aptos_test | 0.904±0.016 |
| Swin-Tiny | idrid_external | 0.634±0.041 |
| Swin-Tiny | messidor2_external | 0.262±0.154 |
| Swin-Tiny | deepdrid_external | 0.514±0.021 |

Swin runs: `results/phase_o_swin_tiny_s{42,43,44}_v1/`
