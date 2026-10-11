# B0 vs MobileViT-S (APTOS/IDRiD QWK; B0 uses CORAL where hybrid uses mild-CORAL)

| Model | APTOS QWK | IDRiD QWK |
| --- | --- | --- |
| B0 RGB×CE | 0.883±0.004 | 0.669±0.027 |
| Hybrid RGB×CE | 0.830±0.006 | 0.395±0.147 |
| B0 G+CLAHE×CE | 0.896±0.010 | 0.711±0.010 |
| Hybrid G+CLAHE×CE | 0.876±0.013 | 0.364±0.148 |
| B0 RGB×CORAL | 0.905±0.002 | 0.721±0.021 |
| Hybrid RGB×mildCORAL | 0.821±0.005 | 0.353±0.222 |
| B0 G+CLAHE×CORAL | 0.905±0.004 | 0.755±0.015 |
| Hybrid G+CLAHE×mildCORAL | 0.838±0.014 | 0.565±0.172 |
