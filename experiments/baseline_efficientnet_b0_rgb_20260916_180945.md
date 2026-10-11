# Experiment 1 — EfficientNet-B0 RGB baseline

**Run ID:** `baseline_efficientnet_b0_rgb_20260916_180945`  
**Status:** completed (trained + evaluated on Colab T4)

## Configuration
- Train: APTOS train (2930) only
- Validate: APTOS val (366) only — selection metric QWK
- Evaluate: APTOS test (366) + IDRiD external (516)
- Preprocess: RGB → 224×224 → ImageNet normalize
- Model: torchvision EfficientNet-B0 (ImageNet pretrained), 5-class head
- Loss: Cross Entropy
- Optimizer: AdamW lr=1e-4, wd=1e-4; CosineAnnealingLR; AMP; seed=42; batch=32; epochs=20

## Training
- Epochs: 20
- Best validation epoch: **9**
- Best val QWK: **0.8963**
- Best val Acc: **0.8361**
- Final train loss: 0.0954
- Final val loss: 0.5502
- Device: cuda (Tesla T4)
- Elapsed: ~8446 s (~2.3 h)

## APTOS test
| Metric | Value |
|--------|------:|
| Accuracy | 0.8497 |
| Macro-F1 | 0.6809 |
| QWK | 0.8922 |
| MAE | 0.2077 |
| Mild DR recall (class 1) | 0.7333 |

## IDRiD external (unseen)
| Metric | Value |
|--------|------:|
| Accuracy | 0.5155 |
| Macro-F1 | 0.4001 |
| QWK | 0.6687 |
| MAE | 0.6531 |
| Mild DR recall (class 1) | 0.5200 |

## Artifact locations (Google Drive / Colab project)
```
/content/drive/MyDrive/DR-Research/project/checkpoints/baseline_efficientnet_b0_rgb_20260916_180945/best.pt
/content/drive/MyDrive/DR-Research/project/results/baseline_efficientnet_b0_rgb_20260916_180945/
/content/drive/MyDrive/DR-Research/project/experiments/baseline_efficientnet_b0_rgb_20260916_180945/
```

## Notes
- IDRiD was never used for training or validation/model selection.
- No Green+CLAHE / Swin / hybrid / Focal / CORAL in this run.
