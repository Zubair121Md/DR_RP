# Grad-CAM index (Phase N)

Target layer: MobileViT `final_conv` (7×7).
Score: predicted-class probability (CE or CORAL).
Sample IDs fixed from `hybrid_h2_s42_v1` predictions (fair cross-run compare).

## Selected IDs

### aptos_test
- grade 0: `e4dcca36ceb4`
- grade 1: `ead23cc922ed`
- grade 2: `e4e343eaae2a`
- grade 3: `e93394175a19`
- grade 4: `eadc57064154`

### idrid_external
- grade 0: `IDRiD_138`
- grade 1: `IDRiD_021`
- grade 2: `IDRiD_003`
- grade 3: `IDRiD_001`
- grade 4: `IDRiD_104`

## Outputs

- Per-run panels: `paper/figures/gradcam/hybrid_h*_s*_v1/`
- Grade strips: `paper/figures/gradcam/grids/`
- Cross-run compare: `paper/figures/gradcam/compare/`
- Manifest: `paper/figures/gradcam/manifest.json`

Generated **120** overlays across **12** runs.

GRADCAM_ALL_OK
