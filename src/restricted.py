"""
Guards for outputs that contain dataset fundus pixels (example images, preprocessing panels, augmentation draws,
Grad-CAM overlays, and the manuscript .docx files that embed them).

These outputs are generated locally only and are excluded from the repository by .gitignore, because the dataset
licences do not all permit redistribution of images. Scripts that build them call `require_or_skip` and print a
SKIP message instead of failing when the datasets or checkpoints are not available.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIRS = {
    "APTOS": ROOT / "datasets" / "archive",
    "IDRiD": ROOT / "datasets" / "idrid",
    "Messidor-2": ROOT / "Messidor ",
    "DeepDRiD": ROOT / "datasets" / "DeepDRiD",
}
CHECKPOINTS = ROOT / "checkpoints"

RESTRICTED_FIGURES = {
    "figures/main/fig02_preprocess.png", "figures/supp/figS04_augmentation.png",
    "figures/supp/figS06_dataset_examples.png", "figures/supp/figS19_gradcam_mild.png",
}


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def require_or_skip(figure: str, *datasets: str, extra: tuple[Path, ...] = ()) -> bool:
    """Return True if every dataset folder and extra path exists; otherwise print a SKIP message and return False."""
    missing = [f"{d} images ({_rel(DATASET_DIRS[d])})" for d in datasets if not DATASET_DIRS[d].is_dir()]
    missing += [_rel(p) for p in extra if not p.exists()]
    if missing:
        print(f"SKIP {figure}: it contains dataset fundus pixels and needs files that are not distributed with this "
              f"repository (missing: {'; '.join(missing)}). Obtain the datasets as described in DATASET_PROVENANCE.md "
              f"to regenerate it locally; the output is git-ignored.")
        return False
    return True
