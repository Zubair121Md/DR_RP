#!/usr/bin/env python3
"""
PHASE D — Visualize preprocessing for all datasets.

Saves samples under experiments/results/preprocessing_diagnostics/
Does not modify preprocessing code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[2]
import sys as _sys
_sys.path.insert(0, str(PROJECT))
from src.restricted import CHECKPOINTS, require_or_skip  # noqa: E402
sys.path.insert(0, str(PROJECT))

from src.dataset import build_dataset  # noqa: E402
from src.preprocessing import apply_retina_frontend, make_cached_frame  # noqa: E402
from src.utils import load_config, resolve_data_roots  # noqa: E402


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _panel(orig: Image.Image, rgb224: Image.Image, g224: Image.Image, title: str) -> Image.Image:
    w, h = 224, 224
    orig_r = orig.resize((w, h))
    canvas = Image.new("RGB", (w * 3 + 40, h + 50), (255, 255, 255))
    canvas.paste(orig_r, (10, 30))
    canvas.paste(rgb224, (20 + w, 30))
    canvas.paste(g224, (30 + 2 * w, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 5), f"{title} | orig→224 | rgb224 | green_clahe224", fill=(0, 0, 0))
    return canvas


def main() -> None:
    if not require_or_skip("preprocessing diagnostics (experiments/results/preprocessing_diagnostics/)", "APTOS", "IDRiD", "Messidor-2", "DeepDRiD"):
        return
    cfg = load_config(PROJECT / "configs/baseline_efficientnet_b0_green_clahe_mild_coral_local.yaml")
    roots = resolve_data_roots(cfg, PROJECT)
    out_root = PROJECT / "experiments" / "results" / "preprocessing_diagnostics"
    out_root.mkdir(parents=True, exist_ok=True)

    specs = [
        ("aptos_train", "aptos", {"root": roots["aptos"], "split": "train"}),
        ("aptos_val", "aptos", {"root": roots["aptos"], "split": "val"}),
        ("aptos_test", "aptos", {"root": roots["aptos"], "split": "test"}),
        ("idrid", "idrid", {"root": roots["idrid"], "split": "all"}),
        ("messidor2", "messidor2", {"root": roots["messidor2"], "split": "all"}),
        ("deepdrid", "deepdrid", {"root": roots["deepdrid"], "split": "all"}),
    ]

    manifest = {"samples": [], "notes": []}
    # Document FOV gap
    manifest["notes"].append(
        "P1 green_clahe has NO retinal FOV crop in src/preprocessing.py "
        "(protocol text mentions FOV crop; implementation does green+CLAHE only)."
    )
    manifest["notes"].append("Eval pipelines use train=False (no random augs).")
    manifest["notes"].append("Channel order: PIL RGB; green = channel index 1.")

    for tag, name, kwargs in specs:
        try:
            ds = build_dataset(name, transform=None, **kwargs)
        except Exception as exc:
            manifest["samples"].append({"tag": tag, "error": str(exc)})
            continue
        # pick one mild if possible else first
        idx = 0
        for i in range(min(len(ds), 200)):
            if int(ds[i]["label"]) == 1:
                idx = i
                break
        sample = ds[idx]
        path = Path(sample["path"])
        # If cached path, still fine — it's RGB PNG
        orig = _load_rgb(path)
        # For cache frames, frontend already applied — load raw when possible
        raw_path = path
        if "cache" in str(path):
            # fall back: use dataset sample path stored; DeepDRiD/APTOS raw preferred
            raw_path = path
        rgb224 = make_cached_frame("rgb", orig, 224)
        g224 = make_cached_frame("green_clahe", orig, 224)
        # Also show frontend-only green without resize for documentation
        g_front = apply_retina_frontend("green_clahe", orig)

        panel = _panel(orig, rgb224, g224, f"{tag} id={sample['id']} y={sample['label']}")
        out = out_root / f"{tag}_{sample['id']}_y{sample['label']}.png"
        panel.save(out)
        g_front.save(out_root / f"{tag}_{sample['id']}_green_frontend.png")

        # Numeric sanity
        arr = np.asarray(g224)
        manifest["samples"].append(
            {
                "tag": tag,
                "id": sample["id"],
                "label": int(sample["label"]),
                "path": str(path),
                "panel": str(out.relative_to(PROJECT)),
                "green_224_shape": list(arr.shape),
                "green_channels_equal": bool(
                    np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 1], arr[:, :, 2])
                ),
                "n_dataset": len(ds),
            }
        )
        print("WROTE", out)

    man_path = out_root / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print("MANIFEST", man_path)


if __name__ == "__main__":
    main()
