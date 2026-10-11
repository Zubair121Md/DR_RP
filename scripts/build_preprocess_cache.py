"""
Build lossless 224×224 preprocess cache frames.

Caches only the deterministic prefix (retina frontend + Resize), matching
online training/eval exactly. Stochastic augs stay online.

Layout:
  <cache_root>/<preprocess>/aptos/{train,val,test}/<id>.png
  <cache_root>/<preprocess>/idrid/{training,testing}/<id>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    APTOSDataset,
    DeepDRiDDataset,
    IDRiDDataset,
    Messidor2Dataset,
)
from src.preprocessing import make_cached_frame  # noqa: E402
from src.utils import load_config, resolve_data_roots  # noqa: E402


def _write_frame(out: Path, frame: Image.Image) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # PNG is lossless — bit-identical reload of the Resize(224) frontend output
    frame.save(out, format="PNG", compress_level=1)


def build_aptos_split(
    ds: APTOSDataset,
    out_dir: Path,
    preprocess: str,
    image_size: int,
    clahe_clip: float,
    clahe_tiles: tuple[int, int],
) -> int:
    n = 0
    for image_id, _label, path in tqdm(ds.samples, desc=f"aptos/{ds.split}/{preprocess}"):
        out = out_dir / f"{image_id}.png"
        if out.is_file():
            n += 1
            continue
        with Image.open(path) as img:
            frame = make_cached_frame(
                preprocess,
                img,
                image_size=image_size,
                clahe_clip_limit=clahe_clip,
                clahe_tile_grid_size=clahe_tiles,
            )
        _write_frame(out, frame)
        n += 1
    return n


def build_idrid_split(
    ds: IDRiDDataset,
    out_root: Path,
    preprocess: str,
    image_size: int,
    clahe_clip: float,
    clahe_tiles: tuple[int, int],
) -> int:
    n = 0
    for part_name, image_id, _label, path in tqdm(
        ds.samples, desc=f"idrid/{preprocess}"
    ):
        out = out_root / part_name / f"{image_id}.png"
        if out.is_file():
            n += 1
            continue
        with Image.open(path) as img:
            frame = make_cached_frame(
                preprocess,
                img,
                image_size=image_size,
                clahe_clip_limit=clahe_clip,
                clahe_tile_grid_size=clahe_tiles,
            )
        _write_frame(out, frame)
        n += 1
    return n


def build_messidor2_split(
    ds: Messidor2Dataset,
    out_dir: Path,
    preprocess: str,
    image_size: int,
    clahe_clip: float,
    clahe_tiles: tuple[int, int],
) -> int:
    n = 0
    for image_id, _label, path in tqdm(ds.samples, desc=f"messidor2/{preprocess}"):
        out = out_dir / f"{image_id}.png"
        if out.is_file():
            n += 1
            continue
        with Image.open(path) as img:
            frame = make_cached_frame(
                preprocess,
                img,
                image_size=image_size,
                clahe_clip_limit=clahe_clip,
                clahe_tile_grid_size=clahe_tiles,
            )
        _write_frame(out, frame)
        n += 1
    return n


def build_deepdrid_split(
    ds: DeepDRiDDataset,
    out_root: Path,
    preprocess: str,
    image_size: int,
    clahe_clip: float,
    clahe_tiles: tuple[int, int],
) -> int:
    """Layout: <cache>/deepdrid/<subset>/<image_id>.png (matches dataset.__getitem__)."""
    n = 0
    for part_name, image_id, _label, path in tqdm(ds.samples, desc=f"deepdrid/{preprocess}"):
        out = out_root / part_name / f"{image_id}.png"
        if out.is_file():
            n += 1
            continue
        with Image.open(path) as img:
            frame = make_cached_frame(
                preprocess,
                img,
                image_size=image_size,
                clahe_clip_limit=clahe_clip,
                clahe_tile_grid_size=clahe_tiles,
            )
        _write_frame(out, frame)
        n += 1
    return n


def verify_bit_identical(
    ds: APTOSDataset,
    cache_dir: Path,
    preprocess: str,
    image_size: int,
    clahe_clip: float,
    clahe_tiles: tuple[int, int],
    n_check: int = 8,
) -> None:
    import numpy as np

    for i, (image_id, _label, path) in enumerate(ds.samples[:n_check]):
        with Image.open(path) as img:
            expected = make_cached_frame(
                preprocess,
                img,
                image_size=image_size,
                clahe_clip_limit=clahe_clip,
                clahe_tile_grid_size=clahe_tiles,
            )
        cached = Image.open(cache_dir / f"{image_id}.png").convert("RGB")
        a = np.asarray(expected)
        b = np.asarray(cached)
        if a.shape != b.shape or not np.array_equal(a, b):
            raise RuntimeError(
                f"Cache mismatch for {image_id}: shape {a.shape}/{b.shape}, "
                f"diff={(a.astype(int) - b.astype(int)).max() if a.shape == b.shape else 'n/a'}"
            )
    print(f"BIT_IDENTICAL_OK preprocess={preprocess} checked={min(n_check, len(ds))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DR preprocess cache.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_efficientnet_b0_rgb_local.yaml",
        help="Config used only for dataset paths / CLAHE params / image_size.",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        nargs="+",
        default=["rgb", "green_clahe"],
        help="Preprocess names to cache.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="datasets/cache",
        help="Cache root relative to project (or absolute).",
    )
    parser.add_argument("--verify", action="store_true", help="Bit-identical check after build.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    roots = resolve_data_roots(config, PROJECT_ROOT)
    aptos_cfg = {k: v for k, v in config["data"]["aptos"].items() if k != "root"}
    idrid_cfg = {k: v for k, v in config["data"]["idrid"].items() if k != "root"}
    prep = config["preprocessing"]
    image_size = int(prep.get("image_size", 224))
    clahe_clip = float(prep.get("clahe_clip_limit", 2.0))
    tiles = prep.get("clahe_tile_grid_size", [8, 8])
    clahe_tiles = (int(tiles[0]), int(tiles[1]))

    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = PROJECT_ROOT / cache_root

    for name in args.preprocess:
        print(f"=== Building cache: {name} → {cache_root / name} ===")
        for split in ("train", "val", "test"):
            ds = APTOSDataset(
                root=roots["aptos"], split=split, transform=None, **aptos_cfg
            )
            out_dir = cache_root / name / "aptos" / ("val" if split in {"val", "valid", "validation"} else split)
            n = build_aptos_split(ds, out_dir, name, image_size, clahe_clip, clahe_tiles)
            print(f"  aptos/{split}: {n} frames")

        idrid = IDRiDDataset(
            root=roots["idrid"], split="all", transform=None, **idrid_cfg
        )
        n = build_idrid_split(
            idrid,
            cache_root / name / "idrid",
            name,
            image_size,
            clahe_clip,
            clahe_tiles,
        )
        print(f"  idrid/all: {n} frames")

        messidor_root = roots.get("messidor2")
        if messidor_root is not None and messidor_root.is_dir():
            messidor_cfg = {
                k: v
                for k, v in (config["data"].get("messidor2") or config["data"].get("messidor") or {}).items()
                if k != "root"
            }
            try:
                messidor = Messidor2Dataset(
                    root=messidor_root, split="all", transform=None, **messidor_cfg
                )
                n = build_messidor2_split(
                    messidor,
                    cache_root / name / "messidor2",
                    name,
                    image_size,
                    clahe_clip,
                    clahe_tiles,
                )
                print(f"  messidor2/all: {n} frames")
            except Exception as exc:
                print(f"  messidor2 skipped: {exc}")

        deep_root = roots.get("deepdrid")
        if deep_root is not None and deep_root.is_dir():
            try:
                deep = DeepDRiDDataset(root=deep_root, split="all", transform=None)
                n = build_deepdrid_split(
                    deep,
                    cache_root / name / "deepdrid",
                    name,
                    image_size,
                    clahe_clip,
                    clahe_tiles,
                )
                print(f"  deepdrid/all: {n} frames")
            except Exception as exc:
                print(f"  deepdrid skipped: {exc}")

        if args.verify:
            train_ds = APTOSDataset(
                root=roots["aptos"], split="train", transform=None, **aptos_cfg
            )
            verify_bit_identical(
                train_ds,
                cache_root / name / "aptos" / "train",
                name,
                image_size,
                clahe_clip,
                clahe_tiles,
            )

    print("CACHE_BUILD_DONE", cache_root)


if __name__ == "__main__":
    main()
