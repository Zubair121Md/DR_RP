"""
Dataset loaders for APTOS, IDRiD, Messidor-2, and DeepDRiD (5-class ICDR 0–4).

Read-only: never modifies dataset files on disk.
External sets (IDRiD / Messidor-2 / DeepDRiD) are eval-only — never train on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DATASET_REGISTRY = {
    "aptos": "APTOSDataset",
    "idrid": "IDRiDDataset",
    "messidor2": "Messidor2Dataset",
    "messidor": "Messidor2Dataset",
    "deepdrid": "DeepDRiDDataset",
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG")


def _resolve_image(folder: Path, image_id: str) -> Path:
    """Resolve image path by stem; try common extensions (case-insensitive)."""
    stem = Path(image_id).stem if Path(image_id).suffix else image_id
    # Exact / common extensions first
    for ext in IMAGE_EXTS:
        candidate = folder / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    direct = folder / image_id
    if direct.is_file():
        return direct
    # Case-insensitive scan (Messidor-2: CSV .jpg vs disk .JPG)
    target = stem.lower()
    if folder.is_dir():
        for p in folder.iterdir():
            if p.is_file() and p.stem.lower() == target:
                return p
    raise FileNotFoundError(f"No image for id={image_id!r} under {folder}")


class APTOSDataset(Dataset):
    """APTOS 2019 DR grading dataset (train / val / test)."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Any = None,
        train_csv: str = "train_1.csv",
        val_csv: str = "valid.csv",
        test_csv: str = "test.csv",
        train_images: str = "train_images/train_images",
        val_images: str = "val_images/val_images",
        test_images: str = "test_images/test_images",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.root = Path(root)
        self.split = split.lower().strip()
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None

        split_map = {
            "train": (train_csv, train_images),
            "val": (val_csv, val_images),
            "valid": (val_csv, val_images),
            "validation": (val_csv, val_images),
            "test": (test_csv, test_images),
        }
        if self.split not in split_map:
            raise ValueError(f"APTOS split must be one of {list(split_map)}, got {split!r}")

        csv_rel, img_rel = split_map[self.split]
        csv_path = self.root / csv_rel
        self.image_dir = self.root / img_rel
        if not csv_path.is_file():
            raise FileNotFoundError(f"APTOS labels not found: {csv_path}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"APTOS image folder not found: {self.image_dir}")

        df = pd.read_csv(csv_path)
        df = df.dropna(how="all")
        cols = {c.lower().strip(): c for c in df.columns}
        if "id_code" not in cols or "diagnosis" not in cols:
            raise ValueError(f"APTOS CSV requires id_code and diagnosis; found {list(df.columns)}")

        id_col = cols["id_code"]
        diag_col = cols["diagnosis"]
        self.samples: list[tuple[str, int, Path]] = []
        for _, row in df.iterrows():
            image_id = str(row[id_col]).strip()
            if image_id in {"", "nan", "None"}:
                continue
            label = int(row[diag_col])
            if label not in range(5):
                raise ValueError(f"APTOS label out of 0–4: {label} for {image_id}")
            path = _resolve_image(self.image_dir, image_id)
            self.samples.append((image_id, label, path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id, label, path = self.samples[index]
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{image_id}.png"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing preprocess cache frame: {cache_path}")
            with Image.open(cache_path) as img:
                image = img.convert("RGB")
            load_path = cache_path
        else:
            with Image.open(path) as img:
                image = img.convert("RGB")
            load_path = path
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": label,
            "id": image_id,
            "dataset": "aptos",
            "split": self.split,
            "path": str(load_path),
        }


class IDRiDDataset(Dataset):
    """
    IDRiD Disease Grading subset for external testing only.

    Train/test splits reuse ID strings in different folders — always key by
    (split, image_id). Never use this dataset for training or validation.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        transform: Any = None,
        train_csv: str = "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
        test_csv: str = "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
        train_images: str = "1. Original Images/a. Training Set",
        test_images: str = "1. Original Images/b. Testing Set",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.root = Path(root)
        # Accept either Disease Grading root or idrid package root
        if (self.root / "B. Disease Grading").is_dir():
            self.root = self.root / "B. Disease Grading"
        self.split = split.lower().strip()
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None

        parts: list[tuple[str, str, str]] = []
        if self.split in {"train", "training", "all"}:
            parts.append(("training", train_csv, train_images))
        if self.split in {"test", "testing", "all"}:
            parts.append(("testing", test_csv, test_images))
        if not parts:
            raise ValueError(
                f"IDRiD split must be 'training', 'testing', or 'all'; got {split!r}"
            )

        self.samples: list[tuple[str, str, int, Path]] = []
        for part_name, csv_rel, img_rel in parts:
            csv_path = self.root / csv_rel
            image_dir = self.root / img_rel
            if not csv_path.is_file():
                raise FileNotFoundError(f"IDRiD labels not found: {csv_path}")
            if not image_dir.is_dir():
                raise FileNotFoundError(f"IDRiD image folder not found: {image_dir}")

            df = pd.read_csv(csv_path)
            df = df.dropna(how="all")
            df.columns = [str(c).strip() for c in df.columns]
            cols = {c.lower(): c for c in df.columns}

            image_col = None
            for key in ("image name", "image_name", "id_code", "image"):
                if key in cols:
                    image_col = cols[key]
                    break
            grade_col = None
            for key in ("retinopathy grade", "diagnosis", "grade"):
                if key in cols:
                    grade_col = cols[key]
                    break
            if image_col is None or grade_col is None:
                raise ValueError(
                    f"IDRiD CSV requires image name and retinopathy grade; found {list(df.columns)}"
                )

            for _, row in df.iterrows():
                raw_name = str(row[image_col]).strip()
                if raw_name in {"", "nan", "None"}:
                    continue
                image_id = Path(raw_name).stem
                label = int(row[grade_col])
                if label not in range(5):
                    raise ValueError(f"IDRiD label out of 0–4: {label} for {image_id}")
                path = _resolve_image(image_dir, image_id)
                self.samples.append((part_name, image_id, label, path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        part_name, image_id, label, path = self.samples[index]
        if self.cache_dir is not None:
            cache_path = self.cache_dir / part_name / f"{image_id}.png"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing preprocess cache frame: {cache_path}")
            with Image.open(cache_path) as img:
                image = img.convert("RGB")
            load_path = cache_path
        else:
            with Image.open(path) as img:
                image = img.convert("RGB")
            load_path = path
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": label,
            "id": image_id,
            "dataset": "idrid",
            "split": part_name,
            "path": str(load_path),
        }


class Messidor2Dataset(Dataset):
    """
    Messidor-2 (Google Brain adjudicated ICDR 0–4) — external test only.

    Uses messidor_data.csv (image_id, adjudicated_dr_grade, adjudicated_gradable).
    Drops ungradable rows. Never use for training/validation.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        transform: Any = None,
        labels_csv: str = "messidor_data.csv",
        images_dir: str = "IMAGES",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs  # allow extra config keys
        self.root = Path(root)
        self.split = split.lower().strip()
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None

        csv_path = self.root / labels_csv
        self.image_dir = self.root / images_dir
        if not csv_path.is_file():
            raise FileNotFoundError(f"Messidor-2 labels not found: {csv_path}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Messidor-2 image folder not found: {self.image_dir}")

        df = pd.read_csv(csv_path)
        df = df.dropna(how="all")
        cols = {c.lower().strip(): c for c in df.columns}
        id_col = cols.get("image_id") or cols.get("image") or cols.get("id_code")
        grade_col = cols.get("adjudicated_dr_grade") or cols.get("diagnosis") or cols.get("grade")
        gradable_col = cols.get("adjudicated_gradable")
        if id_col is None or grade_col is None:
            raise ValueError(
                f"Messidor-2 CSV needs image_id + adjudicated_dr_grade; found {list(df.columns)}"
            )

        self.samples: list[tuple[str, int, Path]] = []
        for _, row in df.iterrows():
            if gradable_col is not None:
                try:
                    if int(row[gradable_col]) != 1:
                        continue
                except (TypeError, ValueError):
                    continue
            raw_grade = row[grade_col]
            if pd.isna(raw_grade):
                continue
            label = int(raw_grade)
            if label not in range(5):
                raise ValueError(f"Messidor-2 label out of 0–4: {label}")
            raw_id = str(row[id_col]).strip()
            if raw_id in {"", "nan", "None"}:
                continue
            image_id = Path(raw_id).stem
            path = _resolve_image(self.image_dir, raw_id)
            self.samples.append((image_id, label, path))

        if not self.samples:
            raise RuntimeError(f"Messidor-2: no gradable samples under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id, label, path = self.samples[index]
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{image_id}.png"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing preprocess cache frame: {cache_path}")
            with Image.open(cache_path) as img:
                image = img.convert("RGB")
            load_path = cache_path
        else:
            with Image.open(path) as img:
                image = img.convert("RGB")
            load_path = path
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": label,
            "id": image_id,
            "dataset": "messidor2",
            "split": "all",
            "path": str(load_path),
        }


class DeepDRiDDataset(Dataset):
    """
    DeepDRiD regular-fundus disease grading (ICDR 0–4) — external test only.

    Uses left_eye_DR_Level for *_l* images and right_eye_DR_Level for *_r*.
    Default: training + validation graded regular fundus (challenge eval has no labels).
    Never use for training/validation of APTOS models.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        transform: Any = None,
        cache_dir: str | Path | None = None,
        subsets: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.root = Path(root)
        self.split = split.lower().strip()
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None

        regular = self.root / "regular_fundus_images"
        if not regular.is_dir():
            raise FileNotFoundError(f"DeepDRiD regular_fundus_images missing under {self.root}")

        if subsets is None:
            if self.split in {"train", "training"}:
                subsets = ["regular-fundus-training"]
            elif self.split in {"val", "valid", "validation"}:
                subsets = ["regular-fundus-validation"]
            else:
                subsets = ["regular-fundus-training", "regular-fundus-validation"]

        self.samples: list[tuple[str, str, int, Path]] = []
        for subset in subsets:
            sub_dir = regular / subset
            csv_path = sub_dir / f"{subset}.csv"
            images_root = sub_dir / "Images"
            if not csv_path.is_file():
                raise FileNotFoundError(f"DeepDRiD labels not found: {csv_path}")
            if not images_root.is_dir():
                raise FileNotFoundError(f"DeepDRiD images not found: {images_root}")

            df = pd.read_csv(csv_path)
            df = df.dropna(how="all")
            cols = {c.lower().strip(): c for c in df.columns}
            id_col = cols.get("image_id")
            left_col = cols.get("left_eye_dr_level")
            right_col = cols.get("right_eye_dr_level")
            if id_col is None or left_col is None or right_col is None:
                raise ValueError(
                    f"DeepDRiD CSV needs image_id + left/right_eye_DR_Level; found {list(df.columns)}"
                )

            for _, row in df.iterrows():
                image_id = str(row[id_col]).strip()
                if image_id in {"", "nan", "None"}:
                    continue
                eye = image_id.split("_")[-1][:1].lower() if "_" in image_id else ""
                raw_grade = row[left_col] if eye == "l" else row[right_col] if eye == "r" else None
                if raw_grade is None or pd.isna(raw_grade):
                    continue
                label = int(raw_grade)
                if label not in range(5):
                    continue
                # Paths like Images/<patient_id>/<image_id>.jpg
                patient = image_id.split("_")[0]
                path = images_root / patient / f"{image_id}.jpg"
                if not path.is_file():
                    path = _resolve_image(images_root / patient, image_id)
                self.samples.append((subset, image_id, label, path))

        if not self.samples:
            raise RuntimeError(f"DeepDRiD: no graded samples under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        part_name, image_id, label, path = self.samples[index]
        if self.cache_dir is not None:
            cache_path = self.cache_dir / part_name / f"{image_id}.png"
            if not cache_path.is_file():
                cache_path = self.cache_dir / f"{image_id}.png"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing preprocess cache frame: {cache_path}")
            with Image.open(cache_path) as img:
                image = img.convert("RGB")
            load_path = cache_path
        else:
            with Image.open(path) as img:
                image = img.convert("RGB")
            load_path = path
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": label,
            "id": image_id,
            "dataset": "deepdrid",
            "split": part_name,
            "path": str(load_path),
        }


def build_dataset(name: str, **kwargs: Any) -> Dataset:
    """Factory: build a dataset by config name."""
    key = name.lower().strip()
    if key == "aptos":
        return APTOSDataset(**kwargs)
    if key == "idrid":
        return IDRiDDataset(**kwargs)
    if key in {"messidor2", "messidor", "messidor_2"}:
        return Messidor2Dataset(**kwargs)
    if key in {"deepdrid", "deep_drid", "deep-drid"}:
        return DeepDRiDDataset(**kwargs)
    raise ValueError(f"Unknown dataset {name!r}. Known: {list(DATASET_REGISTRY)}")


def collate_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import torch

    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {
        "image": images,
        "label": labels,
        "id": [b["id"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
        "split": [b["split"] for b in batch],
        "path": [b["path"] for b in batch],
    }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool | None = None,
    **kwargs: Any,
) -> DataLoader:
    """Wrap a dataset in a DataLoader. pin_memory only helps on CUDA."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
        **kwargs,
    )
