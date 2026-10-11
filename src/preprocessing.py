"""
Image preprocessing pipelines for DR grading.

Supported:
- rgb         (Experiment 1 / Exp4)
- green_clahe (Experiment 2 / Exp3): green channel + CLAHE → 3-ch stack

Optional disk cache of the *deterministic* prefix (front-end + Resize 224)
keeps the scientific transform identical while cutting wall-clock under
deployable compute constraints. Stochastic augs stay online.
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

PREPROCESS_REGISTRY = {
    "rgb": "RGBPreprocess",
    "green_clahe": "GreenCLAHEPreprocess",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _aug_and_normalize(
    train: bool,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> T.Compose:
    """Stochastic augs (train only) + ToTensor + ImageNet normalize. No resize."""
    ops: list[Any] = []
    if train:
        ops.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )
    ops.extend([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return T.Compose(ops)


def _build_tensor_pipeline(
    image_size: int,
    train: bool,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> T.Compose:
    """Full online pipeline: Resize → (augs) → normalize."""
    ops: list[Any] = [T.Resize((image_size, image_size))]
    ops.extend(list(_aug_and_normalize(train, mean, std).transforms))
    return T.Compose(ops)


def apply_retina_frontend(
    name: str,
    image: Any,
    *,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
) -> Image.Image:
    """
    Deterministic retina-specific frontend (no resize, no augs).

    rgb: RGB convert only.
    green_clahe: green channel → CLAHE → 3-channel stack.
    """
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    key = name.lower().strip()
    if key == "rgb":
        return image  # type: ignore[return-value]
    if key in {"green_clahe", "green+clahe", "green"}:
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {getattr(rgb, 'shape', None)}")
        green = rgb[:, :, 1]
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=tuple(clahe_tile_grid_size),
        )
        enhanced = clahe.apply(green)
        stacked = np.stack([enhanced, enhanced, enhanced], axis=-1)
        return Image.fromarray(stacked)
    raise ValueError(f"Unknown preprocess {name!r}. Known: {list(PREPROCESS_REGISTRY)}")


def make_cached_frame(
    name: str,
    image: Any,
    image_size: int = 224,
    *,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
) -> Image.Image:
    """Deterministic frontend + Resize(image_size) — what we store on disk."""
    front = apply_retina_frontend(
        name,
        image,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
    )
    return T.Resize((image_size, image_size))(front)


class CachedPreprocess:
    """
    Augment + normalize a pre-resized 224 cache frame.

    Cache already contains retina frontend + resize, so only augs differ by train flag.
    """

    def __init__(
        self,
        image_size: int = 224,
        train: bool = False,
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
        **kwargs: Any,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.transform = _aug_and_normalize(train, mean, std)

    def __call__(self, image: Any) -> Any:
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        return self.transform(image)


class RGBPreprocess:
    """RGB resize + ImageNet normalization for pretrained EfficientNet-B0."""

    def __init__(
        self,
        image_size: int = 224,
        train: bool = False,
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
        **kwargs: Any,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.transform = _build_tensor_pipeline(image_size, train, mean, std)

    def __call__(self, image: Any) -> Any:
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        return self.transform(image)


class GreenCLAHEPreprocess:
    """
    Green-channel + CLAHE preprocessing.

    Steps:
      1. Convert to RGB numpy array
      2. Extract green channel
      3. Apply CLAHE
      4. Stack enhanced green into 3 channels
      5. Same resize / augment / ImageNet normalize as RGB baseline
    """

    def __init__(
        self,
        image_size: int = 224,
        train: bool = False,
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
        **kwargs: Any,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid_size = tuple(clahe_tile_grid_size)
        self.transform = _build_tensor_pipeline(image_size, train, mean, std)

    def __call__(self, image: Any) -> Any:
        pil_img = apply_retina_frontend(
            "green_clahe",
            image,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_grid_size=self.clahe_tile_grid_size,
        )
        return self.transform(pil_img)


def build_preprocess(name: str, **kwargs: Any) -> Callable[..., Any]:
    """Factory: build a preprocessing transform by config name."""
    if kwargs.pop("cached", False):
        return CachedPreprocess(**kwargs)
    key = name.lower().strip()
    if key == "rgb":
        return RGBPreprocess(**kwargs)
    if key in {"green_clahe", "green+clahe", "green"}:
        return GreenCLAHEPreprocess(**kwargs)
    raise ValueError(f"Unknown preprocess {name!r}. Known: {list(PREPROCESS_REGISTRY)}")
