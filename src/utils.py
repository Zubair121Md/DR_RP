"""Shared helpers for config, seeding, and experiment directories."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str | None = None) -> torch.device:
    """Pick accelerator. preferred may be 'cpu' | 'mps' | 'cuda' to override auto."""
    if preferred:
        key = str(preferred).lower().strip()
        if key == "cpu":
            return torch.device("cpu")
        if key == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            raise RuntimeError("train.device=cuda requested but CUDA is unavailable")
        if key == "mps":
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return torch.device("mps")
            raise RuntimeError("train.device=mps requested but MPS is unavailable")
        raise ValueError(f"Unsupported train.device: {preferred}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_id(experiment_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{experiment_name}_{stamp}"


def ensure_run_dirs(project_root: Path, run_id: str) -> dict[str, Path]:
    """Create unique run directories so previous results are never overwritten."""
    result_dir = project_root / "results" / run_id
    ckpt_dir = project_root / "checkpoints" / run_id
    exp_dir = project_root / "experiments" / run_id
    for d in (result_dir, ckpt_dir, exp_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {"results": result_dir, "checkpoints": ckpt_dir, "experiments": exp_dir}


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def resolve_data_roots(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    """
    Resolve APTOS / IDRiD / Messidor-2 / DeepDRiD roots.

    Prefer explicit absolute paths in config; otherwise join project_root.
    Optional config key data.datasets_root prefixes relative dataset roots.
    """
    data = config["data"]
    datasets_root = data.get("datasets_root")
    if datasets_root:
        base = Path(datasets_root)
        if not base.is_absolute():
            base = project_root / base
    else:
        base = project_root

    def _root(block: dict[str, Any] | None, default_rel: str) -> Path:
        if not block:
            path = Path(default_rel)
        else:
            path = Path(block.get("root", default_rel))
        if path.is_absolute():
            return path
        if datasets_root:
            return base / path
        return project_root / path

    # Messidor folder may have a trailing space from Finder rename; YAML may strip it
    def _messidor_root() -> Path:
        block = data.get("messidor2") or data.get("messidor")
        candidates: list[Path] = []
        if block and block.get("root"):
            candidates.append(_root(block, str(block["root"])))
        candidates.extend(
            [
                project_root / "Messidor ",
                project_root / "Messidor",
                base / "Messidor ",
                base / "Messidor",
            ]
        )
        for c in candidates:
            try:
                resolved = c.resolve()
            except Exception:
                resolved = c
            if resolved.is_dir() and (resolved / "IMAGES").is_dir():
                return resolved
            # Trailing-space directory that Path.resolve may normalize away
            if c.exists() and c.is_dir():
                return c
        # Last resort: scan project_root for Messidor*
        for p in project_root.iterdir():
            if p.is_dir() and p.name.strip().lower() == "messidor" and (p / "IMAGES").is_dir():
                return p
        return candidates[0] if candidates else project_root / "Messidor"

    out = {
        "aptos": _root(data.get("aptos"), "datasets/archive"),
        "idrid": _root(data.get("idrid"), "datasets/idrid/B. Disease Grading"),
        "messidor2": _messidor_root(),
        "deepdrid": _root(data.get("deepdrid"), "datasets/DeepDRiD"),
    }
    return out


def resolve_cache_root(config: dict[str, Any], project_root: Path) -> Path | None:
    """
    Optional preprocess cache root from preprocessing.cache_root.

    Expected layout:
      <cache_root>/<preprocess>/aptos/{train,val,test}/
      <cache_root>/<preprocess>/idrid/{training,testing}/
      <cache_root>/<preprocess>/messidor2/
      <cache_root>/<preprocess>/deepdrid/
    """
    raw = config.get("preprocessing", {}).get("cache_root")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return path


def aptos_cache_dir(cache_root: Path | None, preprocess: str, split: str) -> Path | None:
    if cache_root is None:
        return None
    key = split.lower().strip()
    if key in {"val", "valid", "validation"}:
        key = "val"
    return cache_root / preprocess / "aptos" / key


def idrid_cache_dir(cache_root: Path | None, preprocess: str) -> Path | None:
    if cache_root is None:
        return None
    return cache_root / preprocess / "idrid"


def messidor2_cache_dir(cache_root: Path | None, preprocess: str) -> Path | None:
    if cache_root is None:
        return None
    return cache_root / preprocess / "messidor2"


def deepdrid_cache_dir(cache_root: Path | None, preprocess: str) -> Path | None:
    if cache_root is None:
        return None
    return cache_root / preprocess / "deepdrid"
