#!/usr/bin/env python3
"""Phase N: Grad-CAM for all hybrid H1–H4 × seeds 42/43/44.

Uses last MobileViT spatial map (final_conv, 7×7).
Score for backprop:
  CE: predicted-class logit
  CORAL: sum of the four ordinal logits (documented; no class index exists)

Outputs under paper/figures/gradcam/:
  {run_id}/{split}_g{grade}_{id}.png     — single overlays
  grids/{run_id}_{split}_by_grade.png    — 1×5 grade strip per run
  compare/{split}_grade{g}_all_runs.png  — same image across all 12 runs
  INDEX.md                               — manifest
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.dataset import build_dataset
from src.models import build_model, decode_predictions, logits_to_probs
from src.preprocessing import build_preprocess
from src.utils import (
    aptos_cache_dir,
    get_device,
    idrid_cache_dir,
    load_config,
    resolve_cache_root,
    resolve_data_roots,
)

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(ROOT))
from src.restricted import CHECKPOINTS, require_or_skip  # noqa: E402
OUT = ROOT / "paper" / "figures" / "gradcam"
SEEDS = (42, 43, 44)
CELLS = ("H1", "H2", "H3", "H4")
SPLITS = (
    ("aptos_test", "aptos", "test"),
    ("idrid_external", "idrid", "all"),
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def run_id(cell: str, seed: int) -> str:
    return f"hybrid_{cell.lower()}_s{seed}_v1"


def resolve_head(cfg: dict) -> str:
    loss = str(cfg.get("loss", {}).get("name", "cross_entropy")).lower()
    head = str(cfg.get("model", {}).get("head", "ce")).lower()
    if loss in {"coral", "mild_aware_coral", "mild-aware-coral", "mild_coral"}:
        return "coral"
    if head in {"coral", "ordinal", "mild_aware_coral"}:
        return "coral"
    return "ce"


def pick_target_layer(model: nn.Module) -> nn.Module:
    last = None
    for mod in model.backbone.modules():
        if isinstance(mod, nn.Conv2d):
            last = mod
    if last is None:
        raise RuntimeError("No Conv2d found in backbone")
    return last


class GradCAM:
    """
    Grad-CAM on MobileViT last Conv2d (final_conv-scale map).

    Score for backprop (documented):
      CE head:    predicted-class logit  logits[pred]
      CORAL head: sum of the K-1=4 ordinal logits  (no class index exists)
    """

    def __init__(self, model: nn.Module, head: str) -> None:
        self.model = model
        self.head = head
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        self.layer = pick_target_layer(model)
        self._fh = self.layer.register_forward_hook(self._on_fwd)
        self._bh = self.layer.register_full_backward_hook(self._on_bwd)

    def _on_fwd(self, module, inp, out) -> None:
        self.acts = out

    def _on_bwd(self, module, gin, gout) -> None:
        self.grads = gout[0]

    def close(self) -> None:
        self._fh.remove()
        self._bh.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, float]:
        self.model.zero_grad(set_to_none=True)
        x = x.detach().requires_grad_(True)
        logits = self.model(x)
        pred = int(decode_predictions(logits, head=self.head)[0].item())
        if self.head == "coral":
            # CORAL: no 5-way class logit — use sum of threshold logits
            score = logits[0].sum()
            score_kind = "sum_coral_logits"
        else:
            target = pred if class_idx is None else int(class_idx)
            if target < 0 or target >= logits.shape[1]:
                raise RuntimeError(f"CE class_idx {target} out of range for logits {tuple(logits.shape)}")
            score = logits[0, target]
            score_kind = f"ce_logit_{target}"
        score.backward()
        assert self.acts is not None and self.grads is not None
        acts = self.acts.detach()
        grads = self.grads.detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        # attach kind via score float only; caller logs kind separately if needed
        return cam, pred, float(score.detach().cpu())

    def cam_nonzero(self, x: torch.Tensor) -> bool:
        cam, _, _ = self(x)
        return bool(np.nanmax(cam) > 0)


def denorm(chw: torch.Tensor) -> np.ndarray:
    x = chw.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x, 0, 1)


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = plt.cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)


def select_ids_by_grade(pred_path: Path, n_per_grade: int = 1) -> dict[int, list[str]]:
    """Prefer correctly classified samples; fall back to any with that true label."""
    pred = json.loads(pred_path.read_text())
    ids = pred["ids"]
    yt = pred["y_true"]
    yp = pred["y_pred"]
    chosen: dict[int, list[str]] = {g: [] for g in range(5)}
    # pass 1: correct
    for i, (t, p, iid) in enumerate(zip(yt, yp, ids)):
        g = int(t)
        if p == t and len(chosen[g]) < n_per_grade:
            chosen[g].append(str(iid))
    # pass 2: fill
    for i, (t, iid) in enumerate(zip(yt, ids)):
        g = int(t)
        if len(chosen[g]) < n_per_grade and str(iid) not in chosen[g]:
            chosen[g].append(str(iid))
    return chosen


def index_dataset_by_id(ds) -> dict[str, int]:
    out = {}
    for i in range(len(ds)):
        # avoid loading images: read metadata if available
        if hasattr(ds, "samples"):
            sid = str(ds.samples[i][0] if isinstance(ds.samples[i], (list, tuple)) else ds.samples[i].get("id", i))
        else:
            sid = str(ds[i]["id"])
        out[sid] = i
    return out


def build_eval_dataset(cfg: dict, dataset_name: str, split: str):
    roots = resolve_data_roots(cfg, ROOT)
    prep = cfg["preprocessing"]
    prep_name = prep["name"]
    cache_root = resolve_cache_root(cfg, ROOT)
    use_cache = cache_root is not None
    tf = build_preprocess(
        prep_name,
        image_size=prep["image_size"],
        train=False,
        cached=use_cache,
        clahe_clip_limit=prep.get("clahe_clip_limit", 2.0),
        clahe_tile_grid_size=tuple(prep.get("clahe_tile_grid_size", [8, 8])),
    )
    if dataset_name == "aptos":
        aptos_cfg = cfg["data"]["aptos"]
        kw = {k: v for k, v in aptos_cfg.items() if k != "root"}
        cache = aptos_cache_dir(cache_root, prep_name, split) if use_cache else None
        ds = build_dataset(
            "aptos",
            root=roots["aptos"],
            split=split,
            transform=tf,
            cache_dir=cache,
            **kw,
        )
    elif dataset_name == "idrid":
        idrid_cfg = cfg["data"]["idrid"]
        kw = {k: v for k, v in idrid_cfg.items() if k != "root"}
        cache = idrid_cache_dir(cache_root, prep_name) if use_cache else None
        ds = build_dataset(
            "idrid",
            root=roots["idrid"],
            split=split,
            transform=tf,
            cache_dir=cache,
            **kw,
        )
    else:
        raise ValueError(dataset_name)
    return ds


def load_model(rid: str, device: torch.device):
    cfg = load_config(ROOT / "experiments" / rid / "config.yaml")
    head = resolve_head(cfg)
    ckpt = torch.load(
        ROOT / "checkpoints" / rid / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"].get("num_classes", 5),
        pretrained=False,
        head=head,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, head


def save_panel(rgb, cam, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ov = overlay(rgb, cam)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(rgb)
    axes[0].set_title("input")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(ov)
    axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    if not require_or_skip("Grad-CAM overlays (paper/figures/gradcam/)", "APTOS", "IDRiD", extra=(CHECKPOINTS,)):
        return
    device = torch.device("cpu")  # RAM-safe; Grad-CAM needs activations
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grids").mkdir(exist_ok=True)
    (OUT / "compare").mkdir(exist_ok=True)

    # Fixed sample IDs from H2 s42 predictions (same images across all runs)
    ref = ROOT / "results" / "hybrid_h2_s42_v1"
    sample_plan: dict[str, dict[int, list[str]]] = {}
    for split_name, _, _ in SPLITS:
        p = ref / f"{split_name}_predictions.json"
        if not p.is_file():
            raise FileNotFoundError(p)
        sample_plan[split_name] = select_ids_by_grade(p, n_per_grade=1)

    index_lines = [
        "# Grad-CAM index (Phase N)",
        "",
        "Target layer: MobileViT `final_conv` (7×7).",
        "Score: predicted-class probability (CE or CORAL).",
        "Sample IDs fixed from `hybrid_h2_s42_v1` predictions (fair cross-run compare).",
        "",
        "## Selected IDs",
        "",
    ]
    for sn, grades in sample_plan.items():
        index_lines.append(f"### {sn}")
        for g, ids in grades.items():
            index_lines.append(f"- grade {g}: `{', '.join(ids) or 'NONE'}`")
        index_lines.append("")

    # Store cams for cross-run compare: (split, grade) -> list of (run_id, rgb, cam, pred, conf)
    compare_store: dict[tuple[str, int], list] = {}

    runs = [run_id(c, s) for s in SEEDS for c in CELLS]
    manifest = []

    for ri, rid in enumerate(runs, 1):
        print(f"[{ri}/{len(runs)}] {rid}", flush=True)
        model, cfg, head = load_model(rid, device)
        cam_engine = GradCAM(model, head)
        run_dir = OUT / rid
        run_dir.mkdir(exist_ok=True)

        for split_name, ds_name, split in SPLITS:
            ds = build_eval_dataset(cfg, ds_name, split)
            # map id -> index (may load once; ok for APTOS/IDRiD sizes)
            id_to_idx = {}
            for i in range(len(ds)):
                item = ds[i]
                id_to_idx[str(item["id"])] = i

            strip_imgs = []
            strip_titles = []
            for g in range(5):
                ids = sample_plan[split_name].get(g, [])
                if not ids:
                    continue
                iid = ids[0]
                if iid not in id_to_idx:
                    print(f"  WARN missing id {iid} in {rid}/{split_name}", flush=True)
                    continue
                batch = ds[id_to_idx[iid]]
                x = batch["image"].unsqueeze(0).to(device)
                y = int(batch["label"])
                cam, pred, conf = cam_engine(x)
                rgb = denorm(batch["image"])
                title = f"{rid} | {split_name} | id={iid} true={y} pred={pred} p={conf:.2f}"
                out_png = run_dir / f"{split_name}_g{g}_{iid}.png"
                save_panel(rgb, cam, out_png, title)
                strip_imgs.append(overlay(rgb, cam))
                strip_titles.append(f"g{g} t{y}/p{pred}")
                compare_store.setdefault((split_name, g), []).append(
                    (rid, rgb, cam, pred, conf, iid, y)
                )
                manifest.append(
                    {
                        "run_id": rid,
                        "split": split_name,
                        "grade_true": y,
                        "id": iid,
                        "pred": pred,
                        "conf": conf,
                        "path": str(out_png.relative_to(ROOT)),
                    }
                )

            if strip_imgs:
                fig, axes = plt.subplots(1, len(strip_imgs), figsize=(2.4 * len(strip_imgs), 2.6))
                if len(strip_imgs) == 1:
                    axes = [axes]
                for ax, im, t in zip(axes, strip_imgs, strip_titles):
                    ax.imshow(im)
                    ax.set_title(t, fontsize=8)
                    ax.axis("off")
                fig.suptitle(f"{rid} · {split_name}", fontsize=10)
                fig.tight_layout()
                fig.savefig(OUT / "grids" / f"{rid}_{split_name}_by_grade.png", dpi=140)
                plt.close(fig)

        cam_engine.close()
        del model, cam_engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Comparison grids: same image, all runs
    for (split_name, g), entries in sorted(compare_store.items()):
        if not entries:
            continue
        n = len(entries)
        cols = 4
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.7))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")
        for ax, (rid, rgb, cam, pred, conf, iid, y) in zip(axes, entries):
            ax.imshow(overlay(rgb, cam))
            short = rid.replace("hybrid_", "").replace("_v1", "")
            ax.set_title(f"{short}\np{pred} ({conf:.2f})", fontsize=7)
            ax.axis("off")
        fig.suptitle(f"{split_name} · true grade {g} · id={entries[0][5]}", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "compare" / f"{split_name}_grade{g}_all_runs.png", dpi=140)
        plt.close(fig)

    (OUT / "manifest.json").write_text(json.dumps({"n": len(manifest), "items": manifest}, indent=2))
    index_lines += [
        "## Outputs",
        "",
        f"- Per-run panels: `paper/figures/gradcam/hybrid_h*_s*_v1/`",
        f"- Grade strips: `paper/figures/gradcam/grids/`",
        f"- Cross-run compare: `paper/figures/gradcam/compare/`",
        f"- Manifest: `paper/figures/gradcam/manifest.json`",
        "",
        f"Generated **{len(manifest)}** overlays across **{len(runs)}** runs.",
        "",
        "GRADCAM_ALL_OK",
        "",
    ]
    (OUT / "INDEX.md").write_text("\n".join(index_lines))
    print("GRADCAM_ALL_OK", flush=True)
    print("Open", OUT / "INDEX.md", flush=True)


if __name__ == "__main__":
    main()
