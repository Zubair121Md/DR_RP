#!/usr/bin/env python3
"""
Config-driven Grad-CAM mild panel (H1–H4).

Selection rule from configs/paper/gradcam_mild_panel.yaml:
  APTOS test, true grade 1; 3 H2-correct + 3 H2-miss; RNG seed 42.
CAM score: CE = predicted-class logit; CORAL = sum of 4 logits.
Writes PDF+PNG under paper/figures/main/. Does not overwrite archived gradcam runs
except via new fig10_* outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(ROOT))
from src.restricted import CHECKPOINTS, require_or_skip  # noqa: E402
sys.path.insert(0, str(ROOT))

from scripts.run_gradcam_all import GradCAM, denorm, resolve_head, run_id  # noqa: E402
from src.dataset import build_dataset  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocessing import build_preprocess  # noqa: E402
from src.utils import aptos_cache_dir, load_config, resolve_cache_root, resolve_data_roots  # noqa: E402

CFG_PATH = ROOT / "configs/paper/gradcam_mild_panel.yaml"


def select_ids(cfg: dict) -> tuple[list[str], list[str]]:
    ref_cell = cfg["selection"]["reference_cell"]
    ref_seed = int(cfg["selection"]["reference_seed"])
    split = cfg["selection"]["split"]
    grade = int(cfg["selection"]["true_grade"])
    pred_path = ROOT / "results" / run_id(ref_cell, ref_seed) / f"{split}_predictions.json"
    if not pred_path.is_file():
        raise FileNotFoundError(f"MISSING {pred_path}")
    print("READ", pred_path.relative_to(ROOT))
    pred = json.loads(pred_path.read_text())
    yt = pred["y_true"]
    yp = pred["y_pred"]
    ids = pred["ids"]
    correct = [ids[i] for i in range(len(ids)) if yt[i] == grade and yp[i] == grade]
    incorrect = [ids[i] for i in range(len(ids)) if yt[i] == grade and yp[i] != grade]
    rng = np.random.default_rng(int(cfg["selection"]["rule_seed"]))
    n_c = int(cfg["selection"]["n_correct"])
    n_i = int(cfg["selection"]["n_incorrect"])
    if len(correct) < n_c:
        raise RuntimeError(f"need {n_c} correct mild IDs, found {len(correct)}")
    if len(incorrect) < n_i:
        raise RuntimeError(f"need {n_i} incorrect mild IDs, found {len(incorrect)}")
    # deterministic sample without replacement
    c_idx = rng.choice(len(correct), size=n_c, replace=False)
    i_idx = rng.choice(len(incorrect), size=n_i, replace=False)
    sel_c = [correct[int(j)] for j in sorted(c_idx.tolist())]
    sel_i = [incorrect[int(j)] for j in sorted(i_idx.tolist())]
    return sel_c, sel_i


def load_model(cell: str, seed: int, device: torch.device):
    rid = run_id(cell, seed)
    cfg = load_config(ROOT / "experiments" / rid / "config.json")
    head = resolve_head(cfg)
    model = build_model(
        cfg["model"]["name"],
        num_classes=int(cfg["data"]["num_classes"]),
        pretrained=False,
        head=head,
    )
    ckpt = torch.load(ROOT / "checkpoints" / rid / "best.pt", map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("model")
    if state is None:
        raise KeyError(f"No model weights in checkpoint for {rid}; keys={list(ckpt.keys())}")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    if bool(cfg.get("train", {}).get("freeze_bn", False)):
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
    return model, cfg, head


def tensor_for_id(cfg: dict, image_id: str, device: torch.device) -> tuple[torch.Tensor, int]:
    roots = resolve_data_roots(cfg, ROOT)
    prep = cfg["preprocessing"]
    prep_name = prep["name"]
    cache_root = resolve_cache_root(cfg, ROOT)
    use_cache = cache_root is not None
    tf = build_preprocess(
        prep_name,
        image_size=int(prep["image_size"]),
        train=False,
        cached=use_cache,
        clahe_clip_limit=prep.get("clahe_clip_limit", 2.0),
        clahe_tile_grid_size=tuple(prep.get("clahe_tile_grid_size", [8, 8])),
    )
    aptos_cfg = cfg["data"]["aptos"]
    kw = {k: v for k, v in aptos_cfg.items() if k != "root"}
    cache = aptos_cache_dir(cache_root, prep_name, "test") if use_cache else None
    ds = build_dataset(
        "aptos",
        root=roots["aptos"],
        split="test",
        transform=tf,
        cache_dir=cache,
        **kw,
    )
    for i in range(len(ds)):
        item = ds[i]
        if item["id"] == image_id:
            return item["image"].unsqueeze(0).to(device), int(item["label"])
    raise KeyError(f"id {image_id} not in APTOS test")


def sanity_random_weights(model, head, x, cam_engine, cfg) -> None:
    cam_real, _, _ = cam_engine(x)
    # Fresh model with random weights (avoid deepcopy of hooked module)
    rand_model = build_model(
        cfg["model"]["name"],
        num_classes=int(cfg["data"]["num_classes"]),
        pretrained=False,
        head=head,
    )
    rand_model.eval()
    rand_cam = GradCAM(rand_model, head)
    cam_r, _, _ = rand_cam(x)
    rand_cam.close()
    if np.allclose(cam_real, cam_r, atol=1e-5):
        raise RuntimeError("SANITY FAIL: CAM identical under randomised weights")
    print(
        "SANITY_OK randomised CAM differs; real_max",
        float(cam_real.max()),
        "rand_max",
        float(cam_r.max()),
    )


def main() -> None:
    if not require_or_skip("Grad-CAM mild panel (fig10_gradcam_mild_panel, Figure S19)", "APTOS", extra=(CHECKPOINTS,)):
        return
    print("READ", CFG_PATH.relative_to(ROOT))
    cfg_sel = yaml.safe_load(CFG_PATH.read_text())
    sel_c, sel_i = select_ids(cfg_sel)
    image_ids = sel_c + sel_i
    tags = ["H2 correct"] * len(sel_c) + ["H2 missed"] * len(sel_i)
    print("SELECTED correct", sel_c)
    print("SELECTED miss", sel_i)

    device = torch.device("cpu")
    cells = list(cfg_sel["selection"]["cells_in_figure"])
    seed = int(cfg_sel["selection"]["reference_seed"])

    # load one image tensor per id using H2 preprocess for display consistency of columns?
    # Each cell has its own preprocess (RGB vs CLAHE) — use that cell's preprocess for its column.
    n_rows = len(image_ids)
    n_cols = 1 + len(cells)  # input from H2 green or rgb? show H2 input as reference column 0
    # Column 0 = H2 input image; then CAM overlays per cell
    fig_w = 6.9
    fig_h = max(3.0, 1.15 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = np.array([axes])

    models = {}
    cams = {}
    cfgs = {}
    for cell in cells:
        model, cfg, head = load_model(cell, seed, device)
        models[cell] = (model, head)
        cfgs[cell] = cfg
        cams[cell] = GradCAM(model, head)
        print("LOADED", run_id(cell, seed), "head", head)

    # sanity on first image with H4 if present else H2
    sanity_cell = "H4" if "H4" in cells else cells[0]
    x0, _ = tensor_for_id(cfgs[sanity_cell], image_ids[0], device)
    sanity_random_weights(
        models[sanity_cell][0],
        models[sanity_cell][1],
        x0,
        cams[sanity_cell],
        cfgs[sanity_cell],
    )
    # also require non-zero CAM for CORAL
    cam_s, pred_s, _ = cams[sanity_cell](x0)
    if float(np.nanmax(cam_s)) <= 0:
        raise RuntimeError(f"CAM still blank for {sanity_cell} after logit-sum fix")
    print("CAM_NONZERO", sanity_cell, "max", float(cam_s.max()), "pred", pred_s)

    ref_cfg = cfgs.get("H2", cfgs[cells[0]])
    for r, (iid, tag) in enumerate(zip(image_ids, tags)):
        x_ref, y_true = tensor_for_id(ref_cfg, iid, device)
        axes[r, 0].imshow(denorm(x_ref[0]))
        axes[r, 0].set_ylabel(f"Image {r + 1}\n{tag}\ntrue grade {y_true}", fontsize=8)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("(a) Shared CLAHE display", fontsize=8)

        for c, cell in enumerate(cells):
            x, y_true = tensor_for_id(cfgs[cell], iid, device)
            cam, pred, score = cams[cell](x)
            rgb = denorm(x[0])
            heat = plt.cm.viridis(cam)[..., :3]
            overlay = np.clip(0.55 * rgb + 0.45 * heat, 0, 1)
            ax = axes[r, c + 1]
            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"({'bcdefgh'[c]}) {cell}", fontsize=8)
            ax.text(
                0.02,
                0.98,
                f"pred {pred}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.45, pad=1, edgecolor="none"),
            )

    for eng in cams.values():
        eng.close()

    fig.tight_layout()
    out_dir = ROOT / cfg_sel["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg_sel["output"]["figure_stem"]
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # selection log
    log = {
        "config": str(CFG_PATH.relative_to(ROOT)),
        "correct_ids": sel_c,
        "miss_ids": sel_i,
        "cam_score": cfg_sel["selection"]["cam_score"],
        "outputs": [str(png.relative_to(ROOT)), str(pdf.relative_to(ROOT))],
    }
    (out_dir / f"{stem}_selection.json").write_text(json.dumps(log, indent=2))
    print("WROTE", png.relative_to(ROOT))
    print("WROTE", pdf.relative_to(ROOT))
    print("GRADCAM_MILD_PANEL_OK")


if __name__ == "__main__":
    main()
