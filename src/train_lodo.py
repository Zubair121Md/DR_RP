#!/usr/bin/env python3
"""
Phase P — Leave-One-Dataset-Out (LODO) training.

Explicitly trains on multi-source data (APTOS train + non-held-out externals).
This overrides the main paper rule “externals never for training” — LODO only.

Val selection remains APTOS val QWK.
Held-out fold is evaluated after training via src.evaluate-style splits.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset

from src.dataset import build_dataloader, build_dataset
from src.losses import build_loss
from src.models import build_model
from src.preprocessing import build_preprocess
from src.train import (
    build_optimizer,
    evaluate_loader,
    train_one_epoch,
)
from src.utils import (
    aptos_cache_dir,
    deepdrid_cache_dir,
    ensure_run_dirs,
    get_device,
    idrid_cache_dir,
    load_config,
    messidor2_cache_dir,
    resolve_cache_root,
    resolve_data_roots,
    save_json,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent
HOLDOUTS = ("idrid", "messidor2", "deepdrid")


def _cache_ready(cache_dir: Path | None, min_files: int = 10) -> bool:
    if cache_dir is None or not cache_dir.is_dir():
        return False
    n = sum(1 for _ in cache_dir.rglob("*.png"))
    return n >= min_files


def _external_ds(
    name: str,
    roots: dict,
    cfg: dict,
    prep_cfg: dict,
    train: bool,
    cache_root: Path | None,
    prep_name: str,
):
    """Build external dataset; fall back to online preprocess if cache incomplete."""
    image_size = int(prep_cfg["image_size"])
    clip = float(prep_cfg.get("clahe_clip_limit", 2.0))
    tiles = tuple(prep_cfg.get("clahe_tile_grid_size", [8, 8]))

    def _tf(use_cached: bool):
        return build_preprocess(
            prep_name,
            image_size=image_size,
            train=train,
            cached=use_cached,
            clahe_clip_limit=clip,
            clahe_tile_grid_size=tiles,
        )

    if name == "idrid":
        cdir = idrid_cache_dir(cache_root, prep_name)
        use = _cache_ready(cdir, 100)
        if not use:
            print(f"WARN {name}: cache incomplete → online preprocess", flush=True)
        kw = {k: v for k, v in cfg["data"]["idrid"].items() if k != "root"}
        return build_dataset(
            "idrid",
            root=roots["idrid"],
            split="all",
            transform=_tf(use),
            cache_dir=cdir if use else None,
            **kw,
        )
    if name == "messidor2":
        cdir = messidor2_cache_dir(cache_root, prep_name)
        use = _cache_ready(cdir, 100)
        if not use:
            print(f"WARN {name}: cache incomplete → online preprocess", flush=True)
        kw = {k: v for k, v in (cfg["data"].get("messidor2") or {}).items() if k != "root"}
        return build_dataset(
            "messidor2",
            root=roots["messidor2"],
            split="all",
            transform=_tf(use),
            cache_dir=cdir if use else None,
            **kw,
        )
    if name == "deepdrid":
        cdir = deepdrid_cache_dir(cache_root, prep_name)
        use = _cache_ready(cdir, 100)
        if not use:
            print(f"WARN {name}: cache incomplete → online preprocess", flush=True)
        kw = {k: v for k, v in (cfg["data"].get("deepdrid") or {}).items() if k != "root"}
        return build_dataset(
            "deepdrid",
            root=roots["deepdrid"],
            split="all",
            transform=_tf(use),
            cache_dir=cdir if use else None,
            **kw,
        )
    raise ValueError(name)


def train_lodo(
    config: dict[str, Any],
    run_id: str,
    holdout: str,
    seed_override: int | None = None,
) -> dict[str, Any]:
    assert holdout in HOLDOUTS
    project_root = ROOT
    seed = int(seed_override if seed_override is not None else config["experiment"]["seed"])
    config = copy.deepcopy(config)
    config.setdefault("experiment", {})["seed"] = seed
    set_seed(seed)
    train_cfg = config["train"]
    device = get_device(train_cfg.get("device"))
    if device.type == "cpu":
        torch.set_num_threads(1)

    dirs = ensure_run_dirs(project_root, run_id)
    save_json(dirs["experiments"] / "config.json", config)
    save_json(dirs["results"] / "config.json", config)
    with (dirs["experiments"] / "config.yaml").open("w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(config, f, sort_keys=False)
    (dirs["experiments"] / "holdout.txt").write_text(holdout)

    roots = resolve_data_roots(config, project_root)
    cache_root = resolve_cache_root(config, project_root)
    prep_cfg = config["preprocessing"]
    prep_name = prep_cfg["name"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    num_classes = int(config["data"]["num_classes"])
    use_cache = cache_root is not None

    train_tf = build_preprocess(
        prep_name, image_size=prep_cfg["image_size"], train=True, cached=use_cache,
        clahe_clip_limit=prep_cfg.get("clahe_clip_limit", 2.0),
        clahe_tile_grid_size=tuple(prep_cfg.get("clahe_tile_grid_size", [8, 8])),
    )
    eval_tf = build_preprocess(
        prep_name, image_size=prep_cfg["image_size"], train=False, cached=use_cache,
        clahe_clip_limit=prep_cfg.get("clahe_clip_limit", 2.0),
        clahe_tile_grid_size=tuple(prep_cfg.get("clahe_tile_grid_size", [8, 8])),
    )

    aptos_kw = {k: v for k, v in config["data"]["aptos"].items() if k != "root"}
    aptos_train = build_dataset(
        "aptos", root=roots["aptos"], split="train", transform=train_tf,
        cache_dir=aptos_cache_dir(cache_root, prep_name, "train") if use_cache else None,
        **aptos_kw,
    )
    aptos_val = build_dataset(
        "aptos", root=roots["aptos"], split="val", transform=eval_tf,
        cache_dir=aptos_cache_dir(cache_root, prep_name, "val") if use_cache else None,
        **aptos_kw,
    )

    train_parts = [aptos_train]
    for name in HOLDOUTS:
        if name == holdout:
            continue
        train_parts.append(
            _external_ds(
                name,
                roots,
                config,
                prep_cfg,
                train=True,
                cache_root=cache_root,
                prep_name=prep_name,
            )
        )
    train_ds = ConcatDataset(train_parts)
    val_ds = aptos_val

    head = str(model_cfg.get("head", "ce")).lower()
    if str(loss_cfg.get("name", "")).lower() in {"coral", "mild_aware_coral", "mild-aware-coral"}:
        head = "coral"

    print(f"LODO run   : {run_id}")
    print(f"Holdout    : {holdout}")
    print(f"Train size : {len(train_ds)} (APTOS train + non-held-out externals)")
    print(f"Val size   : {len(val_ds)} (APTOS val)")
    print(f"Device     : {device}")

    batch_size = int(train_cfg["batch_size"])
    freeze_bn = bool(train_cfg.get("freeze_bn", False))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(
        model_cfg["name"],
        num_classes=model_cfg.get("num_classes", num_classes),
        pretrained=model_cfg.get("pretrained", True),
        head=head,
    ).to(device)
    loss_kwargs = {k: v for k, v in loss_cfg.items() if k != "name"}
    loss_kwargs.setdefault("num_classes", num_classes)
    criterion = build_loss(loss_cfg["name"], **loss_kwargs).to(device)
    optimizer = build_optimizer(
        train_cfg.get("optimizer", "adamw"),
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(train_cfg["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) if str(train_cfg.get("scheduler", "")).lower() == "cosine" else None

    history: list[dict] = []
    best_score = float("-inf")
    best_epoch = -1
    best_state = None
    best_val_metrics: dict = {}
    selection_metric = train_cfg.get("selection_metric", "quadratic_weighted_kappa")
    summary_path = dirs["results"] / "train_summary.json"
    last_path = dirs["checkpoints"] / "last.pt"
    if summary_path.is_file():
        print("Already complete", summary_path)
        return {}

    start_epoch, start_step = 1, 0
    interrupt_every = int(train_cfg.get("interrupt_every", 40))
    max_steps = int(train_cfg.get("max_steps_per_launch", 0))
    if last_path.is_file():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history = list(ckpt.get("history") or [])
        best_score = float(ckpt.get("best_score", float("-inf")))
        best_epoch = int(ckpt.get("best_epoch", -1))
        best_val_metrics = dict(ckpt.get("best_val_metrics") or {})
        if ckpt.get("best_state_dict") is not None:
            best_state = ckpt["best_state_dict"]
        start_epoch = int(ckpt.get("epoch", 1))
        start_step = int(ckpt.get("step", 0))

    def save_last(epoch: int, step: int) -> None:
        torch.save(
            {
                "run_id": run_id,
                "holdout": holdout,
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "history": history,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
                "best_state_dict": best_state,
                "config": config,
            },
            last_path,
        )

    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = start_step if epoch == start_epoch else 0
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        train_loader = build_dataloader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g
        )

        def _cb(step: int, _e: int = epoch) -> None:
            save_last(_e, step)
            print(f"\n[lodo-interrupt] epoch={_e} step={step}", flush=True)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, None, False,
            start_step=epoch_start, interrupt_every=interrupt_every, interrupt_cb=_cb,
            freeze_bn=freeze_bn, max_steps=max_steps,
        )
        save_last(epoch + 1, 0)
        val_metrics = evaluate_loader(model, val_loader, criterion, device, num_classes, head=head)
        if scheduler is not None:
            scheduler.step()
        score = float(val_metrics.get(selection_metric, val_metrics.get("quadratic_weighted_kappa", 0.0)))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_qwk": val_metrics["quadratic_weighted_kappa"],
            "val_mae": val_metrics["mae"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_qwk={val_metrics['quadratic_weighted_kappa']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"run_id": run_id, "holdout": holdout, "model_state_dict": best_state,
                        "epoch": epoch, "best_score": best_score, "config": config},
                       dirs["checkpoints"] / "best.pt")
        save_json(dirs["results"] / "history.json", history)
        start_step = 0

    if best_state is not None:
        model.load_state_dict(best_state)
    summary = {
        "run_id": run_id,
        "holdout": holdout,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_val_metrics": best_val_metrics,
        "elapsed_sec": time.time() - t0,
        "note": "Phase P LODO — externals used in training except held-out fold",
    }
    save_json(summary_path, summary)
    save_json(dirs["experiments"] / "train_summary.json", summary)
    print("LODO_TRAIN_DONE", run_id, flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--holdout", required=True, choices=list(HOLDOUTS))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, default=None, help="Override config experiment.seed")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)
    train_lodo(cfg, args.run_id, args.holdout, seed_override=args.seed)


if __name__ == "__main__":
    main()
