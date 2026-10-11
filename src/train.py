"""
Training entrypoint for Experiment 1 — EfficientNet-B0 RGB baseline on APTOS.

Train: APTOS train only
Validate: APTOS validation only
Never touches IDRiD during training / model selection.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.dataset import build_dataloader, build_dataset
from src.losses import build_loss
from src.metrics import compute_metrics
from src.models import build_model, decode_predictions
from src.preprocessing import build_preprocess
from src.utils import (
    aptos_cache_dir,
    ensure_run_dirs,
    get_device,
    load_config,
    make_run_id,
    resolve_cache_root,
    resolve_data_roots,
    save_json,
    set_seed,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int = 5,
    head: str = "ce",
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        n_batches += 1
        preds = decode_predictions(logits, head=head)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    start_step: int = 0,
    interrupt_every: int = 0,
    interrupt_cb: Any | None = None,
    freeze_bn: bool = False,
    max_steps: int = 0,
) -> float:
    model.train()
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.eval()
    total_loss = 0.0
    n_batches = 0
    steps_run = 0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if step < start_step:
            continue
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        steps_run += 1
        del logits, loss, images, labels

        if interrupt_every > 0 and interrupt_cb is not None and (step + 1) % interrupt_every == 0:
            interrupt_cb(step + 1)

        if max_steps > 0 and steps_run >= max_steps:
            if interrupt_cb is not None:
                interrupt_cb(step + 1)
            print(f"\n[chunk-exit] epoch-step={step + 1} ran={steps_run}", flush=True)
            raise SystemExit(75)

    return total_loss / max(n_batches, 1)


def build_optimizer(name: str, params: Any, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    key = name.lower().strip()
    if key == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def train(
    config: dict[str, Any],
    project_root: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    project_root = project_root or _project_root()
    seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    train_cfg = config["train"]
    device = get_device(train_cfg.get("device"))
    if device.type == "cpu":
        n_threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
        torch.set_num_threads(max(1, n_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    exp_name = config["experiment"]["name"]
    run_id = run_id or make_run_id(exp_name)
    dirs = ensure_run_dirs(project_root, run_id)

    # Persist exact config for this run (never overwrite prior runs)
    save_json(dirs["experiments"] / "config.json", config)
    save_json(dirs["results"] / "config.json", config)
    with (dirs["experiments"] / "config.yaml").open("w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(config, f, sort_keys=False)

    roots = resolve_data_roots(config, project_root)
    aptos_cfg = config["data"]["aptos"]
    prep_cfg = config["preprocessing"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    num_classes = int(config["data"]["num_classes"])

    if train_cfg.get("train_dataset", "aptos") != "aptos":
        raise RuntimeError("Experiment 1 must train on APTOS only.")

    cache_root = resolve_cache_root(config, project_root)
    prep_name = prep_cfg["name"]
    use_cache = cache_root is not None
    if use_cache:
        train_cache = aptos_cache_dir(cache_root, prep_name, "train")
        val_cache = aptos_cache_dir(cache_root, prep_name, "val")
        assert train_cache is not None and val_cache is not None
        if not train_cache.is_dir() or not val_cache.is_dir():
            raise FileNotFoundError(
                f"Preprocess cache missing under {cache_root / prep_name}. "
                "Run: python scripts/build_preprocess_cache.py --verify"
            )
        print(f"Cache     : {cache_root / prep_name} (deterministic frontend+224)")

    train_tf = build_preprocess(
        prep_name,
        image_size=prep_cfg["image_size"],
        train=True,
        cached=use_cache,
        clahe_clip_limit=prep_cfg.get("clahe_clip_limit", 2.0),
        clahe_tile_grid_size=tuple(prep_cfg.get("clahe_tile_grid_size", [8, 8])),
    )
    eval_tf = build_preprocess(
        prep_name,
        image_size=prep_cfg["image_size"],
        train=False,
        cached=use_cache,
        clahe_clip_limit=prep_cfg.get("clahe_clip_limit", 2.0),
        clahe_tile_grid_size=tuple(prep_cfg.get("clahe_tile_grid_size", [8, 8])),
    )

    aptos_kw = {k: v for k, v in aptos_cfg.items() if k != "root"}
    train_ds = build_dataset(
        "aptos",
        root=roots["aptos"],
        split="train",
        transform=train_tf,
        cache_dir=aptos_cache_dir(cache_root, prep_name, "train") if use_cache else None,
        **aptos_kw,
    )
    val_ds = build_dataset(
        "aptos",
        root=roots["aptos"],
        split="val",
        transform=eval_tf,
        cache_dir=aptos_cache_dir(cache_root, prep_name, "val") if use_cache else None,
        **aptos_kw,
    )

    # Safety: never construct IDRiD / Messidor / DeepDRiD during training
    head = str(model_cfg.get("head", "ce")).lower().strip()
    loss_name = str(loss_cfg.get("name", "")).lower().strip()
    if loss_name in {"coral", "mild_aware_coral", "mild-aware-coral", "mild_coral"}:
        head = "coral"

    print(f"Run ID     : {run_id}")
    print(f"Device     : {device}")
    print(f"Head/Loss  : {head} / {loss_cfg.get('name')}")
    print(f"APTOS train: {len(train_ds)} images from {roots['aptos']}")
    print(f"APTOS val  : {len(val_ds)} images")
    print("Externals  : NOT used during training/validation")

    batch_size = int(train_cfg["batch_size"])
    freeze_bn = bool(train_cfg.get("freeze_bn", False))
    if freeze_bn:
        print("Freeze BN  : True (ImageNet running stats; BN-safe for small batches)")
    num_workers = int(train_cfg.get("num_workers", 2))
    train_loader = build_dataloader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = build_dataloader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

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
    scheduler = None
    if str(train_cfg.get("scheduler", "")).lower() == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = -1
    best_state = None
    best_val_metrics: dict[str, Any] = {}
    selection_metric = train_cfg.get("selection_metric", "quadratic_weighted_kappa")
    start_epoch = 1
    start_step = 0
    interrupt_every = int(train_cfg.get("interrupt_every", 80))
    max_steps = int(train_cfg.get("max_steps_per_launch", 0))
    last_path = dirs["checkpoints"] / "last.pt"
    summary_path = dirs["results"] / "train_summary.json"

    if summary_path.is_file():
        print(f"Already complete: {summary_path}")
        import json

        return json.loads(summary_path.read_text())

    if last_path.is_file():
        print(f"Resuming from {last_path}")
        try:
            ckpt = torch.load(last_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(last_path, map_location=device)
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
        # last.pt stores the next step to run within epoch (0 = start of epoch)
        start_epoch = int(ckpt.get("epoch", 1))
        start_step = int(ckpt.get("step", 0))
        print(f"Resume at epoch={start_epoch} step={start_step} history={len(history)}")

    def _atomic_save_last(epoch: int, step: int, *, light: bool = False) -> None:
        # light=True for mid-epoch interrupt: model + step only (cuts RAM/disk spikes)
        if light:
            payload = {
                "run_id": run_id,
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None,
                "history": history,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
                "best_state_dict": None,
                "config": config,
            }
        else:
            payload = {
                "run_id": run_id,
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "history": history,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
                "best_state_dict": best_state,
                "config": config,
            }
        tmp = last_path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(last_path)

    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_step = start_step if epoch == start_epoch else 0
        # Deterministic shuffle per epoch so mid-epoch resume skips the same batches
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        train_loader = build_dataloader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=g,
        )

        def _interrupt_cb(step: int, _epoch: int = epoch) -> None:
            _atomic_save_last(_epoch, step, light=True)
            if device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            print(f"\n[interrupt-ckpt] epoch={_epoch} step={step}", flush=True)

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            start_step=epoch_start_step,
            interrupt_every=interrupt_every,
            interrupt_cb=_interrupt_cb,
            freeze_bn=freeze_bn,
            max_steps=max_steps,
        )
        # Finished this epoch's training batches — mark step 0 of next epoch before val
        _atomic_save_last(epoch + 1, 0)

        val_metrics = evaluate_loader(
            model, val_loader, criterion, device, num_classes, head=head
        )
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_qwk": val_metrics["quadratic_weighted_kappa"],
            "val_mae": val_metrics["mae"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_qwk={val_metrics['quadratic_weighted_kappa']:.4f} | "
            f"val_mild_f1={val_metrics.get('mild_dr_f1', float('nan')):.4f}"
        )

        score = float(val_metrics.get(selection_metric, val_metrics["quadratic_weighted_kappa"]))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            ckpt_path = dirs["checkpoints"] / "best.pt"
            torch.save(
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                    "selection_metric": selection_metric,
                    "selection_score": best_score,
                },
                ckpt_path,
            )

        save_json(dirs["results"] / "history.json", history)
        _atomic_save_last(epoch + 1, 0)

        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    elapsed = time.time() - t0
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epochs
        best_val_metrics = history[-1] if history else {}
    summary = {
        "run_id": run_id,
        "experiment_name": exp_name,
        "device": str(device),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_selection_metric": selection_metric,
        "best_selection_score": best_score,
        "best_val_metrics": best_val_metrics,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "elapsed_sec": elapsed,
        "checkpoint": str(dirs["checkpoints"] / "best.pt"),
        "results_dir": str(dirs["results"]),
        "experiments_dir": str(dirs["experiments"]),
        "aptos_train_size": len(train_ds),
        "aptos_val_size": len(val_ds),
        "note": "IDRiD / Messidor-2 / DeepDRiD were not used for training or validation.",
    }
    save_json(dirs["results"] / "train_summary.json", summary)
    save_json(dirs["experiments"] / "train_summary.json", summary)
    save_json(dirs["results"] / "history.json", history)
    # Clear interrupt marker so resume won't re-enter a finished run
    if last_path.is_file():
        last_path.unlink(missing_ok=True)
    print(f"\nBest epoch: {best_epoch} ({selection_metric}={best_score:.4f})")
    print(f"Checkpoint: {dirs['checkpoints'] / 'best.pt'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Experiment 1 DR baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_efficientnet_b0_rgb.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root (defaults to repository root).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Fixed run id (enables interrupt resume into the same directories).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root) if args.project_root else _project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(config_path)
    train(config, project_root=project_root, run_id=args.run_id)


if __name__ == "__main__":
    main()
