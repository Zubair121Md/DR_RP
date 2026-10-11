"""
Evaluation entrypoint.

Evaluates a saved best checkpoint on:
  A. APTOS test (in-domain)
  B. IDRiD external
  C. Messidor-2 external (if present)
  D. DeepDRiD external (if present)

Saves predictions, logits, class probabilities, grade-1 F1, ECE,
temperature-scaled ECE (T fit on APTOS val), and bootstrap CIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.dataset import build_dataloader, build_dataset
from src.metrics import (
    bootstrap_metric_cis,
    compute_metrics,
    expected_calibration_error,
    fit_temperature,
    temperature_scale_probs,
)
from src.models import build_model, decode_predictions, logits_to_probs
from src.preprocessing import build_preprocess
from src.utils import (
    aptos_cache_dir,
    deepdrid_cache_dir,
    get_device,
    idrid_cache_dir,
    load_config,
    messidor2_cache_dir,
    resolve_cache_root,
    resolve_data_roots,
    save_json,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_head(config: dict[str, Any]) -> str:
    model_cfg = config.get("model", {})
    loss_name = str(config.get("loss", {}).get("name", "cross_entropy")).lower().strip()
    head = str(model_cfg.get("head", "ce")).lower().strip()
    if loss_name in {"coral", "mild_aware_coral", "mild-aware-coral", "mild_coral"}:
        return "coral"
    if head in {"coral", "ordinal", "mild_aware_coral"}:
        return "coral"
    return head if head else "ce"


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    head: str = "ce",
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    ids: list[str] = []
    splits: list[str] = []
    probs_list: list[list[float]] = []
    logits_list: list[list[float]] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        logits = model(images)
        preds = decode_predictions(logits, head=head)
        probs = logits_to_probs(logits, head=head)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        ids.extend(batch["id"])
        splits.extend(batch["split"])
        probs_list.extend(probs.cpu().tolist())
        logits_list.extend(logits.cpu().tolist())

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "ids": ids,
        "splits": splits,
        "probs": probs_list,
        "logits": logits_list,
    }


def evaluate(
    config: dict[str, Any],
    checkpoint: str | Path,
    project_root: Path | None = None,
    results_dir: str | Path | None = None,
    n_boot: int = 1000,
    skip_missing_external: bool = True,
) -> dict[str, Any]:
    project_root = project_root or _project_root()
    device = get_device()
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    run_id = ckpt.get("run_id", ckpt_path.parent.name)

    if results_dir is None:
        out_dir = project_root / "results" / run_id
    else:
        out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = resolve_data_roots(config, project_root)
    aptos_cfg = config["data"]["aptos"]
    idrid_cfg = config["data"].get("idrid", {})
    messidor_cfg = config["data"].get("messidor2") or config["data"].get("messidor") or {}
    deepdrid_cfg = config["data"].get("deepdrid") or {}
    prep_cfg = config["preprocessing"]
    model_cfg = config["model"]
    num_classes = int(config["data"]["num_classes"])
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"].get("num_workers", 2))
    head = _resolve_head(config)

    cache_root = resolve_cache_root(config, project_root)
    prep_name = prep_cfg["name"]
    use_cache = cache_root is not None
    if use_cache:
        print(f"Cache     : {cache_root / prep_name}")

    eval_tf = build_preprocess(
        prep_name,
        image_size=prep_cfg["image_size"],
        train=False,
        cached=use_cache,
        clahe_clip_limit=prep_cfg.get("clahe_clip_limit", 2.0),
        clahe_tile_grid_size=tuple(prep_cfg.get("clahe_tile_grid_size", [8, 8])),
    )
    model = build_model(
        model_cfg["name"],
        num_classes=model_cfg.get("num_classes", num_classes),
        pretrained=False,
        head=head,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    report: dict[str, Any] = {
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "head": head,
        "preprocess": prep_name,
        "cache_root": str(cache_root) if cache_root else None,
        "note": (
            "IDRiD / Messidor-2 / DeepDRiD evaluated as unseen external tests only. "
            "Temperature T fit on APTOS val only."
        ),
        "bootstrap": {"n_boot": n_boot, "seed": 42, "alpha": 0.05},
    }

    def _eval_split(
        name: str,
        dataset: Any,
        include_splits: bool = False,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        loader = build_dataloader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        pred = predict_loader(model, loader, device, head=head)
        metrics = compute_metrics(
            pred["y_true"],
            pred["y_pred"],
            num_classes=num_classes,
            probs=pred["probs"],
        )
        cis = bootstrap_metric_cis(
            pred["y_true"],
            pred["y_pred"],
            num_classes=num_classes,
            n_boot=n_boot,
            seed=42,
            probs=pred["probs"],
        )
        metrics["bootstrap_cis"] = cis

        if temperature is not None:
            scaled = temperature_scale_probs(pred["logits"], temperature, head=head)
            metrics["ece_temperature_scaled"] = expected_calibration_error(
                pred["y_true"], scaled
            )
            metrics["temperature"] = float(temperature)

        save_json(out_dir / f"{name}_metrics.json", metrics)
        pred_payload: dict[str, Any] = {
            "ids": pred["ids"],
            "y_true": pred["y_true"],
            "y_pred": pred["y_pred"],
            "probs": pred["probs"],
            "logits": pred["logits"],
        }
        if include_splits:
            pred_payload["splits"] = pred["splits"]
        save_json(out_dir / f"{name}_predictions.json", pred_payload)
        return {"n": len(dataset), "metrics": metrics, "pred": pred}

    # --- Fit temperature on APTOS val (never on externals / test) ---
    aptos_kw = {k: v for k, v in aptos_cfg.items() if k != "root"}
    aptos_val = build_dataset(
        "aptos",
        root=roots["aptos"],
        split="val",
        transform=eval_tf,
        cache_dir=aptos_cache_dir(cache_root, prep_name, "val") if use_cache else None,
        **aptos_kw,
    )
    val_loader = build_dataloader(
        aptos_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_pred = predict_loader(model, val_loader, device, head=head)
    temperature = fit_temperature(val_pred["logits"], val_pred["y_true"], head=head)
    report["temperature"] = {
        "value": temperature,
        "fit_on": "aptos_val",
        "n": len(aptos_val),
    }
    print(f"Temperature (APTOS val NLL): T={temperature:.4f}")

    # A. APTOS test
    aptos_test = build_dataset(
        "aptos",
        root=roots["aptos"],
        split="test",
        transform=eval_tf,
        cache_dir=aptos_cache_dir(cache_root, prep_name, "test") if use_cache else None,
        **aptos_kw,
    )
    aptos_block = _eval_split("aptos_test", aptos_test, temperature=temperature)
    report["aptos_test"] = {"n": aptos_block["n"], "metrics": aptos_block["metrics"]}

    # B. IDRiD external
    try:
        idrid_ds = build_dataset(
            "idrid",
            root=roots["idrid"],
            split="all",
            transform=eval_tf,
            cache_dir=idrid_cache_dir(cache_root, prep_name) if use_cache else None,
            **{k: v for k, v in idrid_cfg.items() if k != "root"},
        )
        idrid_block = _eval_split(
            "idrid_external", idrid_ds, include_splits=True, temperature=temperature
        )
        report["idrid_external"] = {"n": idrid_block["n"], "metrics": idrid_block["metrics"]}
    except Exception as exc:
        if not skip_missing_external:
            raise
        report["idrid_external"] = {"error": str(exc)}
        print(f"IDRiD skipped: {exc}")

    # C. Messidor-2 external
    try:
        messidor_ds = build_dataset(
            "messidor2",
            root=roots["messidor2"],
            split="all",
            transform=eval_tf,
            cache_dir=messidor2_cache_dir(cache_root, prep_name) if use_cache else None,
            **{k: v for k, v in messidor_cfg.items() if k != "root"},
        )
        # Prefer online load if cache missing
        if use_cache:
            cache_d = messidor2_cache_dir(cache_root, prep_name)
            assert cache_d is not None
            if not cache_d.is_dir() or not any(cache_d.glob("*.png")):
                messidor_ds = build_dataset(
                    "messidor2",
                    root=roots["messidor2"],
                    split="all",
                    transform=build_preprocess(
                        prep_name,
                        image_size=prep_cfg["image_size"],
                        train=False,
                        cached=False,
                        clahe_clip_limit=prep_cfg.get("clahe_clip_limit", 2.0),
                        clahe_tile_grid_size=tuple(
                            prep_cfg.get("clahe_tile_grid_size", [8, 8])
                        ),
                    ),
                    cache_dir=None,
                    **{k: v for k, v in messidor_cfg.items() if k != "root"},
                )
        m_block = _eval_split(
            "messidor2_external", messidor_ds, temperature=temperature
        )
        report["messidor2_external"] = {"n": m_block["n"], "metrics": m_block["metrics"]}
    except Exception as exc:
        if not skip_missing_external:
            raise
        report["messidor2_external"] = {"error": str(exc)}
        print(f"Messidor-2 skipped: {exc}")

    # D. DeepDRiD external (optional)
    try:
        deep_ds = build_dataset(
            "deepdrid",
            root=roots["deepdrid"],
            split="all",
            transform=build_preprocess(
                prep_name,
                image_size=prep_cfg["image_size"],
                train=False,
                cached=False,
                clahe_clip_limit=prep_cfg.get("clahe_clip_limit", 2.0),
                clahe_tile_grid_size=tuple(prep_cfg.get("clahe_tile_grid_size", [8, 8])),
            ),
            cache_dir=None,
            **{k: v for k, v in deepdrid_cfg.items() if k != "root"},
        )
        d_block = _eval_split(
            "deepdrid_external", deep_ds, include_splits=True, temperature=temperature
        )
        report["deepdrid_external"] = {"n": d_block["n"], "metrics": d_block["metrics"]}
    except Exception as exc:
        if not skip_missing_external:
            raise
        report["deepdrid_external"] = {"error": str(exc)}
        print(f"DeepDRiD skipped: {exc}")

    save_json(out_dir / "eval_report.json", report)
    exp_dir = project_root / "experiments" / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_json(exp_dir / "eval_report.json", report)

    def _print_block(title: str, block: dict[str, Any] | None) -> None:
        if not block or "metrics" not in block:
            return
        m = block["metrics"]
        print(f"\n=== {title} (n={block.get('n')}) ===")
        print(
            f"Acc={m['accuracy']:.4f}  "
            f"MacroF1={m['macro_f1']:.4f}  "
            f"QWK={m['quadratic_weighted_kappa']:.4f}  "
            f"MAE={m['mae']:.4f}  "
            f"MildF1={m.get('mild_dr_f1', float('nan')):.4f}  "
            f"MildRec={m['mild_dr_recall']:.4f}  "
            f"ECE={m.get('ece', float('nan')):.4f}  "
            f"ECE_T={m.get('ece_temperature_scaled', float('nan')):.4f}"
        )
        qwk_ci = m["bootstrap_cis"]["quadratic_weighted_kappa"]
        print(f"QWK 95% CI [{qwk_ci['ci_low']:.4f}, {qwk_ci['ci_high']:.4f}]")

    _print_block("APTOS test", report.get("aptos_test"))
    _print_block("IDRiD external", report.get("idrid_external"))
    _print_block("Messidor-2 external", report.get("messidor2_external"))
    _print_block("DeepDRiD external", report.get("deepdrid_external"))
    print(f"Report saved to: {out_dir / 'eval_report.json'}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DR checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_efficientnet_b0_rgb.yaml",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root) if args.project_root else _project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(config_path)
    evaluate(
        config,
        checkpoint=args.checkpoint,
        project_root=project_root,
        results_dir=args.results_dir,
        n_boot=args.n_boot,
    )


if __name__ == "__main__":
    main()
