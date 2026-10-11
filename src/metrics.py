"""
Evaluation metrics for DR grading and cross-dataset generalization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

METRIC_REGISTRY = {
    "accuracy": "accuracy",
    "quadratic_weighted_kappa": "quadratic_weighted_kappa",
    "macro_f1": "macro_f1",
    "mae": "mae",
    "per_class_precision": "per_class_precision",
    "per_class_recall": "per_class_recall",
    "per_class_f1": "per_class_f1",
    "mild_dr_recall": "mild_dr_recall",
    "mild_dr_precision": "mild_dr_precision",
    "mild_dr_f1": "mild_dr_f1",
    "grade1_f1": "mild_dr_f1",
    "confusion_0_1": "confusion_0_1",
    "confusion_1_2": "confusion_1_2",
    "ece": "ece",
}

# Scalar metrics suitable for bootstrap CIs
BOOTSTRAP_METRICS = (
    "accuracy",
    "macro_f1",
    "quadratic_weighted_kappa",
    "mae",
    "mild_dr_recall",
    "mild_dr_precision",
    "mild_dr_f1",
)


def _as_int_arrays(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    return yt, yp


def accuracy(y_true: Any, y_pred: Any) -> float:
    yt, yp = _as_int_arrays(y_true, y_pred)
    return float(accuracy_score(yt, yp))


def quadratic_weighted_kappa(y_true: Any, y_pred: Any, num_classes: int = 5) -> float:
    """Quadratic weighted Cohen's kappa for ordinal DR grades."""
    yt, yp = _as_int_arrays(y_true, y_pred)
    conf = sk_confusion_matrix(yt, yp, labels=list(range(num_classes))).astype(np.float64)
    n = conf.sum()
    if n == 0:
        return 0.0

    hist_true = conf.sum(axis=1)
    hist_pred = conf.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / n

    indices = np.arange(num_classes, dtype=np.float64)
    weights = (indices[:, None] - indices[None, :]) ** 2 / ((num_classes - 1) ** 2)

    numer = (weights * conf).sum()
    denom = (weights * expected).sum()
    if denom == 0:
        return 1.0 if numer == 0 else 0.0
    return float(1.0 - numer / denom)


def macro_f1(y_true: Any, y_pred: Any, num_classes: int = 5) -> float:
    yt, yp = _as_int_arrays(y_true, y_pred)
    return float(
        f1_score(yt, yp, labels=list(range(num_classes)), average="macro", zero_division=0)
    )


def mae(y_true: Any, y_pred: Any) -> float:
    yt, yp = _as_int_arrays(y_true, y_pred)
    return float(mean_absolute_error(yt, yp))


def per_class_precision(y_true: Any, y_pred: Any, num_classes: int = 5) -> list[float]:
    yt, yp = _as_int_arrays(y_true, y_pred)
    precision, _, _, _ = precision_recall_fscore_support(
        yt, yp, labels=list(range(num_classes)), average=None, zero_division=0
    )
    return [float(x) for x in precision]


def per_class_recall(y_true: Any, y_pred: Any, num_classes: int = 5) -> list[float]:
    yt, yp = _as_int_arrays(y_true, y_pred)
    _, recall, _, _ = precision_recall_fscore_support(
        yt, yp, labels=list(range(num_classes)), average=None, zero_division=0
    )
    return [float(x) for x in recall]


def per_class_f1(y_true: Any, y_pred: Any, num_classes: int = 5) -> list[float]:
    yt, yp = _as_int_arrays(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=list(range(num_classes)), average=None, zero_division=0
    )
    return [float(x) for x in f1]


def mild_dr_recall(y_true: Any, y_pred: Any, num_classes: int = 5) -> float:
    """Recall for Mild NPDR (class 1)."""
    recalls = per_class_recall(y_true, y_pred, num_classes=num_classes)
    return float(recalls[1])


def mild_dr_precision(y_true: Any, y_pred: Any, num_classes: int = 5) -> float:
    """Precision for Mild NPDR (class 1)."""
    precisions = per_class_precision(y_true, y_pred, num_classes=num_classes)
    return float(precisions[1])


def mild_dr_f1(y_true: Any, y_pred: Any, num_classes: int = 5) -> float:
    """F1 for Mild NPDR / grade-1 (class 1)."""
    f1s = per_class_f1(y_true, y_pred, num_classes=num_classes)
    return float(f1s[1])


def confusion_matrix(y_true: Any, y_pred: Any, num_classes: int = 5) -> list[list[int]]:
    yt, yp = _as_int_arrays(y_true, y_pred)
    cm = sk_confusion_matrix(yt, yp, labels=list(range(num_classes)))
    return cm.astype(int).tolist()


def confusion_pair_count(y_true: Any, y_pred: Any, a: int, b: int, num_classes: int = 5) -> dict[str, int]:
    """Count bidirectional confusions between classes a and b."""
    cm = np.asarray(confusion_matrix(y_true, y_pred, num_classes=num_classes))
    a_as_b = int(cm[a, b])
    b_as_a = int(cm[b, a])
    return {
        f"{a}_predicted_as_{b}": a_as_b,
        f"{b}_predicted_as_{a}": b_as_a,
        "total_bidirectional": a_as_b + b_as_a,
    }


def confusion_0_1(y_true: Any, y_pred: Any, num_classes: int = 5) -> dict[str, int]:
    return confusion_pair_count(y_true, y_pred, 0, 1, num_classes=num_classes)


def confusion_1_2(y_true: Any, y_pred: Any, num_classes: int = 5) -> dict[str, int]:
    return confusion_pair_count(y_true, y_pred, 1, 2, num_classes=num_classes)


def expected_calibration_error(
    y_true: Any,
    probs: Any,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error over max-probability confidence.

    probs: (N, C) class probabilities.
    """
    yt = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != yt.shape[0]:
        raise ValueError(f"probs shape {p.shape} incompatible with y_true {yt.shape}")
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == yt).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(yt)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_metrics(
    y_true: Any,
    y_pred: Any,
    metric_names: list[str] | None = None,
    num_classes: int = 5,
    probs: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute a full metric dict for reporting."""
    yt, yp = _as_int_arrays(y_true, y_pred)
    results: dict[str, Any] = {
        "n_samples": int(len(yt)),
        "accuracy": accuracy(yt, yp),
        "macro_f1": macro_f1(yt, yp, num_classes=num_classes),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(yt, yp, num_classes=num_classes),
        "mae": mae(yt, yp),
        "per_class_precision": per_class_precision(yt, yp, num_classes=num_classes),
        "per_class_recall": per_class_recall(yt, yp, num_classes=num_classes),
        "per_class_f1": per_class_f1(yt, yp, num_classes=num_classes),
        "mild_dr_recall": mild_dr_recall(yt, yp, num_classes=num_classes),
        "mild_dr_precision": mild_dr_precision(yt, yp, num_classes=num_classes),
        "mild_dr_f1": mild_dr_f1(yt, yp, num_classes=num_classes),
        "grade1_f1": mild_dr_f1(yt, yp, num_classes=num_classes),
        "confusion_matrix": confusion_matrix(yt, yp, num_classes=num_classes),
        "confusion_0_1": confusion_0_1(yt, yp, num_classes=num_classes),
        "confusion_1_2": confusion_1_2(yt, yp, num_classes=num_classes),
    }
    if probs is not None:
        results["ece"] = expected_calibration_error(yt, probs)
    if metric_names:
        # Always keep core keys; metric_names acts as a soft filter for extras only
        pass
    return results


def bootstrap_metric_cis(
    y_true: Any,
    y_pred: Any,
    num_classes: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    probs: Any | None = None,
) -> dict[str, dict[str, float]]:
    """
    Nonparametric bootstrap 95% CIs for scalar metrics.

    Resamples indices with replacement. Optional ECE if probs provided.
    """
    if int(n_boot) <= 0:
        return {}
    yt, yp = _as_int_arrays(y_true, y_pred)
    n = len(yt)
    if n == 0:
        return {}

    rng = np.random.default_rng(seed)
    keys = list(BOOTSTRAP_METRICS)
    if probs is not None:
        keys = keys + ["ece"]
        p_all = np.asarray(probs, dtype=np.float64)

    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(
            yt[idx],
            yp[idx],
            num_classes=num_classes,
            probs=p_all[idx] if probs is not None else None,
        )
        for k in keys:
            samples[k].append(float(m[k]))

    out: dict[str, dict[str, float]] = {}
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[k] = {
            "mean": float(arr.mean()),
            "ci_low": float(np.percentile(arr, lo_q)),
            "ci_high": float(np.percentile(arr, hi_q)),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }
    return out


def temperature_scale_probs(
    logits: Any,
    temperature: float,
    head: str = "ce",
) -> np.ndarray:
    """
    Apply temperature scaling to logits then convert to class probabilities.

    For CE: softmax(logits / T).
    For CORAL: sigmoid(logits / T) then cumulative→class probs (same as logits_to_probs).
    """
    import torch
    from src.models import logits_to_probs

    t = max(float(temperature), 1e-6)
    z = torch.as_tensor(logits, dtype=torch.float32)
    return logits_to_probs(z / t, head=head).cpu().numpy()


def fit_temperature(
    logits: Any,
    y_true: Any,
    head: str = "ce",
    init_t: float = 1.0,
    max_iter: int = 50,
) -> float:
    """
    Fit a scalar temperature on a validation set by minimizing NLL.

    Uses a dense log-grid search (stable for both CE and CORAL heads),
    then optional LBFGS polish. Clamped to [0.05, 20].
    """
    import torch
    import torch.nn.functional as F
    from src.models import logits_to_probs

    yt = torch.as_tensor(y_true, dtype=torch.long)
    z = torch.as_tensor(logits, dtype=torch.float32)

    def _nll(t: float) -> float:
        with torch.no_grad():
            probs = logits_to_probs(z / max(t, 1e-6), head=head).clamp(min=1e-8)
            return float(F.nll_loss(probs.log(), yt).item())

    # Coarse-to-fine grid on log-T
    grid = np.unique(
        np.concatenate(
            [
                np.array([init_t], dtype=np.float64),
                np.logspace(np.log10(0.05), np.log10(20.0), 40),
            ]
        )
    )
    best_t = 1.0
    best_nll = float("inf")
    for t in grid:
        nll = _nll(float(t))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    # Local refine around best
    lo = max(0.05, best_t * 0.5)
    hi = min(20.0, best_t * 2.0)
    for t in np.linspace(lo, hi, 21):
        nll = _nll(float(t))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    # Optional LBFGS polish (CE usually; CORAL may be flat)
    try:
        log_t = torch.nn.Parameter(torch.tensor(float(np.log(best_t))))
        opt = torch.optim.LBFGS(
            [log_t], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def _closure() -> torch.Tensor:
            opt.zero_grad()
            t = torch.exp(log_t).clamp(min=0.05, max=20.0)
            probs = logits_to_probs(z / t, head=head).clamp(min=1e-8)
            nll = F.nll_loss(probs.log(), yt)
            nll.backward()
            return nll

        opt.step(_closure)
        with torch.no_grad():
            polished = float(torch.exp(log_t).clamp(min=0.05, max=20.0).item())
        if _nll(polished) <= best_nll + 1e-6:
            best_t = polished
    except Exception:
        pass

    return float(np.clip(best_t, 0.05, 20.0))
