"""Comprehensive classification, calibration, and curve metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across n equal-width bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return float("nan")

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute all cell-level metrics required by the evaluation protocol."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError("y_true and y_prob must have equal length")
    if np.any(~np.isfinite(y_prob_arr)) or np.any((y_prob_arr < 0) | (y_prob_arr > 1)):
        raise ValueError("y_prob must contain finite probabilities in [0, 1]")
    if np.any(~np.isin(y_true_arr, [0, 1])):
        raise ValueError("y_true must contain only binary labels 0 and 1")

    if len(y_true_arr) == 0:
        nan = float("nan")
        return {
            "accuracy": nan, "balanced_accuracy": nan, "sensitivity": nan,
            "specificity": nan, "precision": nan, "f1": nan, "mcc": nan,
            "roc_auc": nan, "pr_auc": nan, "nll": nan, "brier": nan,
            "ece_10": nan, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0,
        }

    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

    # Core metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    has_both_classes = len(np.unique(y_true_arr)) == 2
    bacc = float(balanced_accuracy_score(y_true_arr, y_pred_arr)) if has_both_classes else float("nan")
    sens = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    mcc = float(matthews_corrcoef(y_true_arr, y_pred_arr)) if has_both_classes else 0.0

    # Curve / probability metrics
    if has_both_classes:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        prec_curve, rec_curve, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        pr_auc = float(auc(rec_curve, prec_curve))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    # Calibration & probability losses
    eps = 1e-7
    clipped_probs = np.clip(y_prob_arr, eps, 1.0 - eps)
    nll = float(log_loss(y_true_arr, clipped_probs, labels=[0, 1]))

    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    ece = float(compute_expected_calibration_error(y_true_arr, y_prob_arr, n_bins=10))

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "nll": nll,
        "brier": brier,
        "ece_10": ece,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": len(y_true_arr),
    }


def compute_calibration_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
) -> dict[str, Any]:
    """Compute calibration curve data and metrics."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    base = compute_classification_metrics(y_true_arr, y_prob_arr)
    if len(y_true_arr) == 0 or len(np.unique(y_true_arr)) < 2:
        return {
            "brier": base["brier"],
            "ece_10": base["ece_10"],
            "roc_curve": {"fpr": [], "tpr": [], "thresholds": []},
            "pr_curve": {"precision": [], "recall": [], "thresholds": []},
        }

    fpr, tpr, roc_thresh = roc_curve(y_true_arr, y_prob_arr)
    precision, recall, pr_thresh = precision_recall_curve(y_true_arr, y_prob_arr)

    return {
        "brier": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "ece_10": float(compute_expected_calibration_error(y_true_arr, y_prob_arr, n_bins=10)),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresh.tolist(),
        },
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresh.tolist(),
        },
    }
