"""Evaluation metrics, out-of-fold scoring, and subject-level aggregation."""

from .metrics import compute_classification_metrics, compute_calibration_metrics
from .subject_metrics import aggregate_subject_predictions, compute_subject_metrics
from .oof import OOFPool, aggregate_parent_predictions, compute_parent_metrics, summarize_oof_metrics

__all__ = [
    "compute_classification_metrics",
    "compute_calibration_metrics",
    "aggregate_subject_predictions",
    "compute_subject_metrics",
    "OOFPool",
    "aggregate_parent_predictions",
    "compute_parent_metrics",
    "summarize_oof_metrics",
]
