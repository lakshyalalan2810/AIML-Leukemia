"""Subject-level aggregation and evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import compute_classification_metrics


def aggregate_subject_predictions(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate cell predictions to subject-level using mean probability.

    Each record must have 'subject_id', 'true_label', and 'probability' (or 'prob_all').
    """
    subject_probs = defaultdict(list)
    subject_labels = {}

    for r in records:
        sub_id = r.get("subject_id")
        if not sub_id:
            continue
        prob = r.get("probability", r.get("prob_all", 0.0))
        label = int(r["true_label"])

        subject_probs[sub_id].append(float(prob))
        if sub_id not in subject_labels:
            subject_labels[sub_id] = label
        elif subject_labels[sub_id] != label:
            # Multi-label conflict fallback: any ALL makes subject ALL
            if label == 1:
                subject_labels[sub_id] = 1

    subject_records = []
    for sub_id, probs in sorted(subject_probs.items()):
        mean_prob = float(np.mean(probs))
        true_label = subject_labels[sub_id]
        subject_records.append({
            "subject_id": sub_id,
            "true_label": true_label,
            "probability": mean_prob,
            "cell_count": len(probs),
        })

    return subject_records


def compute_subject_metrics(
    records: Sequence[Mapping[str, Any]],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute complete subject-level metrics after mean-probability aggregation."""
    subject_records = aggregate_subject_predictions(records)
    if not subject_records:
        metrics = compute_classification_metrics([], [], threshold=threshold)
        metrics.update({"num_subjects": 0, "subject_predictions": []})
        return metrics

    y_true = [r["true_label"] for r in subject_records]
    y_prob = [r["probability"] for r in subject_records]

    metrics = compute_classification_metrics(y_true, y_prob, threshold=threshold)
    metrics["num_subjects"] = len(subject_records)
    metrics["subject_predictions"] = subject_records
    return metrics
