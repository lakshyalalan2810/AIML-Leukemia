"""Out-of-fold (OOF) cross-validation pooling, statistics, and model comparison."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import compute_classification_metrics
from .subject_metrics import compute_subject_metrics
from src.data.registry import DATASET_REGISTRY, normalize_dataset


REVIEW_SEED = 42
EXPECTED_FOLDS = frozenset(range(5))
EXPECTED_ACTIVE_DEV = 10655


class OOFPool:
    """Strictly consolidates the five locked seed-42 outer folds."""

    def __init__(self, model_name: str, condition: str, dataset: str = "cnmc") -> None:
        self.model_name = model_name
        self.condition = condition
        self.dataset = normalize_dataset(dataset)
        # keyed by (seed, fold) -> list of record dicts
        self.fits: dict[tuple[int, int], list[dict[str, Any]]] = {}

    def add_fit_predictions(
        self, seed: int, fold: int, records: Sequence[Mapping[str, Any]], expected_protocol_fingerprint: str,
    ) -> None:
        if seed != REVIEW_SEED or fold not in EXPECTED_FOLDS:
            raise ValueError(f"Unexpected OOF fit seed/fold: seed={seed}, fold={fold}")
        if (seed, fold) in self.fits:
            raise ValueError(f"Duplicate OOF fit for seed={seed}, fold={fold}")
        if not records:
            raise ValueError(f"Empty OOF fit for seed={seed}, fold={fold}")
        for record in records:
            if int(record.get("seed", -1)) != seed or int(record.get("outer_fold", -1)) != fold:
                raise ValueError("OOF row seed/fold disagrees with its filename")
            if record.get("model") != self.model_name or record.get("condition") != self.condition:
                raise ValueError("OOF row has an unexpected model or condition")
            if normalize_dataset(str(record.get("dataset", ""))) != self.dataset:
                raise ValueError("OOF row has an unexpected or missing dataset")
            protocol = record.get("protocol_fingerprint")
            if not expected_protocol_fingerprint or protocol != expected_protocol_fingerprint:
                raise ValueError("OOF row has an unexpected or missing protocol")
        self.fits[(seed, fold)] = [dict(r) for r in records]

    def is_complete(self) -> bool:
        return set(self.fits) == {(REVIEW_SEED, fold) for fold in EXPECTED_FOLDS}

    def validate(
        self,
        inventory: Mapping[str, Mapping[str, Any]],
        expected_fold_by_sample: Mapping[str, int] | None = None,
    ) -> None:
        if not self.is_complete():
            found = sorted(fold for seed, fold in self.fits if seed == REVIEW_SEED)
            raise ValueError(f"Seed 42 requires exactly folds 0-4; found {found}")

        records = self.get_seed_pooled(REVIEW_SEED)
        ids = [record["sample_id"] for record in records]
        expected_count = int(DATASET_REGISTRY[self.dataset]["expected_oof_count"])
        if len(records) != expected_count:
            raise ValueError(f"Expected {expected_count} pooled OOF rows for {self.dataset}, found {len(records)}")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate sample in pooled OOF predictions")

        active_ids = {
            sid for sid, row in inventory.items()
            if row.get("modeling_eligible", row.get("eligible", True))
            and (self.dataset != "cnmc" or row.get("phase") == "training")
        }
        if len(active_ids) != expected_count:
            raise ValueError(f"Expected {expected_count} active inventory IDs for {self.dataset}, found {len(active_ids)}")
        predicted = set(ids)
        unknown = predicted - set(inventory)
        if unknown:
            raise ValueError(f"Unknown OOF sample(s): {sorted(unknown)[:3]}")
        forbidden = {
            sid for sid in predicted
            if (self.dataset == "cnmc" and inventory[sid].get("phase") != "training")
            or not inventory[sid].get("modeling_eligible", inventory[sid].get("eligible", True))
        }
        if forbidden:
            raise ValueError(f"OOF contains quarantined or non-development sample(s): {sorted(forbidden)[:3]}")
        if predicted != active_ids:
            raise ValueError(
                f"OOF ID set mismatch: missing={len(active_ids - predicted)}, extra={len(predicted - active_ids)}"
            )
        for (seed, fold), fold_records in self.fits.items():
            for record in fold_records:
                sid = record["sample_id"]
                if int(record["true_label"]) != int(inventory[sid]["label"]):
                    raise ValueError(f"OOF true label disagrees with inventory for {sid}")
                if expected_fold_by_sample is not None and expected_fold_by_sample.get(sid) != fold:
                    raise ValueError(f"OOF sample is assigned to the wrong outer fold: {sid}")

    def get_seed_pooled(self, seed: int) -> list[dict[str, Any]]:
        pooled = []
        for (s, f), records in sorted(self.fits.items()):
            if s == seed:
                pooled.extend(records)
        return pooled

    def evaluate_all(self, threshold: float = 0.5) -> dict[str, Any]:
        """Compute the locked single-seed summary statistics."""
        seeds = sorted({s for (s, f) in self.fits})
        seed_cell_metrics = []
        seed_group_metrics = []

        for seed in seeds:
            records = self.get_seed_pooled(seed)
            if not records:
                continue
            y_true = [int(r["true_label"]) for r in records]
            y_prob = [float(r["probability"]) for r in records]

            cell_m = compute_classification_metrics(y_true, y_prob, threshold=threshold)
            seed_cell_metrics.append(cell_m)
            if self.dataset == "cnmc":
                seed_group_metrics.append(compute_subject_metrics(records, threshold=threshold))
            elif self.dataset == "allidb2":
                seed_group_metrics.append(compute_parent_metrics(records, threshold=threshold))

        def aggregate_metrics(metric_list: list[dict[str, Any]]) -> dict[str, Any]:
            if not metric_list:
                return {}
            keys = [k for k in metric_list[0] if isinstance(metric_list[0][k], (int, float))]
            res = {}
            for k in keys:
                vals = [m[k] for m in metric_list if k in m]
                res[f"{k}_mean"] = float(np.mean(vals))
                res[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            return res

        sample_level = aggregate_metrics(seed_cell_metrics)
        result = {
            "model_name": self.model_name,
            "completed_fits": len(self.fits),
            "dataset": self.dataset,
            "cell_level": sample_level,
            "subject_level": aggregate_metrics(seed_group_metrics) if self.dataset == "cnmc" else {},
            "parent_field_level": aggregate_metrics(seed_group_metrics) if self.dataset == "allidb2" else {},
            "resolution_stratified": (compute_stratified_metrics(
                self.get_seed_pooled(REVIEW_SEED), "resolution_family", threshold
            ) if self.dataset == "allidb1" else {}),
            "nuisance_sensitivity": (compute_nuisance_sensitivity(
                self.get_seed_pooled(REVIEW_SEED)
            ) if self.dataset == "allidb1" else {}),
            "per_seed_cell": seed_cell_metrics,
            "per_seed_subject": seed_group_metrics if self.dataset == "cnmc" else [],
            "per_seed_parent_field": seed_group_metrics if self.dataset == "allidb2" else [],
        }
        if self.dataset == "allidb1":
            result["image_level"] = sample_level
        elif self.dataset == "allidb2":
            result["crop_level"] = sample_level
        return result


def compute_stratified_metrics(
    records: Sequence[Mapping[str, Any]], field: str, threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    strata: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if field not in record:
            raise ValueError(f"OOF row is missing nuisance field {field}")
        strata.setdefault(str(record[field]), []).append(record)
    return {
        key: compute_classification_metrics(
            [int(row["true_label"]) for row in rows], [float(row["probability"]) for row in rows], threshold
        )
        for key, rows in sorted(strata.items())
    }


def compute_nuisance_sensitivity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probabilities = np.asarray([float(row["probability"]) for row in records], dtype=float)
    diagnostics = {}
    for field in ("padding_fraction", "width", "height", "aspect_ratio"):
        values = np.asarray([float(row[field]) for row in records], dtype=float)
        correlation = float(np.corrcoef(values, probabilities)[0, 1]) if len(set(values)) > 1 else float("nan")
        diagnostics[field] = {
            "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
            "probability_pearson_r": correlation,
        }
    return diagnostics


def aggregate_parent_predictions(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        parent = record.get("source_parent_id")
        if not parent:
            raise ValueError("ALL-IDB2 OOF row is missing source_parent_id")
        grouped.setdefault(str(parent), []).append(record)
    result = []
    for parent, crops in sorted(grouped.items()):
        labels = {int(row["true_label"]) for row in crops}
        if len(labels) != 1:
            raise ValueError(f"Inconsistent labels for parent field {parent}")
        result.append({
            "source_parent_id": parent,
            "true_label": labels.pop(),
            "probability": float(np.mean([float(row["probability"]) for row in crops])),
            "crop_count": len(crops),
        })
    return result


def compute_parent_metrics(records: Sequence[Mapping[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    parents = aggregate_parent_predictions(records)
    metrics = compute_classification_metrics(
        [row["true_label"] for row in parents], [row["probability"] for row in parents], threshold
    )
    allowed = {"accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc",
               "roc_auc", "pr_auc", "tp", "tn", "fp", "fn", "total"}
    result = {key: value for key, value in metrics.items() if key in allowed}
    result.update(num_parent_fields=len(parents), parent_predictions=parents)
    return result


def summarize_oof_metrics(
    convnext_eval: Mapping[str, Any] | None,
    efficientnet_eval: Mapping[str, Any] | None,
) -> str:
    """Format markdown comparison table between ConvNeXtV2 and EfficientNet-B1."""
    dataset = (convnext_eval or efficientnet_eval or {}).get("dataset", "cnmc")
    if dataset == "cnmc":
        levels = (("cell_level", "per_seed_cell", "ConvNeXtV2-Tiny (Cell)", "EfficientNet-B1 (Cell)"),
                  ("subject_level", "per_seed_subject", "ConvNeXtV2-Tiny (Subject)", "EfficientNet-B1 (Subject)"))
    elif dataset == "allidb1":
        levels = (("cell_level", "per_seed_cell", "ConvNeXtV2-Tiny (Field)", "EfficientNet-B1 (Field)"),)
    else:
        levels = (("cell_level", "per_seed_cell", "ConvNeXtV2-Tiny (Crop)", "EfficientNet-B1 (Crop)"),)
    header = ["Metric"] + [label for level in levels for label in level[2:]]
    lines = [
        "# Model Comparison: ConvNeXtV2-Tiny vs EfficientNet-B1 (Development OOF)", "",
        "Pooled OOF values are single-seed point estimates; fold variability is reported separately.", "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(":---" for _ in header) + " |",
    ]

    metrics_to_show = [
        ("Accuracy", "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Precision", "precision"),
        ("F1 Score", "f1"),
        ("MCC", "mcc"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("NLL", "nll"),
        ("Brier", "brier"),
        ("ECE-10", "ece_10"),
    ]

    for label, key in metrics_to_show:
        values = []
        for level_key, per_seed_key, _, _ in levels:
            for evaluation in (convnext_eval, efficientnet_eval):
                items = evaluation.get(per_seed_key, []) if evaluation else []
                value = items[0].get(key) if items else None
                values.append(f"{value:.4f}" if isinstance(value, (int, float)) and np.isfinite(value) else "N/A")
        lines.append("| **" + label + "** | " + " | ".join(values) + " |")

    return "\n".join(lines)
