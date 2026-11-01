"""Locked seed-42 review identity and job selection."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import stable_id_hash

REVIEW_SEED = 42
REVIEW_FOLDS = tuple(range(5))
DATASET_SLUG = "cnmc"


def review_jobs(seed: int = REVIEW_SEED, fold: int | None = None) -> list[tuple[int, int]]:
    if seed != REVIEW_SEED:
        raise ValueError(f"Locked review seed is {REVIEW_SEED}, got {seed}")
    if fold is not None and fold not in REVIEW_FOLDS:
        raise ValueError(f"Outer fold must be one of {REVIEW_FOLDS}, got {fold}")
    return [(seed, selected) for selected in ((fold,) if fold is not None else REVIEW_FOLDS)]


def validate_locked_config(
    config: Mapping[str, Any], model: str, recipe: str, input_size: int, parameter_count: int, condition: str,
    dataset: str = "cnmc",
) -> None:
    expected = {
        ("model", "name"): model,
        ("model", "backbone"): recipe,
        ("model", "input_size"): input_size,
        ("model", "parameter_count"): parameter_count,
        ("preprocessing", "clahe"): False,
        ("preprocessing", "stain_normalization"): False,
        ("preprocessing", "grayscale"): False,
        ("augmentation", "policy"): "deterministic_d4_only",
        ("training", "seeds"): [42],
        ("training", "outer_folds"): 5,
        ("selection", "criteria"): ["mcc", "balanced_accuracy", "nll", "epoch"],
        ("loss", "condition"): condition,
        ("outputs", "models_dir"): f"outputs/models/{model}",
        ("outputs", "predictions_dir"): f"outputs/predictions/{model}",
    }
    if dataset == "cnmc":
        expected.update({
            ("dataset", "name"): "C-NMC",
            ("dataset", "positive_label"): 1,
            ("dataset", "negative_label"): 0,
            ("dataset", "active_dev_cells"): 10655,
            ("dataset", "data_dir"): "data/cnmc",
        })
    mismatches = [f"{section}.{key}" for (section, key), value in expected.items()
                  if config.get(section, {}).get(key) != value]
    if mismatches:
        raise ValueError(f"Configuration violates the locked review protocol: {', '.join(mismatches)}")


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_code_fingerprint(model: str) -> str:
    root = Path(__file__).resolve().parents[2]
    files = [
        "src/data/augmentation.py", "src/data/allidb.py", "src/data/cnmc.py", "src/data/preprocessing.py",
        "src/data/registry.py", "src/data/splits.py",
        "src/evaluation/metrics.py", "src/training/artifacts.py", "src/training/losses.py",
        "src/training/protocol.py", "src/training/trainer.py", f"src/models/{model}.py",
        f"scripts/train_{'convnextv2' if model == 'convnextv2_tiny' else model}.py",
    ]
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def build_fit_metadata(
    config: Mapping[str, Any],
    splits_data: Mapping[str, Any],
    *,
    model: str,
    architecture: str,
    condition: str,
    seed: int,
    fold: int,
    class_counts: list[int],
    class_weights: list[float],
    outer_val_sample_ids: list[str],
    dataset: str = "cnmc",
    dataset_name: str = "C-NMC",
    class_order: Mapping[str, int] | None = None,
    code_fingerprint: str | None = None,
) -> dict:
    class_order = dict(class_order or {"HEM": 0, "ALL": 1})
    scientific_config = {
        "dataset": ({key: value for key, value in config["dataset"].items() if key not in {"root", "data_dir"}}
                    if dataset == "cnmc" else {"name": dataset_name, "class_mapping": class_order}),
        **{key: config[key] for key in (
            "model", "preprocessing", "augmentation", "training", "loss", "selection"
        )},
    }
    code_fingerprint = code_fingerprint or _implementation_code_fingerprint(model)
    configuration_identity = {
        "scientific_config": scientific_config,
        "condition": condition,
        "seed": seed,
        "fold": fold,
    }
    configuration_fingerprint = _fingerprint(configuration_identity)
    scientific_protocol_fingerprint = _fingerprint({
        "configuration_fingerprint": configuration_fingerprint,
        "split_schema": splits_data.get("schema_version", splits_data.get("protocol_version")),
        "split_id": splits_data.get("split_id", splits_data.get("manifest_sha256")),
        "inventory_identity": splits_data.get("inventory_identity", splits_data.get("inventory_sha256")),
        "outer_policy": splits_data.get("outer_policy", splits_data.get("splitter")),
        "outer_val_sample_ids_sha256": stable_id_hash(outer_val_sample_ids),
        "inner_split_policy": ("group_preserving_sha256_seed_fold_v1" if dataset == "cnmc"
                               else splits_data.get("inner_splitter")),
    })
    return {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "model": model,
        "architecture": architecture,
        "condition": condition,
        "seed": seed,
        "fold": fold,
        "input_resolution": int(config["model"]["input_size"]),
        "class_order": class_order,
        "loss_mode": "balanced_weighted_cross_entropy",
        "augmentation_mode": "deterministic_d4_inner_train_only",
        "pretrained": True,
        "max_epochs": int(config["training"]["max_epochs"]),
        "configuration_fingerprint": configuration_fingerprint,
        "scientific_protocol_fingerprint": scientific_protocol_fingerprint,
        "protocol_fingerprint": scientific_protocol_fingerprint,
        "code_fingerprint": code_fingerprint,
        "inner_train_class_counts": class_counts,
        "class_weights": class_weights,
        "outer_val_count": len(outer_val_sample_ids),
        "outer_val_sample_ids_sha256": stable_id_hash(outer_val_sample_ids),
    }


def fit_artifact_paths(repo_root: str | Path, config: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[Path, Path]:
    """Keep current model roots while adding dataset/condition identity below them."""
    root = Path(repo_root)
    fit_dir = (root / config["outputs"]["models_dir"] / metadata["dataset"] /
               metadata["condition"] / f"seed_{metadata['seed']}" / f"fold_{metadata['fold']}")
    oof = (root / config["outputs"]["predictions_dir"] / metadata["dataset"] /
           metadata["condition"] / f"seed_{metadata['seed']}" / f"oof_fold_{metadata['fold']}.csv")
    return fit_dir, oof
