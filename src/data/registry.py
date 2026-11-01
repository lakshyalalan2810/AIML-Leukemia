"""Executable dataset registry with no cross-dataset assumptions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .allidb import load_inventory
from .allidb_protocol import load_eligibility_decision, load_split_manifest
from .cnmc import load_cnmc_inventory
from .splits import load_splits, validate_development_inventory

MODELS = ("convnextv2_tiny", "efficientnet_b1")
CELL_METRICS = (
    "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc",
    "roc_auc", "pr_auc", "nll", "brier", "ece_10",
)
DATASET_REGISTRY = {
    "cnmc": {
        "name": "C-NMC 2019",
        "class_mapping": {"HEM": 0, "ALL": 1},
        "inventory": "data/cnmc/cnmc_inventory.json",
        "eligibility_manifest": "data/cnmc/cnmc_inventory.json",
        "split_manifest": "data/cnmc/cnmc_splits.json",
        "group_key": "final_group_id",
        "preprocessing": "direct_bicubic_imagenet",
        "expected_oof_count": 10655,
        "supported_metrics": CELL_METRICS + ("subject_level",),
        "supported_models": MODELS,
        "scientific_claim": "subject/group-disjoint cross-validation",
    },
    "allidb1": {
        "name": "ALL-IDB1",
        "class_mapping": {"healthy_field": 0, "all_field": 1},
        "inventory": "data/allidb1/allidb1_inventory.json",
        "eligibility_manifest": "data/allidb1/allidb1_eligibility_v1.json",
        "split_manifest": "data/allidb1/allidb1_splits_v1.json",
        "group_key": "canonical full-field / exact-duplicate component",
        "preprocessing": "aspect_preserving_bicubic_fit_imagenet_mean_padding",
        "expected_oof_count": 107,
        "supported_metrics": CELL_METRICS + ("resolution_stratified", "nuisance_sensitivity"),
        "supported_models": MODELS,
        "scientific_claim": "field-disjoint cross-validation; patient and specimen identities are unavailable",
        "cross_dataset_policy": "separate training/evaluation only; not external validation of ALL-IDB2; no pooling",
    },
    "allidb2": {
        "name": "ALL-IDB2",
        "class_mapping": {"non_blast": 0, "blast": 1},
        "inventory": "data/allidb2/allidb2_inventory.json",
        "eligibility_manifest": "data/allidb2/allidb2_eligibility_v1.json",
        "split_manifest": "data/allidb2/allidb2_splits_v1.json",
        "group_key": "source_parent_id",
        "preprocessing": "aspect_preserving_bicubic_fit_imagenet_mean_padding",
        "expected_oof_count": 258,
        "supported_metrics": CELL_METRICS + ("parent_field_level",),
        "supported_models": MODELS,
        "scientific_claim": "parent-field-disjoint cross-validation; patient and specimen identities are unavailable",
        "cross_dataset_policy": "separate training/evaluation only; not external validation of ALL-IDB1; no pooling",
    },
}


def normalize_dataset(value: str) -> str:
    aliases = {"c-nmc": "cnmc", "all-idb1": "allidb1", "all_idb1": "allidb1",
               "all-idb2": "allidb2", "all_idb2": "allidb2"}
    return aliases.get(value.lower(), value.lower())


def load_dataset_bundle(dataset: str, root: str | Path | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    dataset = normalize_dataset(dataset)
    if dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset}")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    spec = DATASET_REGISTRY[dataset]
    if dataset == "cnmc":
        inventory = load_cnmc_inventory(root / "data" / "cnmc", root / "data" / "raw" / "cnmc")
        splits = load_splits(root / spec["split_manifest"])
        validate_development_inventory(inventory, splits)
        return inventory, splits

    inventory_document = load_inventory(root / spec["inventory"])
    manifest = load_split_manifest(root / spec["split_manifest"])
    decision = load_eligibility_decision(root / spec["eligibility_manifest"])
    if decision.get("decision_sha256") != manifest["eligibility_decision_hash"]:
        raise ValueError("Eligibility decision/split manifest mismatch")
    source_by_id = {row["sample_id"]: row for row in inventory_document["records"]}
    inventory = {}
    for split_row in manifest["records"]:
        row = {**source_by_id[split_row["sample_id"]], **split_row}
        row["modeling_eligible"] = row["eligible"]
        row["phase"] = "training"
        row["image_path"] = str(root / source_by_id[row["sample_id"]]["relative_path"])
        inventory[row["sample_id"]] = row
    return inventory, manifest
