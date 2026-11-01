"""Frozen ALL-IDB eligibility and group-disjoint split protocols."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from .allidb import PARENT_MAPPING_PROTOCOL, file_sha256, load_inventory

SKLEARN_VERSION = "1.7.2"
SEED = 42
PROTOCOL_VERSION = "allidb_approved_cv_v1"
DECISION_VERSION = "allidb_eligibility_v1"
PARENT_MAP_SHA256 = "F04D6DE9F6B8C6C3542CC8A77B0D77411E5901B2CC4FE47CE50ADFB6BD73841B"
EXCLUSIONS = {
    "allidb1": {
        "allidb1_000108": "EXACT_DUPLICATE_OF_allidb1_000093",
    },
    "allidb2": {
        "allidb2_000256": "CONFIRMED_SAME_SOURCE_CONTENT_REDUNDANCY_WITH_allidb2_000222",
        "allidb2_000257": "EXACT_DUPLICATE_OF_allidb2_000223",
    },
}
RESTORED = {
    "allidb1": {},
    "allidb2": {
        "allidb2_000028": "DISTINCT_SAMPLE_DIFFERENT_PARENT_FROM_allidb2_000031",
        "allidb2_000031": "DISTINCT_SAMPLE_DIFFERENT_PARENT_FROM_allidb2_000028",
        "allidb2_000138": "DISTINCT_SAMPLE_DIFFERENT_PARENT_FROM_allidb2_000146",
        "allidb2_000146": "DISTINCT_SAMPLE_DIFFERENT_PARENT_FROM_allidb2_000138",
        "allidb2_000222": "CANONICAL_REPRESENTATIVE_OF_CONFIRMED_SAME_SOURCE_CONTENT",
    },
}
EXPECTED_ELIGIBLE = {"allidb1": 107, "allidb2": 258}


def _hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def preprocessing_protocol(dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "rgb": True,
        "geometry": "aspect_preserving_bicubic_fit_imagenet_mean_padding",
        "normalization": "imagenet_mean_std",
        "clahe": False,
    }


def preprocessing_protocol_hash(dataset: str) -> str:
    return _hash(preprocessing_protocol(dataset))


def _paths(root: Path, dataset: str) -> tuple[Path, Path, Path]:
    base = root / "data" / dataset
    return (
        base / f"{dataset}_inventory.json",
        base / f"{dataset}_eligibility_v1.json",
        base / f"{dataset}_splits_v1.json",
    )


def _read_parent_map(root: Path) -> list[dict[str, str]]:
    path = root / "data" / "allidb2" / "allidb2_parent_map.csv"
    if file_sha256(path).upper() != PARENT_MAP_SHA256:
        raise ValueError("ALL-IDB2 parent-map SHA-256 mismatch")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 260 or len({row["allidb2_sample_id"] for row in rows}) != 260:
        raise ValueError("ALL-IDB2 parent map must contain 260 unique child rows")
    if len({row["allidb1_parent_sample_id"] for row in rows}) != 102:
        raise ValueError("ALL-IDB2 parent map must contain 102 canonical parents")
    if {row["mapping_status"] for row in rows} != {"REPRODUCED_HIGH_CONFIDENCE"}:
        raise ValueError("ALL-IDB2 parent map contains unknown mappings")
    return rows


def build_eligibility_decision(root: str | Path, dataset: str) -> dict[str, Any]:
    root, dataset = Path(root), dataset.lower().replace("-", "")
    inventory_path, _, _ = _paths(root, dataset)
    inventory = load_inventory(inventory_path)
    parent_hash = PARENT_MAP_SHA256 if dataset == "allidb2" else None
    evidence = [
        {
            "path": f"data/{dataset}/{dataset}_exact_duplicates.csv",
            "sha256": file_sha256(root / "data" / dataset / f"{dataset}_exact_duplicates.csv").upper(),
        }
    ]
    if dataset == "allidb2":
        evidence.extend([
            {
                "path": "data/allidb2/allidb2_near_duplicates.csv",
                "sha256": file_sha256(root / "data" / "allidb2" / "allidb2_near_duplicates.csv").upper(),
            },
            {"path": "data/allidb2/allidb2_parent_map.csv", "sha256": PARENT_MAP_SHA256},
        ])
    decisions = []
    for row in inventory["records"]:
        sample_id = row["sample_id"]
        if sample_id in EXCLUSIONS[dataset]:
            state, reason = "EXCLUDED", EXCLUSIONS[dataset][sample_id]
        elif sample_id in RESTORED[dataset]:
            state, reason = "RESTORED_ELIGIBLE", RESTORED[dataset][sample_id]
        else:
            state, reason = "ELIGIBLE", "APPROVED_CANONICAL_SAMPLE"
        decisions.append({"sample_id": sample_id, "decision": state, "eligible": state != "EXCLUDED", "reason": reason})
    document = {
        "decision_version": DECISION_VERSION,
        "dataset": dataset,
        "inventory_sha256": inventory["inventory_sha256"].upper(),
        "parent_map_sha256": parent_hash,
        "source_evidence": evidence,
        "decisions": decisions,
    }
    document["decision_sha256"] = _hash(document)
    if sum(row["eligible"] for row in decisions) != EXPECTED_ELIGIBLE[dataset]:
        raise ValueError(f"{dataset} eligibility decision has the wrong population")
    return document


def _group_and_stratum(dataset: str, row: Mapping[str, Any]) -> tuple[str, str, str]:
    if dataset == "allidb1":
        canonical = "allidb1_000093" if row["sample_id"] == "allidb1_000108" else row["sample_id"]
        group = row.get("duplicate_group_id") or f"field-{canonical}"
        return group, group, f"{row['label']}|{row['resolution_family']}"
    parent = str(row["source_parent_id"])
    return parent, parent, f"{row['label']}|{bool(row['standard_crop'])}"


def _fold_counts(rows: Sequence[Mapping[str, Any]], ids: Sequence[str], stratum_field: str) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in rows}
    chosen = [by_id[sample_id] for sample_id in ids]
    return {
        "sample_count": len(chosen),
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in chosen).items())),
        "stratum_counts": dict(sorted(Counter(str(row[stratum_field]) for row in chosen).items())),
        "group_count": len({row["group_id"] for row in chosen}),
    }


def build_split_manifest(
    root: str | Path, dataset: str, decision: Mapping[str, Any] | None = None,
    implementation_commit: str | None = None,
) -> dict[str, Any]:
    if sklearn.__version__ != SKLEARN_VERSION:
        raise RuntimeError(f"Split freezing requires scikit-learn {SKLEARN_VERSION}, found {sklearn.__version__}")
    root, dataset = Path(root), dataset.lower().replace("-", "")
    inventory_path, _, _ = _paths(root, dataset)
    inventory_document = load_inventory(inventory_path)
    inventory_rows = inventory_document["records"]
    decision = dict(decision or build_eligibility_decision(root, dataset))
    if decision.get("decision_sha256") != _hash({k: v for k, v in decision.items() if k != "decision_sha256"}):
        raise ValueError("Eligibility decision SHA-256 mismatch")
    decisions = {row["sample_id"]: row for row in decision["decisions"]}

    parent_by_child: dict[str, dict[str, str]] = {}
    if dataset == "allidb2":
        parent_by_child = {row["allidb2_sample_id"]: row for row in _read_parent_map(root)}
    rows = []
    for source in inventory_rows:
        sample_id = source["sample_id"]
        merged = dict(source)
        if dataset == "allidb2":
            mapping = parent_by_child[sample_id]
            merged["source_parent_id"] = mapping["allidb1_parent_sample_id"]
        group_id, component_id, stratum = _group_and_stratum(dataset, merged)
        row = {
            "sample_id": sample_id,
            "dataset": dataset,
            "label": int(source["label"]),
            "eligible": bool(decisions[sample_id]["eligible"]),
            "eligibility_decision_version": decision["decision_version"],
            "eligibility_decision_hash": decision["decision_sha256"],
            "group_id": group_id,
            "source_component_id": component_id,
            "duplicate_group_id": source.get("duplicate_group_id"),
            "near_duplicate_group_id": source.get("near_duplicate_group_id"),
            "fold": None,
            "seed": SEED,
            "splitter_name": "StratifiedGroupKFold",
            "splitter_version": SKLEARN_VERSION,
            "inventory_sha256": inventory_document["inventory_sha256"].upper(),
            "preprocessing_protocol_hash": preprocessing_protocol_hash(dataset),
            "protocol_version": PROTOCOL_VERSION,
            "implementation_commit": implementation_commit,
            "stratification_key": stratum,
            "width": int(source["width"]),
            "height": int(source["height"]),
            "aspect_ratio": float(source["aspect_ratio"]),
        }
        if dataset == "allidb1":
            row.update(
                resolution_family=source["resolution_family"],
                canonical_sample_id="allidb1_000093" if sample_id == "allidb1_000108" else sample_id,
            )
        else:
            mapping = parent_by_child[sample_id]
            row.update(
                source_parent_id=mapping["allidb1_parent_sample_id"],
                parent_map_sha256=PARENT_MAP_SHA256,
                parent_mapping_protocol=mapping["mapping_method"],
                parent_mapping_status=mapping["mapping_status"],
                parent_ncc=float(mapping["full_resolution_ncc"]),
                parent_runner_up_gap=float(mapping["runner_up_gap"]),
                standard_crop=bool(source["standard_crop"]),
            )
        rows.append(row)

    eligible = [row for row in rows if row["eligible"]]
    labels = [row["stratification_key"] for row in eligible]
    groups = [row["group_id"] for row in eligible]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_folds = []
    for fold, (_, test_indices) in enumerate(splitter.split(eligible, labels, groups)):
        sample_ids = [eligible[index]["sample_id"] for index in test_indices]
        for index in test_indices:
            eligible[index]["fold"] = fold
        outer_folds.append({"fold": fold, "sample_ids": sample_ids})

    inner_splits = []
    by_id = {row["sample_id"]: row for row in eligible}
    all_ids = set(by_id)
    for outer_fold in range(5):
        outer_train = [by_id[sample_id] for sample_id in sorted(all_ids) if by_id[sample_id]["fold"] != outer_fold]
        inner_seed = 42007 + 10 * outer_fold
        inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=inner_seed)
        train_indices, val_indices = next(inner.split(
            outer_train,
            [row["stratification_key"] for row in outer_train],
            [row["group_id"] for row in outer_train],
        ))
        inner_splits.append({
            "outer_fold": outer_fold,
            "inner_seed": inner_seed,
            "train_sample_ids": [outer_train[index]["sample_id"] for index in train_indices],
            "validation_sample_ids": [outer_train[index]["sample_id"] for index in val_indices],
        })

    stratum_field = "resolution_family" if dataset == "allidb1" else "standard_crop"
    for fold in outer_folds:
        fold.update(_fold_counts(eligible, fold["sample_ids"], stratum_field))
        fold["group_ids"] = sorted({by_id[sample_id]["group_id"] for sample_id in fold["sample_ids"]})
    claim = (
        "field-disjoint cross-validation over 107 canonical ALL-IDB1 full fields; patient and specimen identities are unavailable."
        if dataset == "allidb1" else
        "parent-field-disjoint cross-validation over ALL-IDB2 crops mapped to canonical ALL-IDB1 source fields; patient and specimen identities are unavailable."
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "scientific_claim": claim,
        "sklearn_version": SKLEARN_VERSION,
        "seed": SEED,
        "splitter": {"name": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True, "random_state": SEED},
        "inner_splitter": {"name": "StratifiedGroupKFold", "n_splits": 2, "shuffle": True,
                           "seed_formula": "42007 + 10 * outer_fold", "selected_split": 0},
        "inventory_sha256": inventory_document["inventory_sha256"].upper(),
        "eligibility_decision_version": decision["decision_version"],
        "eligibility_decision_hash": decision["decision_sha256"],
        "parent_map_sha256": PARENT_MAP_SHA256 if dataset == "allidb2" else None,
        "preprocessing_protocol": preprocessing_protocol(dataset),
        "preprocessing_protocol_hash": preprocessing_protocol_hash(dataset),
        "expected_oof_rows": EXPECTED_ELIGIBLE[dataset],
        "outer_folds": outer_folds,
        "inner_splits": inner_splits,
        "singleton_resolution_assignment": (
            next(row["fold"] for row in eligible if row.get("resolution_family") == "1226x652")
            if dataset == "allidb1" else None
        ),
        "records": sorted(rows, key=lambda row: row["sample_id"]),
    }
    validate_split_manifest(manifest)
    manifest["manifest_sha256"] = _hash(manifest)
    return manifest


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    dataset = str(manifest["dataset"])
    if manifest.get("sklearn_version") != SKLEARN_VERSION:
        raise ValueError("Split manifest was not generated with locked scikit-learn")
    rows = list(manifest["records"])
    eligible = [row for row in rows if row["eligible"]]
    excluded = {row["sample_id"] for row in rows if not row["eligible"]}
    if len(eligible) != EXPECTED_ELIGIBLE[dataset] or excluded != set(EXCLUSIONS[dataset]):
        raise ValueError(f"{dataset} eligibility population mismatch")
    if {row["fold"] for row in eligible} != set(range(5)) or any(row["fold"] is not None for row in rows if not row["eligible"]):
        raise ValueError("Outer folds must be exactly 0..4 and exclude ineligible samples")
    if len({row["sample_id"] for row in eligible}) != len(eligible):
        raise ValueError("Eligible samples must appear exactly once")
    group_fold: dict[str, int] = {}
    for row in eligible:
        previous = group_fold.setdefault(row["group_id"], row["fold"])
        if previous != row["fold"]:
            raise ValueError("Group leakage across outer folds")
    for fold in range(5):
        if {row["label"] for row in eligible if row["fold"] == fold} != {0, 1}:
            raise ValueError(f"Outer fold {fold} does not contain both labels")
    by_id = {row["sample_id"]: row for row in eligible}
    if len(manifest["inner_splits"]) != 5:
        raise ValueError("Expected five inner splits")
    for split in manifest["inner_splits"]:
        train, validation = set(split["train_sample_ids"]), set(split["validation_sample_ids"])
        outer_test = {sample_id for sample_id, row in by_id.items() if row["fold"] == split["outer_fold"]}
        if train & validation or train | validation | outer_test != set(by_id) or (train | validation) & outer_test:
            raise ValueError("Inner/outer sample partitions are invalid")
        train_groups = {by_id[sample_id]["group_id"] for sample_id in train}
        validation_groups = {by_id[sample_id]["group_id"] for sample_id in validation}
        if train_groups & validation_groups:
            raise ValueError("Group leakage across inner train/validation")
        if {by_id[sample_id]["label"] for sample_id in train} != {0, 1}:
            raise ValueError("Inner train does not contain both labels")
        if {by_id[sample_id]["label"] for sample_id in validation} != {0, 1}:
            raise ValueError("Inner validation does not contain both labels")
    if dataset == "allidb2":
        if manifest.get("parent_map_sha256") != PARENT_MAP_SHA256:
            raise ValueError("ALL-IDB2 parent-map hash mismatch")
        if len(rows) != 260 or len({row["source_parent_id"] for row in rows}) != 102:
            raise ValueError("ALL-IDB2 parent-map coverage mismatch")
        if any(row["source_parent_id"] == "UNKNOWN" for row in rows):
            raise ValueError("ALL-IDB2 parent map contains an unknown parent")
        parent_labels: dict[str, int] = {}
        for row in rows:
            previous = parent_labels.setdefault(row["source_parent_id"], row["label"])
            if previous != row["label"]:
                raise ValueError("ALL-IDB2 parent labels are inconsistent")


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = _hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    if manifest.get("manifest_sha256") != expected:
        raise ValueError("ALL-IDB split manifest SHA-256 mismatch")
    validate_split_manifest(manifest)
    return manifest


def load_eligibility_decision(path: str | Path) -> dict[str, Any]:
    decision = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = _hash({key: value for key, value in decision.items() if key != "decision_sha256"})
    if decision.get("decision_sha256") != expected:
        raise ValueError("ALL-IDB eligibility decision SHA-256 mismatch")
    return decision


def freeze_protocols(root: str | Path, implementation_commit: str | None = None) -> list[Path]:
    root = Path(root)
    written = []
    for dataset in ("allidb1", "allidb2"):
        _, decision_path, split_path = _paths(root, dataset)
        decision = build_eligibility_decision(root, dataset)
        manifest = build_split_manifest(root, dataset, decision, implementation_commit)
        for path, document in ((decision_path, decision), (split_path, manifest)):
            path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            written.append(path)
    return written
