"""Subject-aware and group-disjoint cross-validation splitting for C-NMC."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cnmc import get_default_data_dir


def load_splits(splits_path: str | Path | None = None) -> dict[str, Any]:
    """Load the verified C-NMC 5-fold subject-disjoint split manifest."""
    if splits_path is None:
        splits_path = get_default_data_dir() / "cnmc_splits.json"
    splits_path = Path(splits_path)

    if not splits_path.is_file():
        raise FileNotFoundError(f"C-NMC splits manifest not found at: {splits_path}")

    with splits_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate structure
    outer_folds = data.get("outer_folds", [])
    if len(outer_folds) != 5:
        raise ValueError(f"Expected 5 outer folds in splits, found {len(outer_folds)}")
    if {fold.get("fold") for fold in outer_folds} != set(range(5)):
        raise ValueError("Outer fold IDs must be exactly 0,1,2,3,4")

    seen_samples: set[str] = set()
    seen_groups: set[str] = set()
    seen_subjects: set[str] = set()
    for fold in outer_folds:
        samples = set(fold["sample_ids"])
        groups = set(fold["group_ids"])
        subjects = set(fold["subject_ids"])
        if len(samples) != fold["image_count"] or seen_samples & samples:
            raise ValueError(f"Fold {fold['fold']} has an invalid count or sample overlap")
        if seen_groups & groups or seen_subjects & subjects:
            raise ValueError(f"Fold {fold['fold']} has group or subject overlap")
        seen_samples.update(samples)
        seen_groups.update(groups)
        seen_subjects.update(subjects)

    total_images = sum(f["image_count"] for f in outer_folds)
    if total_images != 10655:
        raise ValueError(f"Expected 10,655 active dev images across folds, found {total_images}")

    return data


def get_outer_fold_data(
    splits_data: Mapping[str, Any],
    fold: int,
) -> tuple[list[str], list[str]]:
    """Return (train_sample_ids, val_sample_ids) for a given outer fold.

    Outer fold validation set is strictly subject/group-disjoint from outer train.
    """
    if fold < 0 or fold >= 5:
        raise ValueError(f"Outer fold must be in range [0, 4], got {fold}")

    if "records" in splits_data:
        eligible = [row for row in splits_data["records"] if row["eligible"]]
        return (
            [row["sample_id"] for row in eligible if row["fold"] != fold],
            [row["sample_id"] for row in eligible if row["fold"] == fold],
        )

    outer_folds = splits_data["outer_folds"]
    val_fold = next(item for item in outer_folds if item["fold"] == fold)
    val_sids = list(val_fold["sample_ids"])
    val_sid_set = set(val_sids)

    # Outer train is all dev samples not in this fold
    train_sids = []
    for f in outer_folds:
        if f["fold"] != fold:
            for sid in f["sample_ids"]:
                if sid not in val_sid_set:
                    train_sids.append(sid)

    return train_sids, val_sids


def validate_development_inventory(
    inventory: Mapping[str, Mapping[str, Any]], splits_data: Mapping[str, Any]
) -> None:
    """Fail closed unless the split manifest is the exact active development inventory."""
    active = {
        sid for sid, row in inventory.items()
        if row.get("phase") == "training" and row.get("modeling_eligible", True)
    }
    folded = {sid for fold in splits_data["outer_folds"] for sid in fold["sample_ids"]}
    if len(active) != 10655 or active != folded:
        raise ValueError(f"Active development inventory/split mismatch: active={len(active)}, folded={len(folded)}")
    if {(inventory[sid].get("class_name"), inventory[sid].get("label")) for sid in active} != {
        ("HEM", 0), ("ALL", 1)
    }:
        raise ValueError("Locked class mapping must be HEM=0 and ALL=1")
    for fold in splits_data["outer_folds"]:
        samples = fold["sample_ids"]
        if {inventory[sid].get("final_group_id") for sid in samples} != set(fold["group_ids"]):
            raise ValueError(f"Fold {fold['fold']} final_group membership disagrees with inventory")
        if {inventory[sid].get("subject_id") for sid in samples} != set(fold["subject_ids"]):
            raise ValueError(f"Fold {fold['fold']} subject membership disagrees with inventory")
        counts = {str(label): sum(int(inventory[sid]["label"]) == label for sid in samples) for label in (0, 1)}
        if counts != fold["counts_by_label"]:
            raise ValueError(f"Fold {fold['fold']} label counts disagree with inventory")


def get_inner_split(
    inventory: Mapping[str, Mapping[str, Any]],
    outer_train_sids: Sequence[str],
    seed: int,
    fold: int,
    val_fraction: float = 0.20,
    splits_data: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """Deterministic, group-preserving inner split of the outer training set.

    Uses inner_seed = seed * 1000 + outer_fold * 10 + 7.
    Groups are assigned to inner validation or inner train atomically.
    """
    if splits_data is not None and "inner_splits" in splits_data:
        if seed != 42:
            raise ValueError("ALL-IDB split manifests are frozen for seed 42")
        selected = next(item for item in splits_data["inner_splits"] if item["outer_fold"] == fold)
        train, validation = list(selected["train_sample_ids"]), list(selected["validation_sample_ids"])
        if set(train) | set(validation) != set(outer_train_sids) or set(train) & set(validation):
            raise ValueError("Frozen inner split does not match the requested outer train population")
        return train, validation
    inner_seed = seed * 1000 + fold * 10 + 7

    # Group sample IDs by final_group_id
    groups: dict[str, list[str]] = {}
    for sid in outer_train_sids:
        rec = inventory[sid]
        gid = rec.get("final_group_id") or rec.get("subject_id") or sid
        groups.setdefault(gid, []).append(sid)

    # Sort groups deterministically by hash(inner_seed, group_id)
    def group_sort_key(gid: str) -> tuple[int, str]:
        digest = hashlib.sha256(f"inner_split:{inner_seed}:{gid}".encode("utf-8")).hexdigest()
        return (int(digest[:16], 16), gid)

    sorted_gids = sorted(groups.keys(), key=group_sort_key)

    total_samples = len(outer_train_sids)
    target_val_count = int(total_samples * val_fraction)

    inner_val_sids: list[str] = []
    inner_train_sids: list[str] = []
    current_val_count = 0

    for gid in sorted_gids:
        sids = groups[gid]
        if current_val_count + len(sids) <= target_val_count or current_val_count == 0:
            inner_val_sids.extend(sids)
            current_val_count += len(sids)
        else:
            inner_train_sids.extend(sids)

    # Fallback safety: ensure both sets non-empty
    if not inner_val_sids and sorted_gids:
        first_gid = sorted_gids[0]
        inner_val_sids = groups[first_gid]
        inner_train_sids = [s for s in outer_train_sids if s not in set(inner_val_sids)]

    return inner_train_sids, inner_val_sids
