"""C-NMC 2019 dataset inventory, duplicate quarantine, and PyTorch Dataset."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import D4Transform, compute_record_opaque_key

EXPECTED_INVENTORY_TOTAL = 15114
EXPECTED_RAW_TRAIN = 10661
EXPECTED_RAW_PRELIM = 1867
EXPECTED_FINAL_UNLABELED = 2586
EXPECTED_ACTIVE_DEV = 10655
EXPECTED_LOCKED_PRELIM = 1861
EXPECTED_QUARANTINED = 12  # 6 pairs = 6 dev + 6 prelim


def load_cross_phase_duplicate_ids(csv_path: str | Path) -> set[str]:
    """Reconstruct the locked 12-endpoint quarantine from the audited CSV."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"C-NMC duplicate audit not found at: {csv_path}")

    required = {"sample_a", "sample_b", "phase_a", "phase_b", "crosses_challenge_phase"}
    pairs: list[tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Duplicate CSV schema mismatch; required fields: {sorted(required)}")
        for row in reader:
            if row["crosses_challenge_phase"].strip().lower() == "true":
                if {row["phase_a"], row["phase_b"]} != {"training", "preliminary_validation"}:
                    raise ValueError("Cross-phase duplicate does not join training and preliminary validation")
                pairs.append((row["sample_a"], row["sample_b"]))

    endpoints = {sid for pair in pairs for sid in pair}
    if len(pairs) != 6 or len(endpoints) != EXPECTED_QUARANTINED:
        raise ValueError(f"Expected 6 cross-phase pairs/12 endpoints, found {len(pairs)}/{len(endpoints)}")
    return endpoints


def get_default_data_dir() -> Path:
    """Return default path to data/cnmc."""
    # Look relative to repository root
    cwd = Path.cwd()
    if (cwd / "data" / "cnmc").is_dir():
        return cwd / "data" / "cnmc"
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "cnmc"


def load_cnmc_inventory(
    data_dir: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load and index C-NMC dataset inventory with quarantine status.

    Returns dict mapping sample_id to record.
    Guarantees cross-phase pixel duplicates remain quarantined.
    """
    if data_dir is None:
        data_dir = get_default_data_dir()
    data_dir = Path(data_dir)

    inventory_json = data_dir / "cnmc_inventory.json"
    if not inventory_json.is_file():
        raise FileNotFoundError(f"C-NMC inventory not found at: {inventory_json}")

    with inventory_json.open("r", encoding="utf-8") as f:
        records_list = json.load(f)

    quarantined_ids = load_cross_phase_duplicate_ids(data_dir / "cnmc_exact_duplicate_pairs.csv")
    inventory_quarantine = {r["sample_id"] for r in records_list if not r.get("modeling_eligible", True)}
    if quarantined_ids != inventory_quarantine:
        missing = sorted(quarantined_ids - inventory_quarantine)
        extra = sorted(inventory_quarantine - quarantined_ids)
        raise ValueError(f"Duplicate CSV/inventory quarantine mismatch; missing={missing}, extra={extra}")

    inventory: dict[str, dict[str, Any]] = {}
    for row in records_list:
        sid = row["sample_id"]
        is_quarantined = sid in quarantined_ids
        row["modeling_eligible"] = not is_quarantined
        if is_quarantined:
            row["quarantine_reason"] = "cross_phase_duplicate_quarantine"

        # Update root if external dataset moved
        if dataset_root is not None:
            rel_path = row.get("relative_path")
            if rel_path:
                row["image_path"] = str(Path(dataset_root) / rel_path)

        inventory[sid] = row

    return inventory


class CnmcDataset(Dataset):
    """PyTorch Dataset for C-NMC single-cell microscopy images."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        target_transform: Callable[[int], int] | None = None,
        augment_d4: bool = False,
        seed: int = 42,
        fold: int = 0,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.target_transform = target_transform
        self.augment_d4 = augment_d4
        self.seed = seed
        self.fold = fold
        self.epoch = 0

        if augment_d4 and any(row.get("phase") != "training" for row in self.samples):
            raise ValueError("D4 augmentation is permitted only for inner-training samples")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples[idx]
        image_path = row["image_path"]

        try:
            with Image.open(image_path) as img:
                img.load()
                image = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image at {image_path}: {exc}") from exc

        if self.augment_d4:
            opaque_key = compute_record_opaque_key(row)
            image = D4Transform(opaque_key, self.epoch, self.seed, self.fold)(image)

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        label = row.get("label")
        if label is not None:
            label = int(label)
            if self.target_transform is not None:
                label = self.target_transform(label)

        return {
            "image": image_tensor,
            "label": label if label is not None else -1,
            "sample_id": row["sample_id"],
            "group_id": row.get("final_group_id") or row.get("subject_id", ""),
            "dataset": "cnmc",
            "subject_id": row.get("subject_id", ""),
            "phase": row.get("phase", ""),
            "opaque_key": compute_record_opaque_key(row),
        }


def order_samples_by_opaque_key(samples: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return a deterministic sample order that never consults labels or class names."""
    return sorted(samples, key=lambda row: (compute_record_opaque_key(row), str(row["sample_id"])))
