"""Deterministic label-independent D4 augmentation scheduled via opaque sample keys."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

OPAQUE_KEY_POLICY = "cnmc_runtime_opaque_sample_key_sha256_v3"
IDENTITY_KEY_POLICY = "dataset_sample_identity_sha256_v1"


def opaque_sample_key(dataset: str, sample_id: str) -> str:
    """Hash a canonical label-independent identity for augmentation scheduling."""
    if not dataset or not sample_id:
        raise ValueError("Opaque keys require dataset and sample_id")
    payload = f"{IDENTITY_KEY_POLICY}\0dataset={dataset.lower()}\0sample_id={sample_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_opaque_key(phase: str, original_filename: str | None, current_filename: str) -> str:
    """Compute label-independent opaque sample key that excludes target suffixes.

    Ensures deterministic augmentation scheduling cannot leak target class.
    """
    filename = str(original_filename or current_filename) if phase == "training" else str(current_filename)
    stem = Path(filename).stem
    if phase == "training":
        stem = re.sub(r"_(?:ALL|HEM)$", "", stem, flags=re.IGNORECASE)
    payload = f"{OPAQUE_KEY_POLICY}\0phase={phase}\0stem={stem}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_record_opaque_key(record: Mapping[str, Any]) -> str:
    """Derive the auditable runtime key without reading labels, class names, or paths."""
    return compute_opaque_key(
        str(record.get("phase", "")), record.get("original_filename"), str(record["current_filename"])
    )


def apply_d4_transform(img: Image.Image, transform_idx: int) -> Image.Image:
    """Apply one of the 8 deterministic dihedral D4 transformations."""
    idx = transform_idx % 8
    if idx == 0:
        return img
    elif idx == 1:
        return img.transpose(Image.ROTATE_90)
    elif idx == 2:
        return img.transpose(Image.ROTATE_180)
    elif idx == 3:
        return img.transpose(Image.ROTATE_270)
    elif idx == 4:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif idx == 5:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif idx == 6:
        # Transpose: flip left-right + rotate 90
        return img.transpose(Image.TRANSPOSE)
    elif idx == 7:
        # Anti-transpose
        return img.transpose(Image.TRANSVERSE)
    return img


class D4Transform:
    """Callable that deterministically applies D4 transformation to PIL Image."""

    def __init__(
        self,
        opaque_key: str,
        epoch: int,
        seed: int = 42,
        fold: int = 0,
        dataset: str = "C-NMC",
    ) -> None:
        self.opaque_key = opaque_key
        self.epoch = epoch
        self.seed = seed
        self.fold = fold
        self.dataset = dataset

    def __call__(self, img: Image.Image) -> Image.Image:
        h = hashlib.sha256(
            f"d4:{self.dataset}:{self.seed}:{self.fold}:{self.epoch}:{self.opaque_key}".encode("utf-8")
        ).hexdigest()
        t_idx = int(h[:8], 16) % 8
        return apply_d4_transform(img, t_idx)
