"""Fail-closed ALL-IDB inventory and provenance utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import opaque_sample_key

UNKNOWN = "UNKNOWN"
INVENTORY_PROTOCOL = "allidb_inventory_v1"
PARENT_MAPPING_PROTOCOL = "allidb_full_resolution_template_ncc_v1"
EXPECTED = {
    "allidb1": {"count": 108, "labels": {0: 59, 1: 49}, "extension": ".jpg"},
    "allidb2": {"count": 260, "labels": {0: 130, 1: 130}, "extension": ".tif"},
}
NEAR_DUPLICATE_NUMBERS = ((28, 31), (138, 146), (222, 256))


def parse_image_name(dataset: str, filename: str) -> tuple[int, int]:
    dataset = dataset.lower()
    if dataset not in EXPECTED:
        raise ValueError(f"Unsupported dataset: {dataset}")
    match = re.fullmatch(r"Im(\d{3})_([01])" + re.escape(EXPECTED[dataset]["extension"]), filename)
    if not match:
        raise ValueError(f"Invalid {dataset} image filename: {filename}")
    return int(match.group(1)), int(match.group(2))


def canonical_sample_id(dataset: str, image_number: int) -> str:
    dataset = dataset.lower()
    if dataset not in EXPECTED or image_number < 1:
        raise ValueError("Canonical sample identity requires a supported dataset and positive image number")
    return f"{dataset}_{image_number:06d}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"RGB\0{rgb.width}x{rgb.height}\0".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _image_stats(rgb: np.ndarray) -> dict[str, Any]:
    border = np.concatenate((rgb[0], rgb[-1], rgb[1:-1, 0], rgb[1:-1, -1]))
    means = rgb.reshape(-1, 3).mean(axis=0)
    return {
        "channel_min": [int(value) for value in rgb.reshape(-1, 3).min(axis=0)],
        "channel_max": [int(value) for value in rgb.reshape(-1, 3).max(axis=0)],
        "mean_rgb": [round(float(value), 6) for value in means],
        "brightness_mean": round(float(means.mean()), 6),
        "border_intensity_mean": round(float(border.mean()), 6),
    }


def _annotation_count(path: Path, width: int, height: int) -> int:
    text = path.read_text(encoding="ascii").strip()
    if not text:
        return 0
    count = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Invalid annotation at {path}:{line_number}")
        x, y = map(int, fields)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Out-of-bounds annotation at {path}:{line_number}")
        count += 1
    return count


def _raw_images(raw_root: Path, dataset: str) -> list[Path]:
    folder = raw_root / ("im" if dataset == "allidb1" else "img")
    files = sorted(path for path in folder.iterdir() if path.is_file())
    unexpected = [path.name for path in files if path.suffix.lower() != EXPECTED[dataset]["extension"]]
    if unexpected:
        raise ValueError(f"Unexpected files in {dataset} image directory: {unexpected}")
    return files


def scan_raw_dataset(raw_root: str | Path, dataset: str) -> list[dict[str, Any]]:
    """Decode and inventory a locked raw ALL-IDB directory without writing to it."""
    raw_root = Path(raw_root)
    dataset = dataset.lower()
    paths = _raw_images(raw_root, dataset)
    spec = EXPECTED[dataset]
    if len(paths) != spec["count"]:
        raise ValueError(f"{dataset} locked raw count changed: expected {spec['count']}, found {len(paths)}")

    records: list[dict[str, Any]] = []
    seen_numbers: set[int] = set()
    annotation_stems = {p.stem for p in (raw_root / "xyc").glob("*.xyc")} if dataset == "allidb1" else set()
    for path in paths:
        number, label = parse_image_name(dataset, path.name)
        if number in seen_numbers:
            raise ValueError(f"Duplicate numeric image ID in {dataset}: {number}")
        seen_numbers.add(number)
        with Image.open(path) as image:
            image.load()
            original_mode = image.mode
            original_bands = list(image.getbands())
            width, height = image.size
            rgb_image = image.convert("RGB")
            rgb = np.asarray(rgb_image)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"Unexpected decoded RGB representation: {path}")
            pixel_hash = decoded_pixel_sha256(rgb_image)
        annotation_count = None
        if dataset == "allidb1":
            annotation_path = raw_root / "xyc" / f"{path.stem}.xyc"
            if not annotation_path.is_file():
                raise ValueError(f"Missing annotation for {path.name}")
            annotation_count = _annotation_count(annotation_path, width, height)
            if (annotation_count > 0) != (label == 1):
                raise ValueError(f"Annotation/label disagreement for {path.name}")

        sample_id = canonical_sample_id(dataset, number)
        record = {
            "sample_id": sample_id,
            "opaque_sample_key": opaque_sample_key(dataset, sample_id),
            "dataset": dataset,
            "original_filename": path.name,
            "original_image_number": number,
            "relative_path": path.relative_to(raw_root.parents[2]).as_posix(),
            "label": label,
            "class_name": ("all_field" if label else "healthy_field") if dataset == "allidb1" else ("blast" if label else "non_blast"),
            "width": width,
            "height": height,
            "channels": 3,
            "aspect_ratio": round(width / height, 9),
            "original_mode": original_mode,
            "original_bands": original_bands,
            "bit_depth": 8,
            "file_extension": path.suffix.lower(),
            "file_size": path.stat().st_size,
            "file_sha256": file_sha256(path),
            "pixel_sha256": pixel_hash,
            "duplicate_group_id": None,
            "near_duplicate_group_id": None,
            "subject_id": UNKNOWN,
            "specimen_id": UNKNOWN,
            "source_parent_id": UNKNOWN,
            "acquisition_id": f"resolution:{width}x{height}",
            "camera_source": UNKNOWN,
            "resolution_family": f"{width}x{height}",
            "standard_crop": width == height == 257 if dataset == "allidb2" else None,
            "annotation_count": annotation_count,
            "split_eligible": True,
            "eligibility_state": "ELIGIBLE_CONTENT",
            "quarantine_reason": None,
            "provenance_status": "UNKNOWN_SUBJECT_PROVENANCE",
            "group_id": UNKNOWN,
            **_image_stats(rgb),
        }
        records.append(record)

    if Counter(row["label"] for row in records) != Counter(spec["labels"]):
        raise ValueError(f"{dataset} locked label counts changed")
    if seen_numbers != set(range(1, spec["count"] + 1)):
        raise ValueError(f"{dataset} locked numeric image IDs changed")
    if dataset == "allidb2" and any(row["label"] != int(row["original_image_number"] <= 130) for row in records):
        raise ValueError("ALL-IDB2 locked number/label rule changed")
    if dataset == "allidb1":
        image_stems = {path.stem for path in paths}
        if annotation_stems != image_stems:
            raise ValueError("ALL-IDB1 image/annotation stem set mismatch")
        if sum(row["annotation_count"] for row in records) != 510:
            raise ValueError("ALL-IDB1 locked annotation-coordinate count changed")
    return records


def assign_exact_duplicate_groups(records: list[dict[str, Any]]) -> list[list[str]]:
    """Merge file- or decoded-pixel-identical records and quarantine redundant members."""
    parent = {row["sample_id"]: row["sample_id"] for row in records}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for field in ("file_sha256", "pixel_sha256"):
        buckets: dict[str, list[str]] = defaultdict(list)
        for row in records:
            buckets[row[field]].append(row["sample_id"])
        for members in buckets.values():
            for member in members[1:]:
                union(members[0], member)

    by_id = {row["sample_id"]: row for row in records}
    components: dict[str, list[str]] = defaultdict(list)
    for sample_id in parent:
        components[find(sample_id)].append(sample_id)
    duplicate_groups = [sorted(members, key=lambda sid: by_id[sid]["original_image_number"])
                        for members in components.values() if len(members) > 1]
    for members in sorted(duplicate_groups):
        canonical = members[0]
        group_id = f"exact-{canonical}"
        for sample_id in members:
            by_id[sample_id]["duplicate_group_id"] = group_id
        for sample_id in members[1:]:
            by_id[sample_id].update(
                split_eligible=False,
                eligibility_state="REDUNDANT_EXACT_DUPLICATE",
                quarantine_reason="REDUNDANT_EXACT_DUPLICATE",
            )
    return sorted(duplicate_groups)


def connected_components(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Deterministically merge only explicit, known identity/provenance edges."""
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        if node not in parent:
            raise ValueError(f"Edge references unknown node: {node}")
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[str, list[str]] = defaultdict(list)
    for node in sorted(parent):
        groups[find(node)].append(node)
    return sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])


def near_duplicate_registry(records: list[dict[str, Any]], raw_root: str | Path) -> list[dict[str, Any]]:
    if not records or records[0]["dataset"] != "allidb2":
        return []
    raw_root = Path(raw_root)
    by_number = {row["original_image_number"]: row for row in records}
    rows = []
    for left_number, right_number in NEAR_DUPLICATE_NUMBERS:
        left, right = by_number[left_number], by_number[right_number]
        with Image.open(raw_root / "img" / left["original_filename"]) as a, Image.open(raw_root / "img" / right["original_filename"]) as b:
            x = np.asarray(a.convert("RGB"), dtype=np.float64)
            y = np.asarray(b.convert("RGB"), dtype=np.float64)
        mae = float(np.mean(np.abs(x - y)))
        correlation = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
        group_id = f"near-{left['sample_id']}-{right['sample_id']}"
        for record in (left, right):
            record.update(
                near_duplicate_group_id=group_id,
                split_eligible=False,
                eligibility_state="UNRESOLVED_NEAR_DUPLICATE",
                quarantine_reason="UNRESOLVED_NEAR_DUPLICATE",
            )
        rows.append({
            "sample_a": left["sample_id"], "sample_b": right["sample_id"],
            "relationship_type": "SIMILARITY_CANDIDATE", "evidence_type": "ALIGNED_PIXEL_SIMILARITY",
            "pixel_mae": round(mae, 9), "pixel_correlation": round(correlation, 12),
            "review_status": "UNRESOLVED", "group_action": "QUARANTINE_UNTIL_REVIEW",
        })
    return rows


def reproduce_parent_map(
    allidb1_root: str | Path, allidb2_root: str | Path, *, shortlist_size: int = 3,
    coarse_scale: float = 0.125, score_threshold: float = 0.999, gap_threshold: float = 0.003,
) -> list[dict[str, Any]]:
    """Map crops to fields using a coarse shortlist and full-resolution NCC confirmation."""
    import cv2

    parent_paths = sorted(Path(allidb1_root).joinpath("im").glob("*.jpg"))
    crop_paths = sorted(Path(allidb2_root).joinpath("img").glob("*.tif"))
    coarse_parents = [(path, cv2.resize(cv2.imread(str(path), 0), None, fx=coarse_scale, fy=coarse_scale))
                      for path in parent_paths]
    full_parents: dict[Path, np.ndarray] = {}
    evidence: list[dict[str, Any]] = []
    inverse = round(1 / coarse_scale)
    for crop_path in crop_paths:
        crop = cv2.imread(str(crop_path), 0)
        if crop is None:
            raise ValueError(f"OpenCV could not decode {crop_path}")
        coarse_crop = cv2.resize(crop, None, fx=coarse_scale, fy=coarse_scale)
        candidates = []
        for parent_path, parent in coarse_parents:
            _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(parent, coarse_crop, cv2.TM_CCOEFF_NORMED))
            candidates.append((score, parent_path, location))
        candidates.sort(reverse=True, key=lambda row: row[0])

        fine_by_parent: dict[str, tuple[float, str, tuple[int, int], float]] = {}
        # ponytail: top-3 coarse shortlist; expand only if a future raw revision fails the full-score thresholds.
        for coarse_score, parent_path, (x, y) in candidates[:shortlist_size]:
            if parent_path not in full_parents:
                full_parents[parent_path] = cv2.imread(str(parent_path), 0)
            parent = full_parents[parent_path]
            x, y = x * inverse, y * inverse
            x0, y0 = max(0, x - 12), max(0, y - 12)
            x1 = min(parent.shape[1], x + crop.shape[1] + 12)
            y1 = min(parent.shape[0], y + crop.shape[0] + 12)
            _, fine_score, _, local = cv2.minMaxLoc(
                cv2.matchTemplate(parent[y0:y1, x0:x1], crop, cv2.TM_CCOEFF_NORMED)
            )
            number, _ = parse_image_name("allidb1", parent_path.name)
            canonical_number = 93 if number == 108 else number
            canonical_parent = canonical_sample_id("allidb1", canonical_number)
            candidate = (float(fine_score), canonical_parent, (x0 + local[0], y0 + local[1]), float(coarse_score))
            if canonical_parent not in fine_by_parent or candidate > fine_by_parent[canonical_parent]:
                fine_by_parent[canonical_parent] = candidate

        fine = sorted(fine_by_parent.values(), reverse=True)
        best = fine[0]
        gap = best[0] - fine[1][0] if len(fine) > 1 else 1.0
        crop_number, _ = parse_image_name("allidb2", crop_path.name)
        mapped = best[0] >= score_threshold and gap >= gap_threshold
        evidence.append({
            "allidb2_sample_id": canonical_sample_id("allidb2", crop_number),
            "allidb1_parent_sample_id": best[1] if mapped else UNKNOWN,
            "mapping_status": "REPRODUCED_HIGH_CONFIDENCE" if mapped else "UNRESOLVED",
            "mapping_method": PARENT_MAPPING_PROTOCOL,
            "confidence": "HIGH" if mapped else "INSUFFICIENT",
            "full_resolution_ncc": round(best[0], 12),
            "runner_up_gap": round(gap, 12),
            "match_x": best[2][0] if mapped else None,
            "match_y": best[2][1] if mapped else None,
            "manual_review_status": "NOT_REVIEWED",
        })
    return evidence


FINGERPRINT_FIELDS = (
    "relative_path", "sample_id", "label", "file_sha256", "pixel_sha256", "split_eligible",
    "eligibility_state", "duplicate_group_id", "near_duplicate_group_id", "subject_id", "specimen_id",
    "source_parent_id", "provenance_status", "group_id",
)


def inventory_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    stable = [{field: row.get(field) for field in FINGERPRINT_FIELDS}
              for row in sorted(records, key=lambda row: row["sample_id"])]
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256((INVENTORY_PROTOCOL + "\0" + payload).encode()).hexdigest()


def apply_parent_map(records: list[dict[str, Any]], mapping: Sequence[Mapping[str, Any]]) -> None:
    by_id = {row["sample_id"]: row for row in records}
    if set(by_id) != {row["allidb2_sample_id"] for row in mapping}:
        raise ValueError("ALL-IDB2 parent map does not cover the inventory exactly")
    for item in mapping:
        record = by_id[item["allidb2_sample_id"]]
        parent = item["allidb1_parent_sample_id"]
        record["source_parent_id"] = parent
        if parent == UNKNOWN:
            record.update(
                split_eligible=False, eligibility_state="UNRESOLVED_PROVENANCE",
                quarantine_reason="UNRESOLVED_PROVENANCE", provenance_status="UNRESOLVED_PARENT_PROVENANCE",
                group_id=UNKNOWN,
            )
        else:
            record["provenance_status"] = "REPRODUCED_PARENT_MAPPING_SUBJECT_UNKNOWN"
            record["group_id"] = f"source-{parent}"


def build_inventory_document(dataset: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "protocol": INVENTORY_PROTOCOL,
        "dataset": dataset,
        "inventory_sha256": inventory_sha256(records),
        "status": "BLOCKED_PROVENANCE",
        "records": sorted(records, key=lambda row: row["sample_id"]),
    }


def load_inventory(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("protocol") != INVENTORY_PROTOCOL:
        raise ValueError("Unsupported ALL-IDB inventory protocol")
    if document.get("inventory_sha256") != inventory_sha256(document["records"]):
        raise ValueError("ALL-IDB inventory fingerprint mismatch")
    return document


class AllIdbDataset(Dataset):
    """PyTorch loader for approved ALL-IDB field/crop records."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        transform=None,
        augment_d4: bool = False,
        seed: int = 42,
        fold: int = 0,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.augment_d4 = augment_d4
        self.seed = seed
        self.fold = fold
        self.epoch = 0
        if augment_d4 and any(row.get("split_role") != "inner_train" for row in self.samples):
            raise ValueError("D4 augmentation is permitted only for inner-training samples")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        try:
            with Image.open(row["image_path"]) as image:
                image.load()
                rgb = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image at {row['image_path']}: {exc}") from exc
        key = str(row["opaque_sample_key"])
        if self.augment_d4:
            from .augmentation import D4Transform
            rgb = D4Transform(key, self.epoch, self.seed, self.fold, row["dataset"])(rgb)
        tensor = self.transform(rgb) if self.transform else torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float() / 255
        result = {
            "image": tensor,
            "label": int(row["label"]),
            "sample_id": row["sample_id"],
            "group_id": row["group_id"],
            "dataset": row["dataset"],
            "opaque_key": key,
            "width": int(row["width"]),
            "height": int(row["height"]),
            "aspect_ratio": float(row["aspect_ratio"]),
        }
        size = getattr(self.transform, "input_size", max(int(row["width"]), int(row["height"])))
        scale = min(size / int(row["width"]), size / int(row["height"]))
        resized_width, resized_height = round(int(row["width"]) * scale), round(int(row["height"]) * scale)
        result["padding_fraction"] = 1.0 - (resized_width * resized_height) / (size * size)
        if row["dataset"] == "allidb1":
            result["resolution_family"] = row["resolution_family"]
        if row["dataset"] == "allidb2":
            result["source_parent_id"] = row["source_parent_id"]
        return result


def order_samples_by_opaque_key(samples: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(samples, key=lambda row: (str(row["opaque_sample_key"]), str(row["sample_id"])))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
