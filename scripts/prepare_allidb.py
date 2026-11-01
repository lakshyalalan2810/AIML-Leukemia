"""Generate canonical ALL-IDB integrity artifacts without changing raw data."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.allidb import (
    PARENT_MAPPING_PROTOCOL,
    apply_parent_map,
    assign_exact_duplicate_groups,
    build_inventory_document,
    file_sha256,
    near_duplicate_registry,
    reproduce_parent_map,
    scan_raw_dataset,
    write_csv,
    write_json,
)


def raw_snapshot(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def summarize(records: list[dict], duplicate_groups: list[list[str]]) -> dict:
    by_resolution: dict[str, Counter] = defaultdict(Counter)
    for row in records:
        by_resolution[row["resolution_family"]][row["label"]] += 1
    by_label = {
        str(label): {
            "mean_rgb": [round(fmean(row["mean_rgb"][channel] for row in records if row["label"] == label), 6)
                         for channel in range(3)],
            "brightness_mean": round(fmean(row["brightness_mean"] for row in records if row["label"] == label), 6),
            "border_intensity_mean": round(fmean(row["border_intensity_mean"] for row in records if row["label"] == label), 6),
            "aspect_ratio_mean": round(fmean(row["aspect_ratio"] for row in records if row["label"] == label), 9),
        }
        for label in (0, 1)
    }
    return {
        "raw_count": len(records),
        "label_counts": dict(sorted(Counter(row["label"] for row in records).items())),
        "decoded_modes": dict(sorted(Counter(row["original_mode"] for row in records).items())),
        "bit_depths": dict(sorted(Counter(row["bit_depth"] for row in records).items())),
        "resolution_by_label": {key: dict(sorted(value.items())) for key, value in sorted(by_resolution.items())},
        "descriptive_by_label": by_label,
        "standard_crop_count": sum(row["standard_crop"] is True for row in records),
        "irregular_crop_count": sum(row["standard_crop"] is False for row in records),
        "exact_duplicate_groups": duplicate_groups,
        "eligible_count": sum(row["split_eligible"] for row in records),
        "unresolved_provenance_count": sum(row["source_parent_id"] == "UNKNOWN" for row in records),
        "unresolved_near_duplicate_count": sum(row["near_duplicate_group_id"] is not None for row in records),
        "camera_attribution": "UNKNOWN; source README contains no camera-to-image mapping",
        "scientific_status": "BLOCKED_PROVENANCE",
    }


def write_dataset_artifacts(
    dataset: str, records: list[dict], duplicate_groups: list[list[str]], near_rows: list[dict]
) -> None:
    out = repo_root / "data" / dataset
    document = build_inventory_document(dataset, records)
    write_json(out / f"{dataset}_inventory.json", document)
    fields = list(records[0])
    write_csv(out / f"{dataset}_inventory.csv", records, fields)
    duplicate_rows = [
        {"duplicate_group_id": f"exact-{members[0]}", "canonical_sample_id": members[0], "member_sample_id": member}
        for members in duplicate_groups for member in members
    ]
    write_csv(out / f"{dataset}_exact_duplicates.csv", duplicate_rows,
              ("duplicate_group_id", "canonical_sample_id", "member_sample_id"))
    near_fields = ("sample_a", "sample_b", "relationship_type", "evidence_type", "pixel_mae",
                   "pixel_correlation", "review_status", "group_action")
    write_csv(out / f"{dataset}_near_duplicates.csv", near_rows, near_fields)
    write_json(out / f"{dataset}_audit.json", summarize(records, duplicate_groups))
    write_json(out / f"{dataset}_status.json", {
        "dataset": dataset, "integrity": "PASS", "provenance": "SUBJECT_AND_SPECIMEN_UNKNOWN",
        "preprocessing": "READY", "split_readiness": "BLOCKED_PROVENANCE",
        "training_readiness": "NOT_READY_SPLITS_NOT_APPROVED",
        "inventory_sha256": document["inventory_sha256"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-parent-mapping", action="store_true", help="Leave all crop parents unresolved")
    args = parser.parse_args()
    raw = repo_root / "data" / "raw"
    before = raw_snapshot(raw)

    idb1 = scan_raw_dataset(raw / "ALL_IDB1", "allidb1")
    idb2 = scan_raw_dataset(raw / "ALL_IDB2", "allidb2")
    duplicates1 = assign_exact_duplicate_groups(idb1)
    duplicates2 = assign_exact_duplicate_groups(idb2)
    if duplicates1 != [["allidb1_000093", "allidb1_000108"]]:
        raise ValueError(f"Unexpected ALL-IDB1 exact duplicates: {duplicates1}")
    if duplicates2 != [["allidb2_000223", "allidb2_000257"]]:
        raise ValueError(f"Unexpected ALL-IDB2 exact duplicates: {duplicates2}")

    mapping = ([{
        "allidb2_sample_id": row["sample_id"], "allidb1_parent_sample_id": "UNKNOWN",
        "mapping_status": "UNRESOLVED", "mapping_method": PARENT_MAPPING_PROTOCOL,
        "confidence": "INSUFFICIENT", "full_resolution_ncc": None, "runner_up_gap": None,
        "match_x": None, "match_y": None, "manual_review_status": "NOT_REVIEWED",
    } for row in idb2] if args.skip_parent_mapping else
        reproduce_parent_map(raw / "ALL_IDB1", raw / "ALL_IDB2"))
    apply_parent_map(idb2, mapping)
    near1 = near_duplicate_registry(idb1, raw / "ALL_IDB1")
    near2 = near_duplicate_registry(idb2, raw / "ALL_IDB2")

    write_dataset_artifacts("allidb1", idb1, duplicates1, near1)
    write_dataset_artifacts("allidb2", idb2, duplicates2, near2)
    write_csv(repo_root / "data" / "allidb2" / "allidb2_parent_map.csv", mapping, tuple(mapping[0]))
    write_json(repo_root / "data" / "allidb2" / "allidb2_parent_map_summary.json", {
        "protocol": PARENT_MAPPING_PROTOCOL,
        "mapped_crops": sum(row["mapping_status"] == "REPRODUCED_HIGH_CONFIDENCE" for row in mapping),
        "unique_canonical_parents": len({row["allidb1_parent_sample_id"] for row in mapping
                                         if row["allidb1_parent_sample_id"] != "UNKNOWN"}),
        "unresolved_crops": sum(row["mapping_status"] == "UNRESOLVED" for row in mapping),
        "previous_read_only_result": {"mapped_crops": 149, "unique_parents": 78},
        "comparison": "Recomputed from raw pixels; previous count was not trusted as input.",
    })
    write_json(repo_root / "data" / "allidb_preprocessing.json", {
        "protocol": "shared_imagenet_preprocessing_v1",
        "decode": "RGB_8BIT",
        "resize": "BICUBIC_ASPECT_PRESERVING_FIT_WITH_IMAGENET_MEAN_PADDING",
        "dataset_policy": {"allidb1": "fit_pad", "allidb2": "fit_pad", "cnmc": "direct_unchanged"},
        "model_input_sizes": {"convnextv2_tiny": 224, "efficientnet_b1": 240},
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "learned_dataset_statistics": None,
        "clahe": "OPTIONAL_OFF",
    })
    write_json(repo_root / "data" / "dataset_status.json", {
        "cnmc": {"status": "READY_FOR_TRAINING", "note": "Existing locked 5-fold protocol unchanged"},
        "allidb1": {"status": "BLOCKED_PROVENANCE", "note": "Subject/specimen identity unavailable; no splits approved"},
        "allidb2": {"status": "BLOCKED_PROVENANCE", "note": "All parent fields reproduced; subject/specimen identity unavailable; near-duplicate review open; no splits approved"},
    })
    if before != raw_snapshot(raw):
        raise RuntimeError("Raw ALL-IDB data changed during inventory generation")
    print("ALL-IDB inventories generated; raw snapshot unchanged")


if __name__ == "__main__":
    main()
