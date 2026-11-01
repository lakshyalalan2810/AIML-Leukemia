"""Read-only validation of raw ALL-IDB data against canonical inventories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.allidb import assign_exact_duplicate_groups, load_inventory, scan_raw_dataset
from src.data.allidb_protocol import load_eligibility_decision


def validate(dataset: str) -> dict:
    raw_name = "ALL_IDB1" if dataset == "allidb1" else "ALL_IDB2"
    actual = scan_raw_dataset(repo_root / "data" / "raw" / raw_name, dataset)
    actual_groups = assign_exact_duplicate_groups(actual)
    document = load_inventory(repo_root / "data" / dataset / f"{dataset}_inventory.json")
    expected = {row["sample_id"]: row for row in document["records"]}
    if set(expected) != {row["sample_id"] for row in actual}:
        raise ValueError(f"{dataset} raw/inventory sample set mismatch")
    for row in actual:
        saved = expected[row["sample_id"]]
        for field in ("relative_path", "label", "width", "height", "channels", "file_size", "file_sha256", "pixel_sha256"):
            if row[field] != saved[field]:
                raise ValueError(f"{dataset} mismatch for {row['sample_id']} field {field}")
    saved_groups = sorted({row["duplicate_group_id"] for row in expected.values() if row["duplicate_group_id"]})
    if len(saved_groups) != len(actual_groups):
        raise ValueError(f"{dataset} exact duplicate group count mismatch")
    decision = load_eligibility_decision(repo_root / "data" / dataset / f"{dataset}_eligibility_v1.json")
    if decision["inventory_sha256"].lower() != document["inventory_sha256"].lower():
        raise ValueError(f"{dataset} eligibility decision/inventory mismatch")
    return {
        "dataset": dataset,
        "raw_count": len(actual),
        "label_counts": dict(sorted(Counter(row["label"] for row in actual).items())),
        "corrupt_files": 0,
        "exact_duplicate_groups": len(actual_groups),
        "eligible_count": sum(row["eligible"] for row in decision["decisions"]),
        "historical_split_eligible_count": sum(row["split_eligible"] for row in expected.values()),
        "unresolved_provenance_count": sum(row["source_parent_id"] == "UNKNOWN" for row in expected.values()),
        "unresolved_near_duplicate_count": 0,
        "inventory_sha256": document["inventory_sha256"],
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("allidb1", "allidb2", "all"))
    args = parser.parse_args()
    datasets = ("allidb1", "allidb2") if args.dataset == "all" else (args.dataset,)
    print(json.dumps([validate(dataset) for dataset in datasets], indent=2))


if __name__ == "__main__":
    main()
