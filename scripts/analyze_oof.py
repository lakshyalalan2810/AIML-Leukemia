"""Consolidate dataset-aware OOF predictions and compute registered metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.evaluation.oof import OOFPool, summarize_oof_metrics
from src.data.registry import DATASET_REGISTRY, load_dataset_bundle
from src.training.artifacts import atomic_write_json, fit_is_complete
from scripts.train_all import build_jobs


def load_model_oof(
    pred_dir: Path, model_name: str, condition: str, seed: int,
    expected_metadata_by_fold: dict[int, dict], dataset: str = "cnmc",
) -> OOFPool:
    pool = OOFPool(model_name=model_name, condition=condition, dataset=dataset)
    expected = {pred_dir / f"oof_fold_{fold}.csv" for fold in range(5)}
    actual = set(pred_dir.glob("*.csv")) if pred_dir.is_dir() else set()
    if actual != expected:
        raise ValueError(
            f"{model_name} OOF files mismatch; missing={sorted(str(p.name) for p in expected - actual)}, "
            f"unexpected={sorted(str(p.name) for p in actual - expected)}"
        )

    for fold, csv_file in enumerate(sorted(expected)):
        marker = (repo_root / "outputs" / "models" / model_name / dataset / condition /
                  f"seed_{seed}" / f"fold_{fold}" / "fit_complete.json")
        if not marker.is_file():
            raise ValueError(f"OOF fit lacks completion marker: {marker}")
        metadata = json.loads(marker.read_text(encoding="utf-8")).get("metadata", {})
        expected_metadata = expected_metadata_by_fold[fold]
        if metadata != expected_metadata:
            raise ValueError(f"OOF fit metadata mismatch: {marker}")
        if not fit_is_complete(marker.parent, expected_metadata):
            raise ValueError(f"OOF fit has invalid completion manifest: {marker}")

        with csv_file.open("r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
            pool.add_fit_predictions(
                seed=seed, fold=fold, records=records,
                expected_protocol_fingerprint=expected_metadata["protocol_fingerprint"],
            )

    return pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the locked seed-42 five-fold development OOF predictions",
        epilog="Review command: python scripts/analyze_oof.py --seed 42 --threshold 0.5",
    )
    parser.add_argument("--seed", type=int, default=42, choices=[42])
    parser.add_argument("--dataset", default="cnmc", choices=list(DATASET_REGISTRY))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print("============================================================")
    print("DEVELOPMENT OUT-OF-FOLD (OOF) EVALUATION & MODEL COMPARISON")
    print("============================================================")

    convnext_condition = "convnextv2_tiny_raw_weighted_ce"
    effnet_condition = "efficientnet_b1_raw_weighted_ce"
    convnext_pred_dir = (repo_root / "outputs" / "predictions" / "convnextv2_tiny" / args.dataset /
                         convnext_condition / f"seed_{args.seed}")
    effnet_pred_dir = (repo_root / "outputs" / "predictions" / "efficientnet_b1" / args.dataset /
                       effnet_condition / f"seed_{args.seed}")

    inventory, splits = load_dataset_bundle(args.dataset, repo_root)
    jobs, _ = build_jobs(args.dataset, "all", args.seed, None)
    expected_metadata = {
        model: {job.fold: job.metadata for job in jobs if job.model == model}
        for model in ("convnextv2_tiny", "efficientnet_b1")
    }
    expected_fold_by_sample = ({
        row["sample_id"]: int(row["fold"]) for row in splits["records"] if row["eligible"]
    } if "records" in splits else {
        sid: int(fold["fold"]) for fold in splits["outer_folds"] for sid in fold["sample_ids"]
    })
    convnext_pool = load_model_oof(
        convnext_pred_dir, "convnextv2_tiny", convnext_condition, args.seed,
        expected_metadata["convnextv2_tiny"], args.dataset,
    )
    effnet_pool = load_model_oof(
        effnet_pred_dir, "efficientnet_b1", effnet_condition, args.seed,
        expected_metadata["efficientnet_b1"], args.dataset,
    )
    convnext_pool.validate(inventory, expected_fold_by_sample)
    effnet_pool.validate(inventory, expected_fold_by_sample)

    print(f"ConvNeXtV2-Tiny fits found: {len(convnext_pool.fits)} / 5")
    print(f"EfficientNet-B1 fits found: {len(effnet_pool.fits)} / 5")

    convnext_eval = convnext_pool.evaluate_all(threshold=args.threshold) if convnext_pool.fits else None
    effnet_eval = effnet_pool.evaluate_all(threshold=args.threshold) if effnet_pool.fits else None

    if convnext_eval:
        metrics_out = repo_root / "outputs" / "metrics" / args.dataset / "convnextv2_tiny" / convnext_condition / f"seed_{args.seed}"
        metrics_out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(metrics_out / "oof_summary.json", convnext_eval)
        print(f"Saved ConvNeXtV2 metrics to: {metrics_out / 'oof_summary.json'}")

    if effnet_eval:
        metrics_out = repo_root / "outputs" / "metrics" / args.dataset / "efficientnet_b1" / effnet_condition / f"seed_{args.seed}"
        metrics_out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(metrics_out / "oof_summary.json", effnet_eval)
        print(f"Saved EfficientNet-B1 metrics to: {metrics_out / 'oof_summary.json'}")

    summary_md = summarize_oof_metrics(convnext_eval, effnet_eval)
    print("\n" + summary_md)

    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"{args.dataset}_model_comparison_oof.md"
    report_file.write_text(summary_md, encoding="utf-8")
    print(f"\nWritten comparison report to: {report_file}")
    print("============================================================")


if __name__ == "__main__":
    main()
