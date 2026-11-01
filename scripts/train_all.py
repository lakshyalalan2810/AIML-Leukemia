"""Sequential, fail-closed orchestration for validated training protocols."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.registry import DATASET_REGISTRY, load_dataset_bundle, normalize_dataset
from src.data.splits import get_inner_split, get_outer_fold_data
from src.models.convnextv2_tiny import CONVNEXT_RECIPE, EXPECTED_PARAMS as CONVNEXT_PARAMS
from src.models.efficientnet_b1 import EFFICIENTNET_RECIPE, EXPECTED_PARAMS as EFFICIENTNET_PARAMS
from src.training.artifacts import FitInspection, FitState, inspect_fit
from src.training.losses import compute_class_weights
from src.training.protocol import build_fit_metadata, fit_artifact_paths, review_jobs, validate_locked_config


DATASETS = {
    name: {**spec, "active_samples": spec["expected_oof_count"], "folds": tuple(range(5)),
           "models": spec["supported_models"], "protocol": spec["scientific_claim"]}
    for name, spec in DATASET_REGISTRY.items()
}

MODEL_INFO = {
    "convnextv2_tiny": {
        "config": "convnextv2_tiny.yaml", "script": "train_convnextv2.py",
        "architecture": CONVNEXT_RECIPE, "params": CONVNEXT_PARAMS, "size": 224,
        "condition": "convnextv2_tiny_raw_weighted_ce",
    },
    "efficientnet_b1": {
        "config": "efficientnet_b1.yaml", "script": "train_efficientnet_b1.py",
        "architecture": EFFICIENTNET_RECIPE, "params": EFFICIENTNET_PARAMS, "size": 240,
        "condition": "efficientnet_b1_raw_weighted_ce",
    },
}


@dataclass(frozen=True)
class Job:
    dataset: str
    model: str
    seed: int
    fold: int
    script: Path
    fit_dir: Path
    oof_path: Path
    metadata: dict
    inspection: FitInspection


def _normalize_dataset(value: str) -> str:
    return normalize_dataset(value)


def build_jobs(dataset: str, model: str, seed: int, fold: int | None) -> tuple[list[Job], list[str]]:
    dataset = _normalize_dataset(dataset)
    if dataset not in {*DATASETS, "all"}:
        raise ValueError(f"Unknown dataset {dataset!r}; choose all, cnmc, allidb1, or allidb2")
    if model not in {*MODEL_INFO, "all"}:
        raise ValueError(f"Unknown model {model!r}; choose all, convnextv2_tiny, or efficientnet_b1")

    selected_datasets = list(DATASETS) if dataset == "all" else [dataset]
    unsupported: list[str] = []
    selected_models = list(MODEL_INFO) if model == "all" else [model]
    jobs: list[Job] = []
    for dataset_name in selected_datasets:
        inventory, splits = load_dataset_bundle(dataset_name, repo_root)
        for model_name in selected_models:
            info = MODEL_INFO[model_name]
            config_path = repo_root / "configs" / info["config"]
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            validate_locked_config(config, model_name, info["architecture"], info["size"], info["params"],
                                   info["condition"], dataset_name)
            for _, selected_fold in review_jobs(seed, fold):
                outer_train, outer_val = get_outer_fold_data(splits, selected_fold)
                inner_train, _ = get_inner_split(inventory, outer_train, seed, selected_fold, splits_data=splits)
                labels = [inventory[sid]["label"] for sid in inner_train]
                metadata = build_fit_metadata(
                    config, splits, model=model_name, architecture=info["architecture"],
                    condition=info["condition"], seed=seed, fold=selected_fold,
                    class_counts=[labels.count(0), labels.count(1)],
                    class_weights=compute_class_weights(labels).tolist(), outer_val_sample_ids=outer_val,
                    dataset=dataset_name, dataset_name=DATASETS[dataset_name]["name"],
                    class_order=DATASETS[dataset_name]["class_mapping"],
                )
                fit_dir, oof_path = fit_artifact_paths(repo_root, config, metadata)
                jobs.append(Job(
                    dataset_name, model_name, seed, selected_fold, repo_root / "scripts" / info["script"],
                    fit_dir, oof_path, metadata, inspect_fit(fit_dir, metadata, oof_path),
                ))
    return jobs, unsupported


def print_status(jobs: list[Job]) -> None:
    print(f"{len(jobs)} requested jobs.")
    print("Dataset | Model | Seed | Fold | State | Epoch | Best epoch")
    for job in jobs:
        epoch = (f"{job.inspection.epoch}/{job.metadata['max_epochs']}"
                 if job.inspection.epoch is not None else "-")
        best = str(job.inspection.best_epoch) if job.inspection.best_epoch is not None else "-"
        print(f"{job.dataset} | {job.model} | {job.seed} | {job.fold} | {job.inspection.state.value} | {epoch} | {best}")


def job_progress(job: Job) -> tuple[int, int]:
    identities = [(dataset, model, fold) for dataset, spec in DATASETS.items()
                  for model in spec["models"] for fold in spec["folds"]]
    return identities.index((job.dataset, job.model, job.fold)) + 1, len(identities)


def execute_jobs(
    jobs: list[Job], *, resume: bool, device: str, dry_run: bool, continue_on_error: bool,
    invoke: Callable[[list[str]], int] | None = None,
) -> int:
    invoke = invoke or (lambda command: subprocess.run(command, check=False).returncode)
    failures = 0
    for job in jobs:
        state = job.inspection.state
        identity = f"{job.dataset}/{job.model}/seed_{job.seed}/fold_{job.fold}"
        job_number, total_jobs = job_progress(job)
        print(
            "\n" + "=" * 80 + "\n"
            f"Job {job_number}/{total_jobs}\n"
            f"dataset={job.dataset} | model={job.model} | seed={job.seed} | fold={job.fold}\n" + "=" * 80,
            flush=True,
        )
        if resume and state == FitState.PARTIAL_RESUMABLE and not dry_run:
            print(
                "\n" + "!" * 80 + "\nRESUMING\n"
                f"Job {job_number}/{total_jobs}\n"
                f"dataset={job.dataset} | model={job.model} | seed={job.seed} | fold={job.fold}\n"
                f"Start epoch: {(job.inspection.epoch or 0) + 1}\n"
                f"Best epoch so far: {job.inspection.best_epoch}\n" + "!" * 80,
                flush=True,
            )
        if not dry_run and state == FitState.COMPLETE:
            print(f"SKIP COMPLETE:\n{identity}")
            continue
        error = None
        if state == FitState.PARTIAL_RESUMABLE and not resume and not dry_run:
            error = "PARTIAL_RESUMABLE; rerun with --resume"
        elif state in {FitState.PARTIAL_INVALID, FitState.INCOMPATIBLE}:
            error = f"{state.value}: {job.inspection.reason}"
        if error:
            print(f"FAIL {identity}: {error}", file=sys.stderr)
            failures += 1
            if not continue_on_error:
                return 1
            continue

        command = [sys.executable, str(repo_root / "scripts" / "progress_runner.py"), str(job.script),
                   "--dataset", job.dataset, "--seed", str(job.seed),
                   "--fold", str(job.fold), "--device", device]
        if resume:
            command.append("--resume")
        if dry_run:
            command.append("--dry-run")
        try:
            code = invoke(command)
        except KeyboardInterrupt:
            print(f"Interrupted safely; resume with:\npython scripts/train_all.py --dataset {job.dataset} --model {job.model} --seed {job.seed} --fold {job.fold} --resume --device {device}")
            return 130
        if code:
            failures += 1
            if not continue_on_error:
                return code
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validated dataset/model/fold fits sequentially")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--model", default="all")
    parser.add_argument("--seed", type=int, default=42, choices=[42])
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true", help="Inspect only; never train or write outputs")
    parser.add_argument("--dry-run", action="store_true", help="Validate components without scientific outputs")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    try:
        jobs, unsupported = build_jobs(args.dataset, args.model, args.seed, args.fold)
    except ValueError as exc:
        parser.error(str(exc))
    for name in unsupported:
        print(f"EXCLUDED {DATASETS[name]['name']}: {DATASETS[name]['protocol']}")
    if args.status:
        print_status(jobs)
        return
    raise SystemExit(execute_jobs(
        jobs, resume=args.resume, device=args.device, dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    ))


if __name__ == "__main__":
    main()
