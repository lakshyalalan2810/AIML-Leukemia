"""Training script for EfficientNet-B1 on the registered leukemia datasets."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.allidb import AllIdbDataset, order_samples_by_opaque_key as order_allidb_samples
from src.data.cnmc import CnmcDataset, order_samples_by_opaque_key as order_cnmc_samples
from src.data.preprocessing import get_dataset_transforms
from src.data.registry import DATASET_REGISTRY, load_dataset_bundle, normalize_dataset
from src.data.splits import get_inner_split, get_outer_fold_data
from src.evaluation.metrics import compute_classification_metrics
from src.models.efficientnet_b1 import EFFICIENTNET_RECIPE, EXPECTED_PARAMS, create_efficientnet_b1
from src.training.artifacts import (
    BEST_CHECKPOINT_FILE, FitState, LAST_CHECKPOINT_FILE, atomic_write_csv, inspect_fit,
    write_completion_manifest,
)
from src.training.losses import WeightedCrossEntropyLoss, compute_class_weights
from src.training.protocol import build_fit_metadata, fit_artifact_paths, review_jobs, validate_locked_config
from src.training.trainer import TrainConfig, Trainer


def train_single_fit(
    config: dict,
    seed: int,
    fold: int,
    inventory: dict,
    splits_data: dict,
    condition: str = "efficientnet_b1_raw_weighted_ce",
    device: str = "cuda",
    dry_run: bool = False,
    resume: bool = False,
    dataset: str = "cnmc",
) -> dict:
    dataset = normalize_dataset(dataset)
    spec = DATASET_REGISTRY[dataset]
    print(f"\n==================== {spec['name']} / EfficientNet-B1: Seed {seed} | Fold {fold} ({condition}) ====================")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Outer Fold splits
    outer_train_sids, outer_val_sids = get_outer_fold_data(splits_data, fold=fold)

    # 2. Inner Splits for checkpoint selection
    inner_train_sids, inner_val_sids = get_inner_split(
        inventory, outer_train_sids, seed=seed, fold=fold, splits_data=splits_data
    )
    print(f"Dataset split sizes: Inner Train={len(inner_train_sids)} | Inner Val={len(inner_val_sids)} | Outer Val={len(outer_val_sids)}")

    # 3. Create datasets & transforms
    img_size = config["model"]["input_size"]
    train_transform = get_dataset_transforms(dataset, img_size, config["preprocessing"].get("clahe", False))
    val_transform = get_dataset_transforms(dataset, img_size, False)

    order_samples = order_cnmc_samples if dataset == "cnmc" else order_allidb_samples
    inner_train_samples = order_samples([{**inventory[sid], "split_role": "inner_train"} for sid in inner_train_sids])
    inner_val_samples = order_samples([{**inventory[sid], "split_role": "inner_validation"} for sid in inner_val_sids])
    outer_val_samples = order_samples([{**inventory[sid], "split_role": "outer_validation"} for sid in outer_val_sids])

    dataset_class = CnmcDataset if dataset == "cnmc" else AllIdbDataset
    train_dataset = dataset_class(inner_train_samples, transform=train_transform, augment_d4=True, seed=seed, fold=fold)
    val_dataset = dataset_class(inner_val_samples, transform=val_transform)
    outer_val_dataset = dataset_class(outer_val_samples, transform=val_transform)

    batch_size = config["training"]["batch_size"]
    generator = torch.Generator().manual_seed(seed * 100 + fold)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outer_val_loader = DataLoader(outer_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Loss Condition: weights derived strictly from inner train labels
    inner_train_labels = [s["label"] for s in inner_train_samples]
    class_weights = compute_class_weights(inner_train_labels)
    print(f"Class weights (derived strictly from inner train): class0={class_weights[0]:.4f}, class1={class_weights[1]:.4f}")

    if condition != "efficientnet_b1_raw_weighted_ce":
        raise ValueError("Locked review condition is efficientnet_b1_raw_weighted_ce")
    criterion = WeightedCrossEntropyLoss(weight=class_weights)
    class_counts = torch.bincount(torch.tensor(inner_train_labels), minlength=2)
    fit_metadata = build_fit_metadata(
        config, splits_data, model="efficientnet_b1", architecture=EFFICIENTNET_RECIPE,
        condition=condition, seed=seed, fold=fold, class_counts=class_counts.tolist(),
        class_weights=class_weights.tolist(), outer_val_sample_ids=outer_val_sids,
        dataset=dataset, dataset_name=spec["name"], class_order=spec["class_mapping"],
    )
    fit_out_dir, oof_csv = fit_artifact_paths(repo_root, config, fit_metadata)
    inspection = inspect_fit(fit_out_dir, fit_metadata, oof_csv)
    start_epoch = (inspection.epoch or 0) + 1 if inspection.state == FitState.PARTIAL_RESUMABLE else 1
    decision = "DRY-RUN" if dry_run else (
        "SKIP" if inspection.state == FitState.COMPLETE else
        "RESUME" if inspection.state == FitState.PARTIAL_RESUMABLE and resume else
        "START" if inspection.state == FitState.MISSING else "REFUSE"
    )
    print(
        f"dataset={dataset} model=efficientnet_b1 condition={condition} seed={seed} fold={fold}\n"
        f"state={inspection.state.value} decision={decision} start_epoch={start_epoch} "
        f"max_epochs={config['training']['max_epochs']} device={device} pretrained={not dry_run}\n"
        f"protocol_fingerprint={fit_metadata['protocol_fingerprint']}"
    )

    if not dry_run:
        if inspection.state == FitState.COMPLETE:
            print(f"SKIP COMPLETE:\n{dataset}/efficientnet_b1/{condition}/seed_{seed}/fold_{fold}")
            return {"status": "skipped_complete"}
        if inspection.state == FitState.PARTIAL_RESUMABLE and not resume:
            raise RuntimeError(f"PARTIAL_RESUMABLE: {inspection.reason}; rerun with --resume")
        if inspection.state in {FitState.PARTIAL_INVALID, FitState.INCOMPATIBLE}:
            raise RuntimeError(f"{inspection.state.value}: {inspection.reason}")

    # 5. Create Model only after a completed fit has had a chance to skip.
    print(f"Loading {'uninitialized dry-run' if dry_run else 'pretrained'} EfficientNet-B1 backbone...")
    model = create_efficientnet_b1(pretrained=not dry_run, drop_rate=config["model"].get("dropout", 0.2))

    # Dry run check
    if dry_run:
        print("\n[DRY RUN] Testing single forward and backward pass...")
        device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        model = model.to(device_obj)
        criterion = criterion.to(device_obj)

        batch = next(iter(train_loader))
        images = batch["image"].to(device_obj)
        labels = batch["label"].to(device_obj)

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        val_batch = next(iter(val_loader))
        v_images = val_batch["image"].to(device_obj)
        v_labels = val_batch["label"].to(device_obj)
        with torch.no_grad():
            v_logits = model(v_images)
            v_loss = criterion(v_logits, v_labels)

        print(f"[DRY RUN] Forward pass loss: {loss.item():.4f}, Val loss: {v_loss.item():.4f}")
        print("[DRY RUN] EfficientNet-B1 verification SUCCESSFUL.")
        return {"status": "dry_run_success"}

    # Setup Trainer
    t_cfg = config["training"]
    train_config = TrainConfig(
        warmup_epochs=t_cfg["warmup_epochs"],
        max_epochs=t_cfg["max_epochs"],
        min_executed_epochs=t_cfg["min_executed_epochs"],
        head_warmup_lr=float(t_cfg["head_warmup_lr"]),
        backbone_lr=float(t_cfg["backbone_lr"]),
        head_lr=float(t_cfg["head_lr"]),
        weight_decay=float(t_cfg["weight_decay"]),
        grad_accum_steps=t_cfg.get("grad_accum_steps", 1),
        grad_clip=float(t_cfg.get("grad_clip", 1.0)),
        patience=t_cfg["patience"],
        amp=t_cfg.get("amp", True),
        amp_init_scale=float(t_cfg.get("amp_init_scale", 32768.0)),
        selection_criteria=config["selection"]["criteria"],
    )

    trainer = Trainer(model, criterion, train_config, device=device, output_dir=fit_out_dir)

    resume_path = fit_out_dir / LAST_CHECKPOINT_FILE if inspection.state == FitState.PARTIAL_RESUMABLE else None
    if resume_path:
        print(f"RESUMING:\n{dataset}/efficientnet_b1/{condition}/seed_{seed}/fold_{fold}\n\n"
              f"FROM EPOCH:\n{start_epoch}\n\nBEST EPOCH SO FAR:\n{inspection.best_epoch}")
    else:
        print("Starting training...")
    try:
        result = trainer.fit(
            train_loader, val_loader, checkpoint_name=BEST_CHECKPOINT_FILE, fit_metadata=fit_metadata,
            resume_checkpoint=resume_path,
        )
    except KeyboardInterrupt:
        print(f"Interrupted safely; resume with:\npython scripts/train_efficientnet_b1.py --dataset {dataset} --seed {seed} --fold {fold} --resume --condition {condition} --device {device}")
        raise
    print(f"Training complete. Best epoch: {result['best_epoch']}")

    # 6. Evaluate on Outer Validation (OOF) using best checkpoint
    print("Evaluating OOF on outer fold validation set...")
    best_ckpt_path = fit_out_dir / BEST_CHECKPOINT_FILE
    if not best_ckpt_path.is_file():
        raise RuntimeError(f"Training did not produce a checkpoint: {best_ckpt_path}")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=trainer.device, weights_only=False)["model_state_dict"])
    model.eval()

    oof_records = []
    y_true_oof = []
    y_prob_oof = []

    with torch.no_grad():
        for batch in outer_val_loader:
            imgs = batch["image"].to(trainer.device)
            lbls = batch["label"].to(trainer.device)
            logits = model(imgs)
            probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()

            for i, sid in enumerate(batch["sample_id"]):
                prob = float(probs[i])
                true_lbl = int(lbls[i].cpu().item())
                oof_records.append({
                    "sample_id": sid,
                    "dataset": dataset,
                    "protocol_fingerprint": fit_metadata["protocol_fingerprint"],
                    "group_id": batch["group_id"][i],
                    "model": "efficientnet_b1",
                    "condition": condition,
                    "seed": seed,
                    "outer_fold": fold,
                    "true_label": true_lbl,
                    "predicted_label": 1 if prob >= 0.5 else 0,
                    "probability": prob,
                    **({"subject_id": batch["subject_id"][i]} if dataset == "cnmc" else {}),
                    **({
                        "resolution_family": batch["resolution_family"][i],
                        "width": int(batch["width"][i]), "height": int(batch["height"][i]),
                        "aspect_ratio": float(batch["aspect_ratio"][i]),
                        "padding_fraction": float(batch["padding_fraction"][i]),
                    } if dataset == "allidb1" else {}),
                    **({"source_parent_id": batch["source_parent_id"][i]} if dataset == "allidb2" else {}),
                })
                y_true_oof.append(true_lbl)
                y_prob_oof.append(prob)

    atomic_write_csv(oof_csv, oof_records)
    write_completion_manifest(fit_out_dir, [
        best_ckpt_path,
        fit_out_dir / "training_history.csv",
        fit_out_dir / "training_history.json",
        fit_out_dir / "fit_metadata.json",
        oof_csv,
    ], fit_metadata)

    metrics = compute_classification_metrics(y_true_oof, y_prob_oof)
    print(f"OOF Metrics (Seed {seed}, Fold {fold}): Acc={metrics['accuracy']:.4f}, Sens={metrics['sensitivity']:.4f}, Spec={metrics['specificity']:.4f}, MCC={metrics['mcc']:.4f}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the locked seed-42 EfficientNet-B1 registered-dataset pipeline",
        epilog=r"Review command: python scripts\train_efficientnet_b1.py --seed 42 --condition efficientnet_b1_raw_weighted_ce --device cuda",
    )
    parser.add_argument("--config", type=str, default=str(repo_root / "configs" / "efficientnet_b1.yaml"))
    parser.add_argument("--dataset", default="cnmc", choices=["cnmc", "allidb1", "allidb2"])
    parser.add_argument("--condition", type=str, default="efficientnet_b1_raw_weighted_ce", choices=["efficientnet_b1_raw_weighted_ce"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42, choices=[42])
    parser.add_argument("--fold", type=int, default=None, choices=range(5))
    parser.add_argument("--resume", action="store_true", help="Resume a compatible completed-epoch checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Execute single batch dry-run")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_locked_config(
        config, "efficientnet_b1", EFFICIENTNET_RECIPE, 240, EXPECTED_PARAMS,
        "efficientnet_b1_raw_weighted_ce", args.dataset
    )

    inventory, splits_data = load_dataset_bundle(args.dataset, repo_root)

    jobs = review_jobs(args.seed, args.fold)

    if args.dry_run:
        # Run exactly 1 fold dry-run
        seed, fold = jobs[0]
        train_single_fit(config, seed=seed, fold=fold, inventory=inventory, splits_data=splits_data,
                         condition=args.condition, device=args.device, dry_run=True, resume=args.resume,
                         dataset=args.dataset)
        return

    for seed, fold in jobs:
        train_single_fit(config, seed=seed, fold=fold, inventory=inventory, splits_data=splits_data,
                         condition=args.condition, device=args.device, dry_run=False, resume=args.resume,
                         dataset=args.dataset)


if __name__ == "__main__":
    main()
