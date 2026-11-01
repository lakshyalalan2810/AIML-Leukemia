"""Generate Grad-CAM heatmaps and visual overlays for ConvNeXtV2-Tiny or EfficientNet-B1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np

from src.data.preprocessing import get_dataset_transforms
from src.data.registry import DATASET_REGISTRY, load_dataset_bundle
from src.data.splits import get_outer_fold_data
from src.explainability.gradcam import GradCAM, compute_border_attention_metrics, generate_gradcam_overlay
from src.models.convnextv2_tiny import CONVNEXT_RECIPE, create_convnextv2_tiny
from src.models.efficientnet_b1 import EFFICIENTNET_RECIPE, create_efficientnet_b1
from src.training.artifacts import (
    BEST_CHECKPOINT_FILE, LAST_CHECKPOINT_FILE, atomic_write_csv, atomic_write_json, fit_is_complete,
)


def load_fitted_checkpoint(
    model, checkpoint: str | Path, architecture: str, seed: int, fold: int,
    model_name: str, dataset: str, condition: str, protocol_fingerprint: str,
) -> dict:
    path = Path(checkpoint)
    if path.name == LAST_CHECKPOINT_FILE:
        raise ValueError(
            f"{LAST_CHECKPOINT_FILE} is a resume-only checkpoint and cannot be used for scientific evaluation or Grad-CAM."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Fitted checkpoint not found: {path}")
    canonical = path.parent / BEST_CHECKPOINT_FILE
    if path.name != BEST_CHECKPOINT_FILE or path.resolve() != canonical.resolve():
        raise ValueError(f"Grad-CAM requires the completed fit's canonical {BEST_CHECKPOINT_FILE}")
    marker = path.parent / "fit_complete.json"
    if not marker.is_file():
        raise ValueError(f"Checkpoint fit is incomplete; missing completion marker: {marker}")
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    if not fit_is_complete(path.parent, manifest.get("metadata", {})):
        raise ValueError(f"Checkpoint fit has an invalid completion manifest: {marker}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("Checkpoint is not a fitted pipeline checkpoint")
    metadata = payload.get("metadata")
    expected = {"dataset": dataset, "architecture": architecture, "seed": seed, "fold": fold}
    expected.update(model=model_name, condition=condition, protocol_fingerprint=protocol_fingerprint)
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Checkpoint architecture/seed/fold mismatch; expected {expected}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return metadata


def run_in_memory_check(gradcam: GradCAM, input_size: int, device: torch.device) -> dict[str, float]:
    heatmap = gradcam.generate_heatmap(torch.zeros(1, 3, input_size, input_size, device=device), target_class=1)
    return compute_border_attention_metrics(heatmap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for leukemia models")
    parser.add_argument("--model", type=str, required=True, choices=["convnextv2_tiny", "efficientnet_b1"])
    parser.add_argument("--dataset", default="cnmc", choices=list(DATASET_REGISTRY))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42, choices=[42])
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Execute a non-persistent synthetic in-memory check")
    args = parser.parse_args()

    print(f"==================== Grad-CAM Explainability: {args.model} ====================")
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    if args.model == "convnextv2_tiny":
        model = create_convnextv2_tiny(pretrained=False)
        architecture = CONVNEXT_RECIPE
        input_size = 224
    else:
        model = create_efficientnet_b1(pretrained=False)
        architecture = EFFICIENTNET_RECIPE
        input_size = 240

    print(f"Loading fitted checkpoint from: {args.checkpoint}")
    from scripts.train_all import build_jobs
    jobs, _ = build_jobs(args.dataset, args.model, args.seed, args.fold)
    expected_metadata = jobs[0].metadata
    metadata = load_fitted_checkpoint(
        model, args.checkpoint, architecture, args.seed, args.fold, args.model, args.dataset,
        expected_metadata["condition"], expected_metadata["protocol_fingerprint"],
    )
    if Path(args.checkpoint).resolve() != (jobs[0].fit_dir / BEST_CHECKPOINT_FILE).resolve():
        raise ValueError("Grad-CAM checkpoint path does not match the requested dataset/model/condition/seed/fold")
    out_dir = (repo_root / "outputs" / "gradcam" / args.dataset / args.model / metadata["condition"] /
               f"seed_{args.seed}" / f"fold_{args.fold}")

    model = model.to(device)
    target_layer = model.get_gradcam_target_layer()
    print(f"Grad-CAM Target Layer: {target_layer.__class__.__name__}")

    gradcam = GradCAM(model=model, target_layer=target_layer)
    transform = get_dataset_transforms(args.dataset, input_size)

    if args.dry_run:
        run_in_memory_check(gradcam, input_size, device)
        print(f"[DRY RUN] In-memory Grad-CAM for {args.model} completed; no files written.")
        return

    # Use this checkpoint's outer-fold development samples only.
    inventory, splits = load_dataset_bundle(args.dataset, repo_root)
    _, outer_ids = get_outer_fold_data(splits, args.fold)
    dev_samples = [inventory[sid] for sid in outer_ids]

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix or overwrite Grad-CAM outputs: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    count_to_run = min(args.num_samples, len(dev_samples))
    print(f"Generating heatmaps for {count_to_run} sample(s)...")

    diagnostic_rows = []

    for i in range(count_to_run):
        sample = dev_samples[i]
        img_path = sample["image_path"]
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as exc:
            print(f"Skipping unreadable sample {img_path}: {exc}")
            continue

        input_tensor = transform(img_rgb).unsqueeze(0).to(device)
        heatmap = gradcam.generate_heatmap(input_tensor, target_class=1)
        with torch.no_grad():
            probability = float(torch.softmax(model(input_tensor).float(), dim=-1)[0, 1].item())
        predicted = int(probability >= 0.5)
        true_label = int(sample["label"])
        outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(true_label, predicted)]
        resized_heatmap = np.asarray(
            Image.fromarray(heatmap.astype(np.float32), mode="F").resize(img_rgb.size, resample=Image.BILINEAR)
        )
        diagnostic_rows.append({
            "sample_id": sample["sample_id"],
            "true_label": true_label,
            "predicted_label": predicted,
            "P_ALL": probability,
            "prediction_outcome": outcome,
            **compute_border_attention_metrics(resized_heatmap),
        })
        overlay = generate_gradcam_overlay(img_rgb, heatmap, alpha=0.5)

        sid_clean = sample["sample_id"].replace(":", "_")
        out_path = out_dir / f"gradcam_{args.model}_{sid_clean}.png"
        overlay.save(out_path)
        print(f"  [{i+1}/{count_to_run}] Saved Grad-CAM overlay to: {out_path.name}")

    if not diagnostic_rows:
        raise RuntimeError("No Grad-CAM samples were successfully processed")
    atomic_write_csv(out_dir / "border_attention_samples.csv", diagnostic_rows)
    border_values = np.asarray([row["border_energy_fraction"] for row in diagnostic_rows])
    atomic_write_json(out_dir / "border_attention_summary.json", {
        "name": "Grad-CAM border-attention diagnostic",
        "border_definition": "outer 10% of image width and height on all four edges",
        "sample_count": len(diagnostic_rows),
        "border_energy_fraction": {
            "mean": float(border_values.mean()),
            "median": float(np.median(border_values)),
            "std": float(border_values.std(ddof=1)) if len(border_values) > 1 else 0.0,
        },
    })

    print("============================================================")


if __name__ == "__main__":
    main()
