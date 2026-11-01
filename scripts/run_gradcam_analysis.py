"""Run the locked post-training OOF Grad-CAM and border-attention analysis.

This script is deliberately read-only with respect to scientific training artifacts.
It validates all five completed fits and pooled OOF rows for each requested
dataset/model pair, selects deterministic representative OOF samples, loads each
sample's outer-held-out fold checkpoint, and writes explainability artifacts only
under ``outputs/gradcam`` plus the final report-ready summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.analyze_oof import load_model_oof
from scripts.run_gradcam import load_fitted_checkpoint
from scripts.train_all import MODEL_INFO, build_jobs
from src.data.preprocessing import IMAGENET_MEAN, get_dataset_transforms
from src.data.registry import DATASET_REGISTRY, load_dataset_bundle
from src.explainability.gradcam import GradCAM, compute_border_attention_metrics, generate_gradcam_overlay
from src.models.convnextv2_tiny import CONVNEXT_RECIPE, create_convnextv2_tiny
from src.models.efficientnet_b1 import EFFICIENTNET_RECIPE, create_efficientnet_b1
from src.training.artifacts import BEST_CHECKPOINT_FILE, atomic_write_csv, atomic_write_json


SEED = 42
TARGET_COUNT = 25
OUTCOMES = ("TP", "TN", "FP", "FN")
ERROR_OUTCOMES = frozenset({"FP", "FN"})
MODEL_SPECS = {
    "convnextv2_tiny": {
        "display": "ConvNeXtV2-Tiny",
        "architecture": CONVNEXT_RECIPE,
        "input_size": 224,
        "factory": create_convnextv2_tiny,
        "target_layer": "backbone.stages[-1].blocks[-1]",
    },
    "efficientnet_b1": {
        "display": "EfficientNet-B1",
        "architecture": EFFICIENTNET_RECIPE,
        "input_size": 240,
        "factory": create_efficientnet_b1,
        "target_layer": "model.conv_head",
    },
}


def _outcome(true_label: int, predicted_label: int) -> str:
    return {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[
        (int(true_label), int(predicted_label))
    ]


def _evenly_spaced(records: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select deterministic score-quantile representatives without viewing images."""
    ordered = sorted(records, key=lambda row: (float(row["probability"]), str(row["sample_id"])))
    if count <= 0:
        return []
    if count >= len(ordered):
        return [dict(row) for row in ordered]
    indices = np.linspace(0, len(ordered) - 1, num=count, dtype=int).tolist()
    return [dict(ordered[index]) for index in indices]


def _balanced_subgroup_sample(
    records: Sequence[Mapping[str, Any]], count: int, subgroup_field: str | None,
) -> list[dict[str, Any]]:
    if not subgroup_field or not records or subgroup_field not in records[0]:
        return _evenly_spaced(records, count)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row.get(subgroup_field, ""))].append(row)
    names = sorted(groups)
    if len(names) <= 1:
        return _evenly_spaced(records, count)

    desired = {name: count // len(names) for name in names}
    for name in names[: count % len(names)]:
        desired[name] += 1
    selected: list[dict[str, Any]] = []
    unused: list[Mapping[str, Any]] = []
    for name in names:
        take = min(desired[name], len(groups[name]))
        chosen = _evenly_spaced(groups[name], take)
        selected.extend(chosen)
        chosen_ids = {row["sample_id"] for row in chosen}
        unused.extend(row for row in groups[name] if row["sample_id"] not in chosen_ids)
    if len(selected) < count:
        selected.extend(_evenly_spaced(unused, count - len(selected)))
    return selected[:count]


def _allocate_balanced(count: int, outcomes: Sequence[str]) -> dict[str, int]:
    if count <= 0 or not outcomes:
        return {name: 0 for name in outcomes}
    base, remainder = divmod(count, len(outcomes))
    return {name: base + (index < remainder) for index, name in enumerate(outcomes)}


def select_representative_samples(
    records: Sequence[Mapping[str, Any]], dataset: str, target_count: int = TARGET_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose a deterministic, outcome-aware OOF sample without visual cherry-picking."""
    annotated = []
    for source in records:
        row = dict(source)
        row["true_label"] = int(row["true_label"])
        row["predicted_label"] = int(row["predicted_label"])
        row["outer_fold"] = int(row["outer_fold"])
        row["seed"] = int(row["seed"])
        row["probability"] = float(row["probability"])
        row["prediction_outcome"] = _outcome(row["true_label"], row["predicted_label"])
        annotated.append(row)

    by_outcome = {name: [row for row in annotated if row["prediction_outcome"] == name] for name in OUTCOMES}
    available = [name for name in OUTCOMES if by_outcome[name]]
    errors = by_outcome["FP"] + by_outcome["FN"]
    rare_error_limit = max(1, target_count // 4)
    subgroup_field = "resolution_family" if dataset == "allidb1" else None
    selected: list[dict[str, Any]] = []
    quota: dict[str, int] = {name: 0 for name in OUTCOMES}
    strategy: str

    if len(errors) <= rare_error_limit:
        strategy = "include_all_rare_errors_then_balance_correct_cases"
        for name in ("FP", "FN"):
            quota[name] = len(by_outcome[name])
            selected.extend(_balanced_subgroup_sample(by_outcome[name], quota[name], subgroup_field))
        remaining = max(0, target_count - len(selected))
        correct_available = [name for name in ("TP", "TN") if by_outcome[name]]
        correct_quota = _allocate_balanced(remaining, correct_available)
        for name in correct_available:
            quota[name] = min(correct_quota[name], len(by_outcome[name]))
            selected.extend(_balanced_subgroup_sample(by_outcome[name], quota[name], subgroup_field))
    else:
        strategy = "balanced_outcomes_with_score_quantile_representatives"
        balanced = _allocate_balanced(target_count, available)
        for name in available:
            quota[name] = min(balanced[name], len(by_outcome[name]))
            selected.extend(_balanced_subgroup_sample(by_outcome[name], quota[name], subgroup_field))

    selected_ids = {row["sample_id"] for row in selected}
    if len(selected) < min(target_count, len(annotated)):
        remainder = [row for row in annotated if row["sample_id"] not in selected_ids]
        fill = _evenly_spaced(remainder, min(target_count, len(annotated)) - len(selected))
        selected.extend(fill)
        selected_ids.update(row["sample_id"] for row in fill)

    order = {name: index for index, name in enumerate(OUTCOMES)}
    selected = sorted(
        selected,
        key=lambda row: (order[row["prediction_outcome"]], row["outer_fold"], row["probability"], row["sample_id"]),
    )
    for index, row in enumerate(selected, start=1):
        row["selection_rank"] = index

    policy = {
        "target_count": target_count,
        "selected_count": len(selected),
        "strategy": strategy,
        "rare_error_limit": rare_error_limit,
        "visual_content_used_for_selection": False,
        "within_group_selection": "evenly spaced ranks after sorting by OOF probability then sample_id",
        "allidb1_subgroup_rule": (
            "within each outcome, divide the quota across recorded resolution_family values before score-rank selection"
            if dataset == "allidb1" else None
        ),
        "pooled_outcome_counts": {name: len(by_outcome[name]) for name in OUTCOMES},
        "selected_outcome_counts": dict(Counter(row["prediction_outcome"] for row in selected)),
    }
    return selected, policy


def prepare_model_input_image(original: Image.Image, dataset: str, input_size: int) -> Image.Image:
    """Create the exact model-input geometry for faithful overlay/padding inspection."""
    image = original.convert("RGB")
    if dataset == "cnmc":
        return image.resize((input_size, input_size), resample=Image.BICUBIC)
    scale = min(input_size / image.width, input_size / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        resample=Image.BICUBIC,
    )
    fill = tuple(round(value * 255) for value in IMAGENET_MEAN)
    prepared = Image.new("RGB", (input_size, input_size), fill)
    prepared.paste(resized, ((input_size - resized.width) // 2, (input_size - resized.height) // 2))
    return prepared


def verify_target_layer_shape(
    model: torch.nn.Module, target_layer: torch.nn.Module, input_size: int, device: torch.device,
) -> tuple[int, ...]:
    captured: list[tuple[int, ...]] = []

    def hook(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("Grad-CAM target layer output must be a tensor")
        captured.append(tuple(int(value) for value in output.shape))

    handle = target_layer.register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 3, input_size, input_size, device=device))
    finally:
        handle.remove()
    if len(captured) != 1 or len(captured[0]) != 4 or min(captured[0][2:]) < 1:
        raise RuntimeError(f"Unexpected Grad-CAM target tensor shape: {captured}")
    return captured[0]


def _resize_normalized_heatmap(heatmap: np.ndarray, input_size: int) -> np.ndarray:
    resized = np.asarray(
        Image.fromarray(np.asarray(heatmap, dtype=np.float32), mode="F").resize(
            (input_size, input_size), resample=Image.BILINEAR
        ),
        dtype=np.float32,
    )
    resized = np.clip(resized, 0.0, None)
    maximum = float(resized.max())
    if maximum > 0:
        resized /= maximum
    if not np.all(np.isfinite(resized)):
        raise ValueError("Grad-CAM heatmap contains non-finite values after resize")
    return resized


def _metric_stats(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    return {
        "mean": float(data.mean()) if len(data) else float("nan"),
        "median": float(np.median(data)) if len(data) else float("nan"),
        "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {"n": len(rows)}
    for metric in ("border_energy_fraction", "center_energy_fraction", "center_to_border_ratio"):
        result[metric] = _metric_stats([float(row[metric]) for row in rows])
    return result


def _summary_table_rows(rows: Sequence[Mapping[str, Any]], dataset: str) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[Mapping[str, Any]]]] = [("overall", "all", list(rows))]
    for outcome in OUTCOMES:
        subset = [row for row in rows if row["prediction_outcome"] == outcome]
        if subset:
            groups.append(("outcome", outcome, subset))
    if dataset == "allidb1":
        for family in sorted({str(row.get("resolution_family", "")) for row in rows}):
            subset = [row for row in rows if str(row.get("resolution_family", "")) == family]
            groups.append(("resolution_family", family, subset))
    if dataset == "allidb2":
        errors = [row for row in rows if row["prediction_outcome"] in ERROR_OUTCOMES]
        for parent, count in sorted(Counter(str(row.get("source_parent_id", "")) for row in errors).items()):
            if count >= 2:
                subset = [row for row in errors if str(row.get("source_parent_id", "")) == parent]
                groups.append(("error_parent_field", parent, subset))

    flat = []
    for scope_type, scope_value, subset in groups:
        summary = _summarize_rows(subset)
        flat.append({
            "scope_type": scope_type,
            "scope_value": scope_value,
            "n": summary["n"],
            "small_group": summary["n"] < 3,
            "border_mean": summary["border_energy_fraction"]["mean"],
            "border_median": summary["border_energy_fraction"]["median"],
            "border_std": summary["border_energy_fraction"]["std"],
            "center_mean": summary["center_energy_fraction"]["mean"],
            "center_median": summary["center_energy_fraction"]["median"],
            "center_std": summary["center_energy_fraction"]["std"],
            "center_to_border_ratio_mean": summary["center_to_border_ratio"]["mean"],
            "center_to_border_ratio_median": summary["center_to_border_ratio"]["median"],
            "center_to_border_ratio_std": summary["center_to_border_ratio"]["std"],
        })
    return flat


def _make_contact_sheet(rows: Sequence[Mapping[str, Any]], output: Path, dataset: str, columns: int = 5) -> None:
    if not rows:
        return
    thumb = 220
    label_height = 62
    tile_w, tile_h = thumb, thumb + label_height
    row_count = math.ceil(len(rows) / columns)
    sheet = Image.new("RGB", (columns * tile_w, row_count * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        overlay = Image.open(row["overlay_path"]).convert("RGB")
        overlay.thumbnail((thumb, thumb))
        x = (index % columns) * tile_w + (thumb - overlay.width) // 2
        y = (index // columns) * tile_h
        sheet.paste(overlay, (x, y))
        text_y = y + thumb + 2
        extra = ""
        if dataset == "allidb1":
            extra = f" res={row.get('resolution_family', '')}"
        elif dataset == "allidb2":
            extra = f" parent={row.get('source_parent_id', '')}"
        label = (
            f"{row['prediction_outcome']} f{row['outer_fold']} {row['sample_id']}\n"
            f"B={float(row['border_energy_fraction']):.3f} C/B={float(row['center_to_border_ratio']):.2f}{extra}"
        )
        draw.multiline_text((index % columns * tile_w + 2, text_y), label, fill="black", spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _expected_fold_map(splits: Mapping[str, Any]) -> dict[str, int]:
    if "records" in splits:
        return {row["sample_id"]: int(row["fold"]) for row in splits["records"] if row.get("eligible", True)}
    return {
        sample_id: int(fold["fold"])
        for fold in splits["outer_folds"]
        for sample_id in fold["sample_ids"]
    }


def _source_manifest(jobs: Sequence[Any]) -> dict[str, Any]:
    sources = {}
    for job in jobs:
        marker = json.loads((job.fit_dir / "fit_complete.json").read_text(encoding="utf-8"))
        oof_record = marker["artifacts"].get(str(job.oof_path.resolve()))
        sources[str(job.fold)] = {
            "checkpoint": marker["scientific_checkpoint"],
            "oof_path": str(job.oof_path.resolve()),
            "oof_artifact": oof_record,
        }
    return sources


def run_pair(dataset: str, model_name: str, device: torch.device, target_count: int) -> dict[str, Any]:
    spec = MODEL_SPECS[model_name]
    condition = MODEL_INFO[model_name]["condition"]
    jobs, unsupported = build_jobs(dataset, model_name, SEED, None)
    if unsupported or len(jobs) != 5:
        raise ValueError(f"Expected five supported fits for {dataset}/{model_name}")
    jobs_by_fold = {job.fold: job for job in jobs}
    inventory, splits = load_dataset_bundle(dataset, repo_root)
    expected_fold_by_sample = _expected_fold_map(splits)
    pred_dir = repo_root / "outputs" / "predictions" / model_name / dataset / condition / f"seed_{SEED}"
    pool = load_model_oof(
        pred_dir, model_name, condition, SEED,
        {job.fold: job.metadata for job in jobs}, dataset,
    )
    pool.validate(inventory, expected_fold_by_sample)
    pooled = pool.get_seed_pooled(SEED)
    selected, selection_policy = select_representative_samples(pooled, dataset, target_count)
    if not selected:
        raise RuntimeError(f"No OOF samples selected for {dataset}/{model_name}")

    for row in selected:
        sid = row["sample_id"]
        fold = int(row["outer_fold"])
        if expected_fold_by_sample.get(sid) != fold:
            raise ValueError(f"Selected OOF sample is not mapped to its held-out fold: {sid}")
        expected = jobs_by_fold[fold].metadata
        if (
            row["dataset"] != dataset
            or row["model"] != model_name
            or row["condition"] != condition
            or int(row["seed"]) != SEED
            or row["protocol_fingerprint"] != expected["protocol_fingerprint"]
        ):
            raise ValueError(f"Selected OOF row metadata mismatch: {sid}")

    seed_dir = repo_root / "outputs" / "gradcam" / model_name / dataset / condition / f"seed_{SEED}"
    if seed_dir.exists() and any(seed_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix or overwrite existing Grad-CAM analysis: {seed_dir}")

    model = spec["factory"](pretrained=False).to(device)
    target_layer = model.get_gradcam_target_layer()
    if model_name == "convnextv2_tiny":
        expected_layer = model.backbone.stages[-1].blocks[-1]
    else:
        expected_layer = model.model.conv_head
    if target_layer is not expected_layer:
        raise RuntimeError(f"Grad-CAM target accessor disagrees with required layer {spec['target_layer']}")

    first_fold = int(selected[0]["outer_fold"])
    first_job = jobs_by_fold[first_fold]
    load_fitted_checkpoint(
        model, first_job.fit_dir / BEST_CHECKPOINT_FILE, spec["architecture"], SEED, first_fold,
        model_name, dataset, condition, first_job.metadata["protocol_fingerprint"],
    )
    target_shape = verify_target_layer_shape(model, target_layer, int(spec["input_size"]), device)
    gradcam = GradCAM(model=model, target_layer=target_layer)
    transform = get_dataset_transforms(dataset, int(spec["input_size"]))

    seed_dir.mkdir(parents=True, exist_ok=False)
    overlay_root = seed_dir / "overlays"
    heatmap_root = seed_dir / "heatmaps"
    overlay_root.mkdir()
    heatmap_root.mkdir()

    selection_rows = []
    for row in selected:
        selection_rows.append({
            "selection_rank": row["selection_rank"], "sample_id": row["sample_id"],
            "outer_fold": row["outer_fold"], "true_label": row["true_label"],
            "predicted_label": row["predicted_label"], "probability": row["probability"],
            "prediction_outcome": row["prediction_outcome"], "resolution_family": row.get("resolution_family", ""),
            "subject_id": row.get("subject_id", ""), "source_parent_id": row.get("source_parent_id", ""),
            "selection_strategy": selection_policy["strategy"],
        })
    atomic_write_csv(seed_dir / "sample_selection.csv", selection_rows)

    processed: list[dict[str, Any]] = []
    selected_by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_fold[int(row["outer_fold"])].append(row)

    # Exact class reproduction is mandatory. OOF rows are already cryptographically
    # certified to this checkpoint by fit_complete.json; this looser probability
    # guard records/limits batch-size and CUDA-kernel numerical drift during the
    # single-sample Grad-CAM pass without pretending bitwise OOF reproduction.
    probability_tolerance = 5e-2
    for fold in sorted(selected_by_fold):
        job = jobs_by_fold[fold]
        checkpoint = job.fit_dir / BEST_CHECKPOINT_FILE
        metadata = load_fitted_checkpoint(
            model, checkpoint, spec["architecture"], SEED, fold, model_name, dataset,
            condition, job.metadata["protocol_fingerprint"],
        )
        if metadata != job.metadata:
            raise ValueError(f"Checkpoint metadata does not exactly match locked job metadata for fold {fold}")
        model.eval()
        fold_overlay = overlay_root / f"fold_{fold}"
        fold_heatmap = heatmap_root / f"fold_{fold}"
        fold_overlay.mkdir()
        fold_heatmap.mkdir()

        for row in selected_by_fold[fold]:
            sid = row["sample_id"]
            sample = inventory[sid]
            image_path = Path(sample["image_path"])
            with Image.open(image_path) as image:
                original = image.convert("RGB")
            model_input_image = prepare_model_input_image(original, dataset, int(spec["input_size"]))
            input_tensor = transform(original).unsqueeze(0).to(device)
            with torch.no_grad():
                runtime_probability = float(torch.softmax(model(input_tensor).float(), dim=-1)[0, 1].item())
            probability_delta = abs(runtime_probability - float(row["probability"]))
            runtime_predicted = int(runtime_probability >= 0.5)
            if probability_delta > probability_tolerance or runtime_predicted != int(row["predicted_label"]):
                raise RuntimeError(
                    f"Runtime prediction does not reproduce locked OOF row for {sid}: "
                    f"OOF={float(row['probability']):.8f}, runtime={runtime_probability:.8f}, delta={probability_delta:.3g}"
                )

            target_class = int(row["predicted_label"])
            heatmap = gradcam.generate_heatmap(input_tensor, target_class=target_class)
            heatmap_input = _resize_normalized_heatmap(heatmap, int(spec["input_size"]))
            metrics = compute_border_attention_metrics(heatmap_input, border_fraction=0.10)
            sid_clean = str(sid).replace(":", "_").replace("/", "_").replace("\\", "_")
            overlay_path = fold_overlay / f"{sid_clean}.png"
            heatmap_npy_path = fold_heatmap / f"{sid_clean}.npy"
            heatmap_png_path = fold_heatmap / f"{sid_clean}.png"
            np.save(heatmap_npy_path, heatmap_input, allow_pickle=False)
            Image.fromarray(np.round(heatmap_input * 255).astype(np.uint8), mode="L").save(heatmap_png_path)
            generate_gradcam_overlay(model_input_image, heatmap_input, alpha=0.5).save(overlay_path)

            result = {
                "selection_rank": row["selection_rank"],
                "sample_id": sid,
                "dataset": dataset,
                "model": model_name,
                "condition": condition,
                "seed": SEED,
                "outer_fold": fold,
                "protocol_fingerprint": row["protocol_fingerprint"],
                "checkpoint": str(checkpoint.resolve()),
                "target_layer": spec["target_layer"],
                "target_activation_shape": "x".join(str(v) for v in target_shape),
                "gradcam_target_class": target_class,
                "true_label": int(row["true_label"]),
                "predicted_label": int(row["predicted_label"]),
                "prediction_outcome": row["prediction_outcome"],
                "oof_probability": float(row["probability"]),
                "runtime_probability": runtime_probability,
                "runtime_probability_abs_delta": probability_delta,
                "border_energy_fraction": metrics["border_energy_fraction"],
                "center_energy_fraction": metrics["center_energy_fraction"],
                "center_to_border_ratio": metrics["center_to_border_ratio"],
                "resolution_family": row.get("resolution_family", ""),
                "padding_fraction": row.get("padding_fraction", ""),
                "subject_id": row.get("subject_id", ""),
                "source_parent_id": row.get("source_parent_id", ""),
                "source_image": str(image_path),
                "overlay_path": str(overlay_path.resolve()),
                "raw_heatmap_npy": str(heatmap_npy_path.resolve()),
                "raw_heatmap_png": str(heatmap_png_path.resolve()),
            }
            processed.append(result)
            print(
                f"{dataset}/{model_name} fold={fold} {row['prediction_outcome']} {sid} "
                f"border={metrics['border_energy_fraction']:.4f}",
                flush=True,
            )

    processed.sort(key=lambda row: int(row["selection_rank"]))
    atomic_write_csv(seed_dir / "gradcam_samples.csv", processed)
    summary_rows = _summary_table_rows(processed, dataset)
    atomic_write_csv(seed_dir / "gradcam_summary.csv", summary_rows)
    overall = next(row for row in summary_rows if row["scope_type"] == "overall")
    by_outcome_summary = {
        outcome: _summarize_rows([row for row in processed if row["prediction_outcome"] == outcome])
        for outcome in OUTCOMES if any(row["prediction_outcome"] == outcome for row in processed)
    }
    resolution_summary = {}
    if dataset == "allidb1":
        resolution_summary = {
            family: _summarize_rows([row for row in processed if row.get("resolution_family") == family])
            for family in sorted({str(row.get("resolution_family", "")) for row in processed})
        }
    parent_error_summary = {}
    if dataset == "allidb2":
        error_rows = [row for row in processed if row["prediction_outcome"] in ERROR_OUTCOMES]
        parent_error_summary = {
            parent: _summarize_rows([row for row in error_rows if row.get("source_parent_id") == parent])
            for parent in sorted({str(row.get("source_parent_id", "")) for row in error_rows})
        }

    summary_json = {
        "name": "OOF Grad-CAM border-attention diagnostic",
        "dataset": dataset,
        "model": model_name,
        "condition": condition,
        "seed": SEED,
        "sample_count": len(processed),
        "selection_policy": selection_policy,
        "oof_safety": {
            "held_out_fold_required": True,
            "canonical_checkpoint_name": BEST_CHECKPOINT_FILE,
            "all_five_fit_manifests_validated": True,
            "runtime_probability_tolerance": probability_tolerance,
            "max_runtime_probability_abs_delta": max(float(row["runtime_probability_abs_delta"]) for row in processed),
        },
        "target_layer": spec["target_layer"],
        "target_activation_shape": list(target_shape),
        "gradcam_target_semantics": "OOF predicted class for each selected sample",
        "heatmap_space": f"{spec['input_size']}x{spec['input_size']} model-input geometry after locked resize/padding",
        "border_definition": "outer 10% of model-input width and height on all four edges",
        "metric_definition": "normalized non-negative Grad-CAM activation; metrics are computed from heatmaps, never overlays",
        "overall": overall,
        "by_outcome": by_outcome_summary,
        "by_resolution_family": resolution_summary,
        "selected_error_parent_fields": parent_error_summary,
        "source_artifacts": _source_manifest(jobs),
    }
    atomic_write_json(seed_dir / "gradcam_summary.json", summary_json)
    atomic_write_json(seed_dir / "analysis_manifest.json", {
        "analysis": "oof_gradcam_border_attention_v1",
        "dataset": dataset,
        "model": model_name,
        "condition": condition,
        "seed": SEED,
        "target_layer": spec["target_layer"],
        "target_activation_shape": list(target_shape),
        "selected_sample_ids": [row["sample_id"] for row in processed],
        "source_artifacts": _source_manifest(jobs),
    })
    _make_contact_sheet(processed, seed_dir / "contact_sheet.png", dataset)
    errors = [row for row in processed if row["prediction_outcome"] in ERROR_OUTCOMES]
    if errors:
        _make_contact_sheet(errors, seed_dir / "error_contact_sheet.png", dataset, columns=4)

    return {
        "dataset": dataset,
        "model": model_name,
        "condition": condition,
        "seed_dir": str(seed_dir.resolve()),
        "sample_count": len(processed),
        "target_layer": spec["target_layer"],
        "target_activation_shape": list(target_shape),
        "selection_policy": selection_policy,
        "overall": overall,
        "by_outcome": by_outcome_summary,
        "by_resolution_family": resolution_summary,
        "selected_error_parent_fields": parent_error_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-training OOF Grad-CAM diagnostics only")
    parser.add_argument("--dataset", default="all", choices=["all", *DATASET_REGISTRY])
    parser.add_argument("--model", default="all", choices=["all", *MODEL_SPECS])
    parser.add_argument("--seed", type=int, default=SEED, choices=[SEED])
    parser.add_argument("--num-samples", type=int, default=TARGET_COUNT)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    datasets = list(DATASET_REGISTRY) if args.dataset == "all" else [args.dataset]
    models = list(MODEL_SPECS) if args.model == "all" else [args.model]
    results = []
    for dataset in datasets:
        for model_name in models:
            print("=" * 80)
            print(f"POST-TRAINING OOF GRAD-CAM: dataset={dataset} model={model_name} device={device}")
            print("=" * 80)
            results.append(run_pair(dataset, model_name, device, args.num_samples))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    index_path = repo_root / "outputs" / "gradcam" / "gradcam_analysis_index.json"
    existing_results = []
    if index_path.is_file():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing.get("analysis") != "oof_gradcam_border_attention_v1" or existing.get("seed") != SEED:
            raise ValueError(f"Existing Grad-CAM index is incompatible: {index_path}")
        existing_results = list(existing.get("results", []))
    merged = {(row["dataset"], row["model"]): row for row in existing_results}
    for row in results:
        merged[(row["dataset"], row["model"])] = row
    atomic_write_json(index_path, {
        "analysis": "oof_gradcam_border_attention_v1",
        "seed": SEED,
        "device": str(device),
        "results": [merged[key] for key in sorted(merged)],
    })
    print(f"Completed {len(results)} dataset/model Grad-CAM analyses without training.")


if __name__ == "__main__":
    main()
