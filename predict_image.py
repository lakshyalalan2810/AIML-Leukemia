"""Terminal-only live inference demo for the locked leukemia classifiers.

This utility performs no training and writes no scientific artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from scripts.train_all import build_jobs
from src.data.preprocessing import get_dataset_transforms
from src.data.registry import DATASET_REGISTRY
from src.models.convnextv2_tiny import CONVNEXT_RECIPE, create_convnextv2_tiny
from src.models.efficientnet_b1 import EFFICIENTNET_RECIPE, create_efficientnet_b1
from src.training.artifacts import BEST_CHECKPOINT_FILE, FitState


SEED = 42
FOLDS = tuple(range(5))
SEPARATOR = "=" * 60
SUBSEPARATOR = "-" * 60

DATASET_CHOICES = {
    "1": ("cnmc", "C-NMC"),
    "2": ("allidb1", "ALL-IDB1"),
    "3": ("allidb2", "ALL-IDB2"),
}

MODEL_SPECS = {
    "convnextv2_tiny": {
        "display_name": "ConvNeXtV2-Tiny",
        "section_name": "CONVNEXTV2-TINY",
        "architecture": CONVNEXT_RECIPE,
        "input_size": 224,
        "factory": create_convnextv2_tiny,
    },
    "efficientnet_b1": {
        "display_name": "EfficientNet-B1",
        "section_name": "EFFICIENTNET-B1",
        "architecture": EFFICIENTNET_RECIPE,
        "input_size": 240,
        "factory": create_efficientnet_b1,
    },
}


class DemoError(RuntimeError):
    """Expected, user-facing inference error."""


@dataclass(frozen=True)
class CheckpointSpec:
    fold: int
    path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PredictionResult:
    model_name: str
    fold_probabilities: tuple[tuple[float, float], ...]
    mean_probabilities: tuple[float, float]
    predicted_class: int

    @property
    def confidence(self) -> float:
        return self.mean_probabilities[self.predicted_class]


def print_banner() -> None:
    print(SEPARATOR)
    print("LEUKEMIA IMAGE CLASSIFICATION DEMO")
    print(SEPARATOR)


def select_dataset() -> tuple[str, str]:
    """Prompt until the user enters one of the three supported dataset domains."""
    print_banner()
    print()
    print("Select image dataset/domain:")
    print()
    print("1. C-NMC")
    print("2. ALL-IDB1")
    print("3. ALL-IDB2")
    print()

    while True:
        choice = input("Enter choice: ").strip()
        if choice in DATASET_CHOICES:
            return DATASET_CHOICES[choice]
        print("Invalid choice. Please enter 1, 2, or 3.")


def _strip_surrounding_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def get_image_path() -> Path | None:
    """Prompt for a readable file path; an empty entry exits cleanly."""
    while True:
        raw = _strip_surrounding_quotes(input("\nEnter image path: "))
        if not raw:
            print("No image path entered. Exiting leukemia image classification demo.")
            return None

        path = Path(raw).expanduser()
        if path.is_file():
            return path

        print(f"ERROR: Image file not found: {path}")
        print("Enter another path, or press Enter on an empty line to exit.")


def load_image(path: Path) -> Image.Image:
    """Decode the image with the same PIL-based loading convention used by the datasets."""
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DemoError(f"Unable to read image '{path}': {exc}") from exc


def preprocess_image(dataset: str, image: Image.Image, input_size: int) -> torch.Tensor:
    """Apply the locked deterministic evaluation preprocessing for one model size."""
    transform = get_dataset_transforms(dataset, input_size, use_clahe=False)
    tensor = transform(image)
    if tensor.shape != (3, input_size, input_size):
        raise DemoError(
            f"Preprocessing produced shape {tuple(tensor.shape)}; expected (3, {input_size}, {input_size})."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise DemoError("Preprocessing produced non-finite tensor values.")
    return tensor.unsqueeze(0)


def resolve_checkpoints(dataset: str, model_name: str) -> list[CheckpointSpec]:
    """Resolve the five canonical completed best checkpoints through project job logic."""
    jobs, unsupported = build_jobs(dataset, model_name, SEED, None)
    if unsupported:
        raise DemoError(f"Dataset/model combination is unsupported: {dataset}/{model_name}")

    jobs = sorted(jobs, key=lambda job: job.fold)
    folds = tuple(job.fold for job in jobs)
    if len(jobs) != 5 or folds != FOLDS:
        raise DemoError(f"Expected folds 0-4 for {dataset}/{model_name}; found {list(folds)}.")

    checkpoints: list[CheckpointSpec] = []
    for job in jobs:
        if job.inspection.state != FitState.COMPLETE:
            raise DemoError(
                f"Fold {job.fold} is not a completed certified fit for {dataset}/{model_name}: "
                f"{job.inspection.state.value} ({job.inspection.reason})."
            )
        path = job.fit_dir / BEST_CHECKPOINT_FILE
        if path.name != BEST_CHECKPOINT_FILE or path.parent.resolve() != job.fit_dir.resolve():
            raise DemoError(f"Unexpected checkpoint path for fold {job.fold}: {path}")
        if not path.is_file():
            raise DemoError(f"Missing checkpoint for fold {job.fold}: {path}")
        checkpoints.append(CheckpointSpec(job.fold, path, dict(job.metadata)))

    return checkpoints


def build_model(model_name: str) -> torch.nn.Module:
    """Instantiate the exact project architecture without downloading pretrained weights."""
    try:
        return MODEL_SPECS[model_name]["factory"](pretrained=False)
    except Exception as exc:
        raise DemoError(f"Could not construct {MODEL_SPECS[model_name]['display_name']}: {exc}") from exc


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint: CheckpointSpec,
    dataset: str,
    model_name: str,
) -> None:
    """Validate checkpoint identity/metadata and load its state dict strictly."""
    try:
        payload = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise DemoError(f"Could not read checkpoint '{checkpoint.path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise DemoError(f"Malformed checkpoint (expected dictionary): {checkpoint.path}")
    state_dict = payload.get("model_state_dict")
    metadata = payload.get("metadata")
    if not isinstance(state_dict, dict):
        raise DemoError(f"Malformed checkpoint (missing model_state_dict): {checkpoint.path}")
    if not isinstance(metadata, dict):
        raise DemoError(f"Malformed checkpoint (missing metadata): {checkpoint.path}")

    required_fields = (
        "dataset",
        "model",
        "architecture",
        "condition",
        "seed",
        "fold",
        "input_resolution",
        "class_order",
        "protocol_fingerprint",
    )
    missing = [field for field in required_fields if field not in metadata]
    if missing:
        raise DemoError(
            f"Unsupported checkpoint metadata in fold {checkpoint.fold}; missing: {', '.join(missing)}."
        )

    spec = MODEL_SPECS[model_name]
    registry_class_order = DATASET_REGISTRY[dataset]["class_mapping"]
    explicit_expected = {
        "dataset": dataset,
        "model": model_name,
        "architecture": spec["architecture"],
        "seed": SEED,
        "fold": checkpoint.fold,
        "input_resolution": spec["input_size"],
        "class_order": registry_class_order,
    }
    mismatches = [field for field, expected in explicit_expected.items() if metadata.get(field) != expected]
    if mismatches:
        raise DemoError(
            f"Checkpoint identity mismatch in fold {checkpoint.fold} ({', '.join(mismatches)}): {checkpoint.path}"
        )

    if metadata != checkpoint.metadata:
        raise DemoError(
            f"Checkpoint metadata does not exactly match the locked fit metadata for fold {checkpoint.fold}."
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise DemoError(
            f"Checkpoint state_dict is incompatible with {spec['display_name']} in fold {checkpoint.fold}: {exc}"
        ) from exc


def predict_one_checkpoint(model: torch.nn.Module, tensor: torch.Tensor) -> tuple[float, float]:
    """Run one checkpoint and return its two-class softmax probability vector."""
    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        if logits.shape != (1, 2):
            raise DemoError(f"Model returned logits with shape {tuple(logits.shape)}; expected (1, 2).")
        probabilities = torch.softmax(logits.float(), dim=-1)[0]

    if not bool(torch.isfinite(probabilities).all()):
        raise DemoError("Model produced non-finite probabilities.")
    probability_sum = float(probabilities.sum().item())
    if abs(probability_sum - 1.0) > 1e-5:
        raise DemoError(f"Softmax probabilities do not sum to 1 (sum={probability_sum:.8f}).")
    return float(probabilities[0].item()), float(probabilities[1].item())


def ensemble_predict(
    dataset: str,
    model_name: str,
    tensor: torch.Tensor,
    device: torch.device,
) -> PredictionResult:
    """Average five saved-fold softmax vectors for live demonstration inference."""
    checkpoints = resolve_checkpoints(dataset, model_name)
    fold_probabilities: list[tuple[float, float]] = []

    # The five-fold probability average used here is a live demonstration ensemble over
    # previously trained best checkpoints. It is not the pooled OOF evaluation used for
    # the reported scientific metrics.
    try:
        model = build_model(model_name).to(device)
        input_tensor = tensor.to(device)
        for checkpoint in checkpoints:
            load_checkpoint(model, checkpoint, dataset, model_name)
            fold_probabilities.append(predict_one_checkpoint(model, input_tensor))
    except torch.cuda.OutOfMemoryError as exc:
        raise DemoError("CUDA ran out of memory during inference.") from exc
    except DemoError:
        raise
    except RuntimeError as exc:
        if device.type == "cuda" and "cuda" in str(exc).lower():
            raise DemoError(f"CUDA inference failed: {exc}") from exc
        raise

    stacked = torch.tensor(fold_probabilities, dtype=torch.float64)
    mean = stacked.mean(dim=0)
    if not bool(torch.isfinite(mean).all()):
        raise DemoError("Five-fold mean probabilities are non-finite.")
    mean_sum = float(mean.sum().item())
    if abs(mean_sum - 1.0) > 1e-5:
        raise DemoError(f"Five-fold mean probabilities do not sum to 1 (sum={mean_sum:.8f}).")

    mean_probabilities = (float(mean[0].item()), float(mean[1].item()))
    # Match the locked project convention used when writing OOF predicted_label.
    predicted_class = 1 if mean_probabilities[1] >= 0.5 else 0
    return PredictionResult(
        model_name=model_name,
        fold_probabilities=tuple(fold_probabilities),
        mean_probabilities=mean_probabilities,
        predicted_class=predicted_class,
    )


def _class_name_by_index(dataset: str) -> dict[int, str]:
    mapping = DATASET_REGISTRY[dataset]["class_mapping"]
    inverse = {int(index): str(name) for name, index in mapping.items()}
    if set(inverse) != {0, 1}:
        raise DemoError(f"Unsupported class mapping for {dataset}: {mapping}")
    return inverse


def _probability_label(dataset: str, class_index: int) -> str:
    names = _class_name_by_index(dataset)
    polarity = "Positive" if class_index == 1 else "Negative"
    return f"{names[class_index]} / {polarity}"


def _prediction_label(dataset: str, class_index: int) -> str:
    return _probability_label(dataset, class_index).upper()


def _polarity(class_index: int) -> str:
    return "POSITIVE" if class_index == 1 else "NEGATIVE"


def print_model_result(dataset: str, result: PredictionResult) -> None:
    spec = MODEL_SPECS[result.model_name]
    positive_name = _class_name_by_index(dataset)[1]

    print()
    print(SUBSEPARATOR)
    print(spec["section_name"])
    print(SUBSEPARATOR)
    print()
    print(f"Fold probabilities for Positive ({positive_name}):")
    for fold, probabilities in enumerate(result.fold_probabilities):
        print(f"Fold {fold} : {probabilities[1]:.4f}")

    print()
    print("Mean class probabilities:")
    print()
    print(f"P({_probability_label(dataset, 0)}) : {result.mean_probabilities[0]:.4f}")
    print(f"P({_probability_label(dataset, 1)}) : {result.mean_probabilities[1]:.4f}")
    print()
    print(f"Prediction         : {_prediction_label(dataset, result.predicted_class)}")
    print(f"Confidence         : {result.confidence * 100:.2f}%")


def print_final_comparison(convnext: PredictionResult, efficientnet: PredictionResult) -> None:
    agree = convnext.predicted_class == efficientnet.predicted_class
    print()
    print(SUBSEPARATOR)
    print("FINAL COMPARISON")
    print(SUBSEPARATOR)
    print()
    print(f"ConvNeXtV2-Tiny : {_polarity(convnext.predicted_class)}")
    print(f"EfficientNet-B1 : {_polarity(efficientnet.predicted_class)}")
    print()
    print(f"Models agree    : {'YES' if agree else 'NO'}")
    if not agree:
        print()
        print("WARNING:")
        print("The two architectures disagree on this image.")
        print("This is informational only.")

    print()
    print(SEPARATOR)
    print("Research demonstration only.")
    print("This output is not a clinical diagnosis.")
    print(SEPARATOR)


def print_status(step: int, text: str) -> None:
    print(f"[{step}/5] {text:.<34} OK")


def classify_image(dataset: str, dataset_display: str, image_path: Path, device: torch.device) -> None:
    image = load_image(image_path)

    print()
    print(f"Dataset        : {dataset_display}")
    print(f"Image          : {image_path}")
    print(f"Device         : {device.type.upper()}")
    print()
    print_status(1, "Reading image")

    convnext_tensor = preprocess_image(dataset, image, MODEL_SPECS["convnextv2_tiny"]["input_size"])
    efficientnet_tensor = preprocess_image(dataset, image, MODEL_SPECS["efficientnet_b1"]["input_size"])
    print_status(2, "Applying preprocessing")

    convnext = ensemble_predict(dataset, "convnextv2_tiny", convnext_tensor, device)
    print_status(3, "Running ConvNeXtV2-Tiny")

    efficientnet = ensemble_predict(dataset, "efficientnet_b1", efficientnet_tensor, device)
    print_status(4, "Running EfficientNet-B1")
    print_status(5, "Preparing result")

    print_model_result(dataset, convnext)
    print_model_result(dataset, efficientnet)
    print_final_comparison(convnext, efficientnet)


def ask_repeat() -> bool:
    while True:
        choice = input("\nClassify another image? [Y/N]: ").strip().lower()
        if choice in {"y", "yes"}:
            print()
            return True
        if choice in {"n", "no"}:
            return False
        print("Please enter Y or N.")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        try:
            dataset, dataset_display = select_dataset()
            image_path = get_image_path()
            if image_path is None:
                return
            classify_image(dataset, dataset_display, image_path, device)
        except KeyboardInterrupt:
            print("\n\nExiting leukemia image classification demo.")
            return
        except EOFError:
            print("\n\nInput closed. Exiting leukemia image classification demo.")
            return
        except DemoError as exc:
            print(f"\nERROR: {exc}")
        except Exception as exc:
            print(f"\nERROR: Unexpected inference failure: {exc}")

        if not ask_repeat():
            print("Exiting leukemia image classification demo.")
            return


if __name__ == "__main__":
    main()
