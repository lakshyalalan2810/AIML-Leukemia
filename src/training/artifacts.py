"""Small atomic-write and completed-fit helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

COMPLETION_FILE = "fit_complete.json"
BEST_CHECKPOINT_FILE = "best_checkpoint.pt"
LAST_CHECKPOINT_FILE = "last_checkpoint.pt"
RESUME_COMPATIBILITY_FILE = "resume_compatibility_v1.json"


class FitState(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL_RESUMABLE = "PARTIAL_RESUMABLE"
    PARTIAL_INVALID = "PARTIAL_INVALID"
    MISSING = "MISSING"
    INCOMPATIBLE = "INCOMPATIBLE"


@dataclass(frozen=True)
class FitInspection:
    state: FitState
    reason: str
    epoch: int | None = None
    best_epoch: int | None = None


IDENTITY_FIELDS = (
    "dataset", "model", "architecture", "condition", "seed", "fold",
    "input_resolution", "class_order", "loss_mode", "augmentation_mode",
    "protocol_fingerprint", "configuration_fingerprint", "code_fingerprint", "pretrained",
)
_FINGERPRINT_FIELDS = {
    "scientific_protocol_fingerprint", "protocol_fingerprint",
    "configuration_fingerprint", "code_fingerprint",
}
_FIT_IDENTITY_FIELDS = ("dataset", "model", "condition", "seed", "fold")


def normalize_rng_state(state: Any, name: str, expected_size: int | None = None) -> torch.Tensor:
    """Return a validated contiguous CPU ByteTensor accepted by PyTorch RNG APIs."""
    try:
        value = state.detach().to(device="cpu") if isinstance(state, torch.Tensor) else torch.as_tensor(state, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} RNG state is not a numeric tensor or sequence") from exc
    if value.layout != torch.strided or value.ndim != 1:
        raise ValueError(f"{name} RNG state must be a one-dimensional dense sequence")
    if expected_size is not None and value.numel() != expected_size:
        raise ValueError(f"{name} RNG state has {value.numel()} bytes; expected {expected_size}")
    if value.dtype != torch.uint8:
        if value.dtype == torch.bool or value.is_complex() or value.is_quantized:
            raise ValueError(f"{name} RNG state has unsupported dtype {value.dtype}")
        if torch.is_floating_point(value) and (
            not bool(torch.isfinite(value).all()) or not torch.equal(value, value.trunc())
        ):
            raise ValueError(f"{name} RNG state contains non-integer values")
        if bool(((value < 0) | (value > 255)).any()):
            raise ValueError(f"{name} RNG state contains values outside the byte range")
        value = value.to(dtype=torch.uint8)
    return value.contiguous()

RESUME_REQUIRED_FIELDS = {
    "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
    "scaler_state_dict", "completed_epoch", "next_epoch", "best_epoch",
    "best_metrics", "best_model_state_dict", "early_stopping_state", "history",
    "metadata", "current_phase", "rng_state",
    "dataloader_generator_state", "training_finished",
}

_REQUIRED_HISTORY_FIELDS = {
    "epoch", "train_loss", "train_accuracy", "train_balanced_accuracy", "train_mcc",
    "val_loss", "val_nll", "val_accuracy", "val_balanced_accuracy", "val_mcc",
    "val_roc_auc", "val_pr_auc",
}
_REQUIRED_OOF_FIELDS = {
    "sample_id", "seed", "outer_fold", "model", "condition", "true_label", "probability",
}


def _temp_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    )
    handle.close()
    return Path(handle.name)


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    temp = _temp_path(path)
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty CSV artifact")
    path = Path(path)
    temp = _temp_path(path)
    try:
        with temp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_torch_save(path: str | Path, value: Any) -> None:
    path = Path(path)
    temp = _temp_path(path)
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id_hash(values: Sequence[str]) -> str:
    """Hash an ordered identity list without embedding paths in metadata."""
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()


def _approved_resume_compatibility(
    checkpoint_path: Path, actual: Any, expected: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not isinstance(actual, Mapping) or dict(actual) == dict(expected):
        return None
    try:
        record = json.loads((checkpoint_path.parent / RESUME_COMPATIBILITY_FILE).read_text(encoding="utf-8"))
        old_fingerprints = {key: actual.get(key) for key in _FINGERPRINT_FIELDS if key in actual}
        new_fingerprints = {key: expected.get(key) for key in _FINGERPRINT_FIELDS if key in expected}
        actual_semantics = {key: value for key, value in actual.items() if key not in _FINGERPRINT_FIELDS}
        expected_semantics = {key: value for key, value in expected.items() if key not in _FINGERPRINT_FIELDS}
        valid = (
            record.get("compatibility_version") == "resume_compatibility_v1"
            and record.get("checkpoint_sha256") == _sha256(checkpoint_path)
            and record.get("fit_identity") == {key: expected.get(key) for key in _FIT_IDENTITY_FIELDS}
            and all(actual.get(key) == expected.get(key) for key in _FIT_IDENTITY_FIELDS)
            and record.get("old_fingerprints") == old_fingerprints
            and record.get("new_fingerprints") == new_fingerprints
            and actual_semantics == expected_semantics
            and record.get("scientific_equivalence") is True
            and isinstance(record.get("reason"), str) and bool(record["reason"].strip())
            and isinstance(record.get("reviewed_semantic_differences"), list)
            and bool(record["reviewed_semantic_differences"])
        )
        return record if valid else None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _metadata_issue(
    actual: Any, expected: Mapping[str, Any], compatibility: Mapping[str, Any] | None = None
) -> tuple[FitState, str] | None:
    if not isinstance(actual, Mapping):
        return FitState.PARTIAL_INVALID, "checkpoint/manifest metadata is missing or malformed"
    if compatibility is not None:
        return None
    missing = [key for key in IDENTITY_FIELDS if key not in actual]
    if missing:
        return FitState.PARTIAL_INVALID, f"metadata lacks required identity fields: {', '.join(missing)}"
    mismatches = [key for key in IDENTITY_FIELDS if actual.get(key) != expected.get(key)]
    if mismatches:
        return FitState.INCOMPATIBLE, f"metadata mismatch: {', '.join(mismatches)}"
    if dict(actual) != dict(expected):
        return FitState.INCOMPATIBLE, "metadata differs from the requested fit configuration"
    return None


def _validate_resume_payload(
    payload: Any,
    expected_metadata: Mapping[str, Any],
    compatibility: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint is not a dictionary")
    missing = sorted(RESUME_REQUIRED_FIELDS - payload.keys())
    if missing:
        raise ValueError(f"resume checkpoint lacks fields: {', '.join(missing)}")
    issue = _metadata_issue(payload.get("metadata"), expected_metadata, compatibility)
    if issue:
        raise ValueError(issue[1])
    completed = payload.get("completed_epoch")
    next_epoch = payload.get("next_epoch")
    history = payload.get("history")
    if not isinstance(completed, int) or not isinstance(next_epoch, int) or next_epoch != completed + 1:
        raise ValueError("resume epoch counters are inconsistent")
    if not isinstance(history, list) or [row.get("epoch") for row in history] != list(range(1, completed + 1)):
        raise ValueError("resume history is not contiguous through the completed epoch")
    if payload.get("current_phase") not in {"warmup", "full_finetuning"}:
        raise ValueError("resume phase is invalid")
    if not isinstance(payload.get("training_finished"), bool):
        raise ValueError("resume completion state is invalid")
    if completed > expected_metadata.get("max_epochs", completed):
        raise ValueError("resume checkpoint exceeds configured epochs")
    if (not isinstance(payload.get("early_stopping_state"), dict)
            or "patience_counter" not in payload["early_stopping_state"]):
        raise ValueError("resume early-stopping state is invalid")
    rng = payload.get("rng_state")
    if not isinstance(rng, dict) or not {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(rng):
        raise ValueError("resume RNG state is incomplete")
    try:
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
    except (TypeError, ValueError) as exc:
        raise ValueError("resume Python/NumPy RNG state is invalid") from exc
    normalize_rng_state(rng["torch_cpu"], "torch CPU", torch.get_rng_state().numel())
    if not isinstance(payload.get("optimizer_state_dict"), dict):
        raise ValueError("resume optimizer state is invalid")
    scheduler = payload.get("scheduler_state_dict")
    if ((payload["current_phase"] == "warmup" and scheduler is not None)
            or (payload["current_phase"] == "full_finetuning" and not isinstance(scheduler, dict))):
        raise ValueError("resume scheduler state does not match its phase")
    if not isinstance(payload.get("scaler_state_dict"), dict):
        raise ValueError("resume GradScaler state is invalid")
    if payload.get("best_epoch", -1) < 1 or payload.get("best_model_state_dict") is None:
        raise ValueError("resume best-checkpoint state is incomplete")
    if not isinstance(payload.get("model_state_dict"), dict) or not isinstance(payload.get("best_metrics"), dict):
        raise ValueError("resume model/best metric state is invalid")
    normalize_rng_state(
        payload.get("dataloader_generator_state"),
        "DataLoader generator",
        torch.Generator().get_state().numel(),
    )


def validate_resume_payload(
    payload: Any,
    expected_metadata: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any] | None:
    path = Path(checkpoint_path) if checkpoint_path is not None else None
    compatibility = (
        _approved_resume_compatibility(path, payload.get("metadata"), expected_metadata)
        if path is not None and isinstance(payload, dict) else None
    )
    _validate_resume_payload(payload, expected_metadata, compatibility)
    return compatibility


def load_resume_checkpoint(path: str | Path, expected_metadata: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    compatibility = validate_resume_payload(payload, expected_metadata, path)
    if compatibility is not None:
        payload["_resume_compatibility"] = compatibility
    return payload


def write_completion_manifest(
    fit_dir: str | Path,
    artifacts: Sequence[str | Path],
    metadata: Mapping[str, Any],
) -> Path:
    """Validate all artifacts and atomically write the completion marker last."""
    fit_dir = Path(fit_dir)
    paths = [Path(path).resolve() for path in artifacts]
    resolved_fit_dir = fit_dir.resolve()
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty fit artifact: {path}")
        if not (path.name.startswith("oof_") and path.suffix == ".csv") and path.parent != resolved_fit_dir:
            raise ValueError(f"Fit artifact is outside its identity directory: {path}")

    by_name = {path.name: path for path in paths}
    required = {"training_history.csv", "training_history.json", "fit_metadata.json"}
    if not required.issubset(by_name):
        raise ValueError(f"Completion requires artifacts: {sorted(required)}")
    history = json.loads(by_name["training_history.json"].read_text(encoding="utf-8"))
    fit_metadata = json.loads(by_name["fit_metadata.json"].read_text(encoding="utf-8"))
    with by_name["training_history.csv"].open("r", encoding="utf-8", newline="") as f:
        history_csv = list(csv.DictReader(f))
    if not isinstance(history, list) or not history or len(history_csv) != len(history):
        raise ValueError("Training histories are empty or inconsistent")
    epochs = [int(row["epoch"]) for row in history]
    csv_epochs = [int(row["epoch"]) for row in history_csv]
    if epochs != list(range(epochs[0], epochs[-1] + 1)) or csv_epochs != epochs:
        raise ValueError("Training history epochs must be contiguous and unique")
    if not _REQUIRED_HISTORY_FIELDS.issubset(history[0]) or not _REQUIRED_HISTORY_FIELDS.issubset(history_csv[0]):
        raise ValueError("Training history is missing required metric fields")
    if fit_metadata != dict(metadata):
        raise ValueError("fit_metadata.json does not match completion metadata")

    oof_paths = [path for path in paths if path.name.startswith("oof_") and path.suffix == ".csv"]
    if len(oof_paths) != 1:
        raise ValueError("Completion requires exactly one OOF CSV")
    with oof_paths[0].open("r", encoding="utf-8", newline="") as f:
        oof_rows = list(csv.DictReader(f))
    expected_oof_rows = metadata.get("outer_val_count")
    if not oof_rows or not _REQUIRED_OOF_FIELDS.issubset(oof_rows[0]):
        raise ValueError("OOF CSV is empty or has an invalid schema")
    if expected_oof_rows is not None and len(oof_rows) != expected_oof_rows:
        raise ValueError(f"Expected {expected_oof_rows} OOF rows, found {len(oof_rows)}")
    if len({row["sample_id"] for row in oof_rows}) != len(oof_rows):
        raise ValueError("OOF CSV contains duplicate sample IDs")
    expected_oof_hash = metadata.get("outer_val_sample_ids_sha256")
    if expected_oof_hash and stable_id_hash([row["sample_id"] for row in oof_rows]) != expected_oof_hash:
        raise ValueError("OOF sample IDs do not match the requested outer-fold manifest")
    for row in oof_rows:
        if (int(row["seed"]) != metadata.get("seed") or int(row["outer_fold"]) != metadata.get("fold")
                or row["model"] != metadata.get("model") or row["condition"] != metadata.get("condition")):
            raise ValueError("OOF CSV metadata disagrees with fit metadata")

    checkpoint = by_name.get(BEST_CHECKPOINT_FILE)
    if checkpoint is None:
        raise ValueError(f"Completion requires {BEST_CHECKPOINT_FILE}")
    if any(path.suffix == ".pt" and path != checkpoint for path in paths):
        raise ValueError(f"Completion may only certify {BEST_CHECKPOINT_FILE}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not payload.get("model_state_dict"):
        raise ValueError("Checkpoint is missing model_state_dict")
    if payload.get("metadata") != dict(metadata):
        raise ValueError("Checkpoint metadata does not match fit metadata")

    checkpoint_record = {
        "path": str(checkpoint),
        "name": BEST_CHECKPOINT_FILE,
        "size": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
    }
    manifest = {
        "status": "complete",
        "metadata": dict(metadata),
        "scientific_checkpoint": checkpoint_record,
        "artifacts": {
            str(path): {"size": path.stat().st_size, "sha256": _sha256(path)} for path in paths
        },
    }
    marker = fit_dir / COMPLETION_FILE
    atomic_write_json(marker, manifest)
    return marker


def fit_is_complete(fit_dir: str | Path, expected_metadata: Mapping[str, Any]) -> bool:
    marker = Path(fit_dir) / COMPLETION_FILE
    if not marker.is_file():
        return False
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("metadata") != dict(expected_metadata):
            return False
        artifacts = manifest["artifacts"]
        paths = [Path(raw_path) for raw_path in artifacts]
        resolved_fit_dir = Path(fit_dir).resolve()
        checkpoint_path = (resolved_fit_dir / BEST_CHECKPOINT_FILE).resolve()
        scientific = manifest.get("scientific_checkpoint")
        checkpoint_record = artifacts.get(str(checkpoint_path))
        if (not isinstance(scientific, dict)
                or scientific.get("path") != str(checkpoint_path)
                or scientific.get("name") != BEST_CHECKPOINT_FILE
                or checkpoint_record != {"size": scientific.get("size"), "sha256": scientific.get("sha256")}):
            return False
        if any(path.resolve().parent != resolved_fit_dir
               for path in paths if not (path.name.startswith("oof_") and path.suffix == ".csv")):
            return False
        names = {path.name for path in paths}
        if not {"training_history.csv", "training_history.json", "fit_metadata.json"}.issubset(names):
            return False
        if {path.resolve() for path in paths if path.suffix == ".pt"} != {checkpoint_path}:
            return False
        if sum(path.name.startswith("oof_") and path.suffix == ".csv" for path in paths) != 1:
            return False
        for raw_path, expected in artifacts.items():
            path = Path(raw_path)
            if not path.is_file() or path.stat().st_size != expected["size"] or _sha256(path) != expected["sha256"]:
                return False
        by_name = {path.name: path for path in paths}
        history = json.loads(by_name["training_history.json"].read_text(encoding="utf-8"))
        with by_name["training_history.csv"].open("r", encoding="utf-8", newline="") as f:
            history_csv = list(csv.DictReader(f))
        if (not isinstance(history, list) or not history or len(history_csv) != len(history)
                or not _REQUIRED_HISTORY_FIELDS.issubset(history[0])
                or not _REQUIRED_HISTORY_FIELDS.issubset(history_csv[0])):
            return False
        epochs = [int(row["epoch"]) for row in history]
        if not epochs or epochs != list(range(epochs[0], epochs[-1] + 1)):
            return False
        if [int(row["epoch"]) for row in history_csv] != epochs:
            return False
        if json.loads(by_name["fit_metadata.json"].read_text(encoding="utf-8")) != dict(expected_metadata):
            return False
        oof_path = next(path for path in paths if path.name.startswith("oof_") and path.suffix == ".csv")
        with oof_path.open("r", encoding="utf-8", newline="") as f:
            oof_rows = list(csv.DictReader(f))
        if (not oof_rows or not _REQUIRED_OOF_FIELDS.issubset(oof_rows[0])
                or len({row["sample_id"] for row in oof_rows}) != len(oof_rows)):
            return False
        if len(oof_rows) != expected_metadata.get("outer_val_count", len(oof_rows)):
            return False
        expected_oof_hash = expected_metadata.get("outer_val_sample_ids_sha256")
        if expected_oof_hash and stable_id_hash([row["sample_id"] for row in oof_rows]) != expected_oof_hash:
            return False
        for row in oof_rows:
            if (int(row["seed"]) != expected_metadata.get("seed")
                    or int(row["outer_fold"]) != expected_metadata.get("fold")
                    or row["model"] != expected_metadata.get("model")
                    or row["condition"] != expected_metadata.get("condition")):
                return False
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("metadata") != dict(expected_metadata) or not checkpoint.get("model_state_dict"):
            return False
        return True
    except Exception:
        return False


def refuse_existing_fit(fit_dir: str | Path, expected_metadata: Mapping[str, Any]) -> None:
    fit_dir = Path(fit_dir)
    marker = fit_dir / COMPLETION_FILE
    existing = [p for p in fit_dir.glob("*") if p.is_file()] if fit_dir.is_dir() else []
    if marker.is_file():
        state = "compatible" if fit_is_complete(fit_dir, expected_metadata) else "incompatible/invalid"
        raise FileExistsError(f"Refusing to overwrite {state} completed fit: {fit_dir}")
    if existing:
        raise FileExistsError(f"Refusing to overwrite partial fit without completion marker: {fit_dir}")


def inspect_fit(
    fit_dir: str | Path,
    expected_metadata: Mapping[str, Any],
    oof_path: str | Path | None = None,
) -> FitInspection:
    """Classify one fit without mutating its artifacts."""
    fit_dir = Path(fit_dir)
    marker = fit_dir / COMPLETION_FILE
    oof_path = Path(oof_path) if oof_path is not None else None
    files = sorted(path for path in fit_dir.glob("*") if path.is_file()) if fit_dir.is_dir() else []

    if marker.is_file():
        try:
            manifest = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return FitInspection(FitState.PARTIAL_INVALID, f"invalid completion manifest: {exc}")
        issue = _metadata_issue(manifest.get("metadata"), expected_metadata)
        if issue:
            return FitInspection(*issue)
        if not fit_is_complete(fit_dir, expected_metadata):
            return FitInspection(FitState.PARTIAL_INVALID, "completion manifest or an artifact hash is invalid")
        if oof_path is not None and str(oof_path.resolve()) not in manifest.get("artifacts", {}):
            return FitInspection(FitState.PARTIAL_INVALID, "completion manifest references an unexpected OOF path")
        epoch = best_epoch = None
        try:
            history = json.loads((fit_dir / "training_history.json").read_text(encoding="utf-8"))
            epoch = int(history[-1]["epoch"])
            best_payload = torch.load(fit_dir / BEST_CHECKPOINT_FILE, map_location="cpu", weights_only=False)
            best_epoch = best_payload.get("best_epoch")
        except (KeyError, IndexError, OSError, TypeError, ValueError):
            pass
        return FitInspection(FitState.COMPLETE, "completion manifest and hashes validate", epoch, best_epoch)

    external_oof = bool(oof_path and oof_path.is_file())
    if not files and not external_oof:
        return FitInspection(FitState.MISSING, "no scientific artifacts exist")

    metadata_path = fit_dir / "fit_metadata.json"
    if metadata_path.is_file():
        try:
            issue = _metadata_issue(json.loads(metadata_path.read_text(encoding="utf-8")), expected_metadata)
        except (OSError, json.JSONDecodeError) as exc:
            return FitInspection(FitState.PARTIAL_INVALID, f"fit metadata cannot be read: {exc}")
        if issue:
            return FitInspection(*issue)

    last = fit_dir / LAST_CHECKPOINT_FILE
    if not last.is_file():
        names = [path.name for path in files] + ([oof_path.name] if external_oof and oof_path else [])
        return FitInspection(FitState.PARTIAL_INVALID, f"partial artifacts exist without {LAST_CHECKPOINT_FILE}: {', '.join(names)}")
    try:
        payload = torch.load(last, map_location="cpu", weights_only=False)
    except Exception as exc:
        return FitInspection(FitState.PARTIAL_INVALID, f"resume checkpoint cannot be read: {exc}")
    actual_metadata = payload.get("metadata") if isinstance(payload, dict) else None
    compatibility = _approved_resume_compatibility(last, actual_metadata, expected_metadata)
    issue = _metadata_issue(actual_metadata, expected_metadata, compatibility)
    if issue:
        return FitInspection(*issue)
    try:
        _validate_resume_payload(payload, expected_metadata, compatibility)
    except (TypeError, ValueError) as exc:
        return FitInspection(FitState.PARTIAL_INVALID, str(exc))
    best_path = fit_dir / BEST_CHECKPOINT_FILE
    if best_path.is_file():
        try:
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            return FitInspection(FitState.PARTIAL_INVALID, f"best checkpoint cannot be read: {exc}")
        issue = _metadata_issue(best_payload.get("metadata") if isinstance(best_payload, dict) else None, expected_metadata)
        if issue:
            return FitInspection(*issue)
        if best_payload.get("best_epoch") != payload.get("best_epoch"):
            return FitInspection(FitState.PARTIAL_INVALID, "best checkpoint disagrees with resume state")
    history_json = fit_dir / "training_history.json"
    if history_json.is_file():
        try:
            if json.loads(history_json.read_text(encoding="utf-8")) != payload["history"]:
                return FitInspection(FitState.PARTIAL_INVALID, "JSON history disagrees with resume state")
        except (OSError, json.JSONDecodeError) as exc:
            return FitInspection(FitState.PARTIAL_INVALID, f"JSON history cannot be read: {exc}")
    history_csv = fit_dir / "training_history.csv"
    if history_csv.is_file():
        try:
            with history_csv.open("r", encoding="utf-8", newline="") as f:
                csv_epochs = [int(row["epoch"]) for row in csv.DictReader(f)]
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return FitInspection(FitState.PARTIAL_INVALID, f"CSV history cannot be read: {exc}")
        if csv_epochs != [row["epoch"] for row in payload["history"]]:
            return FitInspection(FitState.PARTIAL_INVALID, "CSV history disagrees with resume state")
    if external_oof and oof_path is not None:
        try:
            with oof_path.open("r", encoding="utf-8", newline="") as f:
                oof_rows = list(csv.DictReader(f))
            if len(oof_rows) != expected_metadata.get("outer_val_count"):
                raise ValueError("row count mismatch")
            if stable_id_hash([row["sample_id"] for row in oof_rows]) != expected_metadata.get("outer_val_sample_ids_sha256"):
                raise ValueError("sample-ID mismatch")
            for row in oof_rows:
                if (int(row["seed"]) != expected_metadata["seed"]
                        or int(row["outer_fold"]) != expected_metadata["fold"]
                        or row["model"] != expected_metadata["model"]
                        or row["condition"] != expected_metadata["condition"]):
                    raise ValueError("fit identity mismatch")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return FitInspection(FitState.PARTIAL_INVALID, f"pre-completion OOF is invalid: {exc}")
    return FitInspection(
        FitState.PARTIAL_RESUMABLE,
        "valid completed-epoch resume checkpoint",
        int(payload["completed_epoch"]),
        int(payload["best_epoch"]),
    )
