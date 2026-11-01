"""Clean two-phase trainer with mixed precision, per-epoch metric logging, and checkpoint selection."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..evaluation.metrics import compute_classification_metrics
from .artifacts import (
    LAST_CHECKPOINT_FILE,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    load_resume_checkpoint,
    normalize_rng_state,
)


def _normalize_cuda_rng_states(state: Any) -> list[torch.Tensor]:
    if isinstance(state, (torch.Tensor, np.ndarray)) or (
        isinstance(state, (list, tuple)) and state and all(np.isscalar(item) for item in state)
    ):
        states = [state]
    elif isinstance(state, (list, tuple)) and state:
        states = list(state)
    else:
        raise ValueError("CUDA RNG state must be one state or a non-empty sequence of device states")
    device_count = torch.cuda.device_count()
    if len(states) != device_count:
        raise ValueError(
            f"CUDA RNG state has {len(states)} device state(s), but this runtime has {device_count}; "
            "exact restoration is unsafe"
        )
    return [
        normalize_rng_state(saved, f"CUDA device {index}", torch.cuda.get_rng_state(index).numel())
        for index, saved in enumerate(states)
    ]


def _restore_rng_states(rng: Mapping[str, Any], generator: torch.Generator | None, require_cuda: bool) -> None:
    try:
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint Python/NumPy RNG state is invalid") from exc

    cpu_state = normalize_rng_state(rng.get("torch_cpu"), "torch CPU", torch.get_rng_state().numel())
    cuda_saved = rng.get("torch_cuda")
    if cuda_saved is None:
        if require_cuda:
            raise ValueError("checkpoint lacks CUDA RNG state required for exact CUDA resume")
        cuda_states = None
    else:
        cuda_states = _normalize_cuda_rng_states(cuda_saved)
    if generator is None:
        raise ValueError("resume DataLoader must expose the checkpointed generator")
    generator_state = normalize_rng_state(
        rng.get("dataloader_generator"), "DataLoader generator", generator.get_state().numel()
    )

    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        if len(cuda_states) == 1:
            torch.cuda.set_rng_state(cuda_states[0], 0)
        else:
            torch.cuda.set_rng_state_all(cuda_states)
    generator.set_state(generator_state)


@dataclass
class TrainConfig:
    """Training configuration parameters."""
    warmup_epochs: int = 3
    max_epochs: int = 30
    min_executed_epochs: int = 11
    head_warmup_lr: float = 3.0e-4
    backbone_lr: float = 1.0e-5
    head_lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    patience: int = 8
    amp: bool = True
    amp_init_scale: float = 32768.0
    selection_criteria: list[str] = field(default_factory=lambda: ["mcc", "balanced_accuracy", "nll", "epoch"])


class Trainer:
    """Executes two-phase training loop with comprehensive per-epoch metric logging."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config: TrainConfig,
        device: torch.device | str = "cuda",
        output_dir: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = GradScaler("cuda", init_scale=self.config.amp_init_scale, enabled=self.config.amp and self.device.type == "cuda")
        self.history: list[dict[str, Any]] = []

    def _setup_phase_optimizer(self, phase: int) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
        if not hasattr(self.model, "classifier_named_parameters"):
            raise TypeError("Model must explicitly expose classifier_named_parameters()")
        classifier = dict(self.model.classifier_named_parameters())
        if not classifier:
            raise ValueError("Model exposed no classifier parameters")

        if phase == 1:
            for name, p in self.model.named_parameters():
                p.requires_grad = name in classifier

            head_params = list(classifier.values())
            optimizer = torch.optim.AdamW(head_params, lr=self.config.head_warmup_lr, weight_decay=self.config.weight_decay)
            scheduler = None
            return optimizer, scheduler

        # Phase 2: Full fine-tuning
        for p in self.model.parameters():
            p.requires_grad = True

        head_ids = {id(p) for p in classifier.values()}
        head_params = list(classifier.values())
        backbone_params = [p for p in self.model.parameters() if id(p) not in head_ids]

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.config.backbone_lr},
                {"params": head_params, "lr": self.config.head_lr},
            ],
            weight_decay=self.config.weight_decay,
        )

        phase2_epochs = max(1, self.config.max_epochs - self.config.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase2_epochs, eta_min=self.config.backbone_lr * 0.1
        )
        return optimizer, scheduler

    def _is_better_checkpoint(self, current: dict[str, Any], best: dict[str, Any] | None) -> bool:
        if best is None:
            return True

        for criterion in self.config.selection_criteria:
            if criterion == "mcc":
                diff = current["val_mcc"] - best["val_mcc"]
                if diff != 0:
                    return diff > 0
            elif criterion == "balanced_accuracy":
                diff = current["val_balanced_accuracy"] - best["val_balanced_accuracy"]
                if diff != 0:
                    return diff > 0
            elif criterion in ("loss", "nll"):
                metric = "val_nll" if criterion == "nll" else "val_loss"
                diff = best[metric] - current[metric]
                if diff != 0:
                    return diff > 0
            elif criterion == "epoch":
                return False  # Earlier epoch tie-break

        return False

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        is_warmup: bool,
    ) -> dict[str, float]:
        if is_warmup:
            if not hasattr(self.model, "set_warmup_mode"):
                raise TypeError("Model must explicitly expose set_warmup_mode()")
            self.model.set_warmup_mode()
        else:
            self.model.train()
        total_loss = 0.0
        y_true_all: list[int] = []
        y_prob_all: list[float] = []

        optimizer.zero_grad()
        num_batches = len(loader)

        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.config.amp and self.device.type == "cuda"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss_scaled = loss / self.config.grad_accum_steps

            if self.scaler.is_enabled():
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step + 1) % self.config.grad_accum_steps == 0 or (step + 1) == num_batches:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits.float(), dim=-1)[:, 1].detach().cpu().numpy()
            y_prob_all.extend(probs)
            y_true_all.extend(labels.cpu().numpy())

        epoch_loss = total_loss / len(y_true_all) if y_true_all else 0.0
        metrics = compute_classification_metrics(y_true_all, y_prob_all)
        metrics["loss"] = float(epoch_loss)
        return metrics

    def validate_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        y_true_all: list[int] = []
        y_prob_all: list[float] = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.config.amp and self.device.type == "cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item() * len(labels)
                probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
                y_prob_all.extend(probs)
                y_true_all.extend(labels.cpu().numpy())

        epoch_loss = total_loss / len(y_true_all) if y_true_all else 0.0
        metrics = compute_classification_metrics(y_true_all, y_prob_all)
        metrics["loss"] = float(epoch_loss)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_name: str = "best_model.pt",
        fit_metadata: Mapping[str, Any] | None = None,
        resume_checkpoint: str | Path | None = None,
    ) -> dict[str, Any]:
        metadata = dict(fit_metadata or {})
        self.history = []
        best_metrics: dict[str, Any] | None = None
        best_epoch = -1
        best_state = None
        patience_counter = 0
        start_epoch = 1
        training_finished = False
        resume_provenance = None

        # Phase 1: Warmup
        optimizer, scheduler = self._setup_phase_optimizer(phase=1)
        current_phase = 1

        if resume_checkpoint is not None:
            payload = load_resume_checkpoint(resume_checkpoint, metadata)
            resume_provenance = payload.get("resume_provenance")
            compatibility = payload.get("_resume_compatibility")
            if compatibility is not None:
                resume_provenance = {
                    "original_lineage": {
                        "metadata": dict(payload["metadata"]),
                        "completed_through_epoch": int(payload["completed_epoch"]),
                    },
                    "resume_from_checkpoint_sha256": compatibility["checkpoint_sha256"],
                    "compatibility_decision": dict(compatibility),
                    "current_implementation_fingerprint": metadata["code_fingerprint"],
                    "scientific_protocol_fingerprint": metadata["scientific_protocol_fingerprint"],
                }
            current_phase = 1 if payload["current_phase"] == "warmup" else 2
            optimizer, scheduler = self._setup_phase_optimizer(phase=current_phase)
            self.model.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            if scheduler is None:
                if payload["scheduler_state_dict"] is not None:
                    raise ValueError("resume checkpoint has scheduler state during warmup")
            else:
                if payload["scheduler_state_dict"] is None:
                    raise ValueError("resume checkpoint lacks full-finetuning scheduler state")
                scheduler.load_state_dict(payload["scheduler_state_dict"])
            self.scaler.load_state_dict(payload["scaler_state_dict"])
            self.history = list(payload["history"])
            best_epoch = int(payload["best_epoch"])
            best_metrics = dict(payload["best_metrics"])
            best_state = payload["best_model_state_dict"]
            patience_counter = int(payload["early_stopping_state"]["patience_counter"])
            start_epoch = int(payload["next_epoch"])
            training_finished = bool(payload.get("training_finished", False))

            rng = {**payload["rng_state"], "dataloader_generator": payload["dataloader_generator_state"]}
            _restore_rng_states(rng, train_loader.generator, require_cuda=self.device.type == "cuda")

            epochs = [int(row["epoch"]) for row in self.history]
            if epochs != list(range(1, int(payload["completed_epoch"]) + 1)):
                raise ValueError("resume history must contain each completed epoch exactly once")

        for epoch in (() if training_finished else range(start_epoch, self.config.max_epochs + 1)):
            required_phase = 1 if epoch <= self.config.warmup_epochs else 2
            if current_phase != required_phase:
                # Switch to Phase 2: Full fine-tuning
                current_phase = required_phase
                optimizer, scheduler = self._setup_phase_optimizer(phase=current_phase)

            is_warmup = (current_phase == 1)
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            train_m = self.train_epoch(train_loader, optimizer, is_warmup=is_warmup)
            val_m = self.validate_epoch(val_loader)

            if scheduler is not None:
                scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "phase": current_phase,
                "train_loss": train_m["loss"],
                "train_accuracy": train_m["accuracy"],
                "train_balanced_accuracy": train_m["balanced_accuracy"],
                "train_sensitivity": train_m["sensitivity"],
                "train_specificity": train_m["specificity"],
                "train_precision": train_m["precision"],
                "train_f1": train_m["f1"],
                "train_mcc": train_m["mcc"],
                "val_loss": val_m["loss"],
                "val_accuracy": val_m["accuracy"],
                "val_balanced_accuracy": val_m["balanced_accuracy"],
                "val_sensitivity": val_m["sensitivity"],
                "val_specificity": val_m["specificity"],
                "val_precision": val_m["precision"],
                "val_f1": val_m["f1"],
                "val_mcc": val_m["mcc"],
                "val_roc_auc": val_m["roc_auc"],
                "val_pr_auc": val_m["pr_auc"],
                "val_nll": val_m["nll"],
            }
            self.history.append(epoch_record)

            # Checkpoint evaluation
            if self._is_better_checkpoint(epoch_record, best_metrics):
                best_metrics = epoch_record.copy()
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping check
            should_stop = epoch >= self.config.min_executed_epochs and patience_counter >= self.config.patience
            next_epoch = epoch + 1
            training_finished = should_stop or next_epoch > self.config.max_epochs

            # At the warmup boundary, checkpoint the freshly initialized phase-2
            # optimizer: warmup optimizer state is intentionally not carried across phases.
            if not training_finished and current_phase == 1 and next_epoch > self.config.warmup_epochs:
                current_phase = 2
                optimizer, scheduler = self._setup_phase_optimizer(phase=2)

            if self.output_dir:
                atomic_torch_save(self.output_dir / LAST_CHECKPOINT_FILE, {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "scaler_state_dict": self.scaler.state_dict(),
                    "completed_epoch": epoch,
                    "next_epoch": next_epoch,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                    "best_model_state_dict": best_state,
                    "early_stopping_state": {"patience_counter": patience_counter},
                    "training_finished": training_finished,
                    "history": self.history,
                    "metadata": metadata,
                    "current_phase": "warmup" if current_phase == 1 else "full_finetuning",
                    "rng_state": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch_cpu": torch.get_rng_state().cpu().to(torch.uint8).contiguous(),
                        "torch_cuda": (
                            [state.cpu().to(torch.uint8).contiguous() for state in torch.cuda.get_rng_state_all()]
                            if torch.cuda.is_available() else None
                        ),
                    },
                    "dataloader_generator_state": (
                        train_loader.generator.get_state().cpu().to(torch.uint8).contiguous()
                        if train_loader.generator is not None else None
                    ),
                    **({"resume_provenance": resume_provenance} if resume_provenance is not None else {}),
                })

            if should_stop:
                break

        # Save best model checkpoint
        if self.output_dir and best_state is not None:
            ckpt_path = self.output_dir / checkpoint_name
            atomic_torch_save(ckpt_path, {
                "model_state_dict": best_state,
                "metadata": metadata,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
            })

            # Save history CSV and JSON
            history_csv = self.output_dir / "training_history.csv"
            atomic_write_csv(history_csv, self.history)

            history_json = self.output_dir / "training_history.json"
            atomic_write_json(history_json, self.history)
            atomic_write_json(self.output_dir / "fit_metadata.json", metadata)

        return {
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "history": self.history,
        }
