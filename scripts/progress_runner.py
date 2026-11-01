"""Run one existing training script with console-only progress instrumentation."""

from __future__ import annotations

import runpy
import sys
from datetime import timedelta
from pathlib import Path
from time import monotonic
from typing import Any

from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.training.trainer import Trainer


_ORIGINAL_FIT = Trainer.fit
_ORIGINAL_TRAIN_EPOCH = Trainer.train_epoch
_ORIGINAL_VALIDATE_EPOCH = Trainer.validate_epoch
_ORIGINAL_IS_BETTER = Trainer._is_better_checkpoint
_STATE: dict[str, Any] = {}


class _ProgressLoader:
    def __init__(self, loader: Any, description: str) -> None:
        self.loader = loader
        self.description = description
        self.bar: tqdm | None = None

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)

    def __iter__(self):
        with tqdm(total=len(self), desc=self.description, unit="batch", dynamic_ncols=True, leave=False) as bar:
            self.bar = bar
            try:
                for batch in self.loader:
                    yield batch
                    bar.update(1)
            finally:
                self.bar = None

    def set_running_loss(self, value: float) -> None:
        if self.bar is not None:
            self.bar.set_postfix(running_loss=f"{value:.4f}")


def _run_epoch(original, description: str, trainer: Trainer, loader: Any, *args: Any, **kwargs: Any):
    progress = _ProgressLoader(loader, description)
    loss_sum = 0.0
    sample_count = 0

    def record_loss(_module, inputs, output) -> None:
        nonlocal loss_sum, sample_count
        batch_size = len(inputs[1])
        loss_sum += output.detach().item() * batch_size
        sample_count += batch_size
        progress.set_running_loss(loss_sum / sample_count)

    handle = trainer.criterion.register_forward_hook(record_loss)
    try:
        return original(trainer, progress, *args, **kwargs)
    finally:
        handle.remove()


def _train_epoch(trainer: Trainer, loader: Any, *args: Any, **kwargs: Any):
    epoch = int(getattr(loader.dataset, "epoch", len(trainer.history) + 1))
    _STATE.update(epoch=epoch, epoch_started=monotonic())
    is_warmup = kwargs.get("is_warmup", args[1] if len(args) > 1 else False)
    print(
        f"\nEpoch {epoch}/{trainer.config.max_epochs} | "
        f"phase={'warmup' if is_warmup else 'full fine-tuning'}",
        flush=True,
    )
    return _run_epoch(_ORIGINAL_TRAIN_EPOCH, "Training", trainer, loader, *args, **kwargs)


def _validate_epoch(trainer: Trainer, loader: Any, *args: Any, **kwargs: Any):
    return _run_epoch(_ORIGINAL_VALIDATE_EPOCH, "Validation", trainer, loader, *args, **kwargs)


def _is_better_checkpoint(trainer: Trainer, current: dict[str, Any], best: dict[str, Any] | None) -> bool:
    improved = _ORIGINAL_IS_BETTER(trainer, current, best)
    _STATE["completed_epochs"] += 1
    elapsed = monotonic() - _STATE["epoch_started"]
    average = (monotonic() - _STATE["fit_started"]) / _STATE["completed_epochs"]
    remaining = average * max(0, trainer.config.max_epochs - current["epoch"])
    best_epoch = current["epoch"] if improved else best["epoch"]
    print(
        f"Epoch {current['epoch']}/{trainer.config.max_epochs} complete | elapsed epoch time "
        f"{timedelta(seconds=round(elapsed))} | estimated time remaining "
        f"{timedelta(seconds=round(remaining))}\n"
        f"  Train: loss={current['train_loss']:.4f} | accuracy={current['train_accuracy']:.4f} | "
        f"balanced_accuracy={current['train_balanced_accuracy']:.4f} | MCC={current['train_mcc']:.4f}\n"
        f"  Validation: loss={current['val_loss']:.4f} | NLL={current['val_nll']:.4f} | "
        f"accuracy={current['val_accuracy']:.4f} | balanced_accuracy={current['val_balanced_accuracy']:.4f} | "
        f"MCC={current['val_mcc']:.4f}\n"
        f"  Best epoch so far: {best_epoch} | improved checkpoint ranking: {'YES' if improved else 'NO'}",
        flush=True,
    )
    return improved


def _fit(trainer: Trainer, *args: Any, **kwargs: Any):
    _STATE.update(fit_started=monotonic(), completed_epochs=0)
    result = _ORIGINAL_FIT(trainer, *args, **kwargs)
    _STATE["result"] = result
    return result


def install_progress_logging() -> None:
    Trainer.fit = _fit
    Trainer.train_epoch = _train_epoch
    Trainer.validate_epoch = _validate_epoch
    Trainer._is_better_checkpoint = _is_better_checkpoint


def remove_progress_logging() -> None:
    Trainer.fit = _ORIGINAL_FIT
    Trainer.train_epoch = _ORIGINAL_TRAIN_EPOCH
    Trainer.validate_epoch = _ORIGINAL_VALIDATE_EPOCH
    Trainer._is_better_checkpoint = _ORIGINAL_IS_BETTER


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: progress_runner.py TRAINING_SCRIPT [arguments ...]")
    target = Path(sys.argv[1]).resolve()
    sys.argv = [str(target), *sys.argv[2:]]
    _STATE.clear()
    _STATE["job_started"] = monotonic()
    install_progress_logging()
    runpy.run_path(str(target), run_name="__main__")

    result = _STATE.get("result")
    if result and result.get("best_metrics"):
        best = result["best_metrics"]
        print(
            "\n" + "=" * 80 + "\nFOLD COMPLETE\n"
            f"Best epoch: {result['best_epoch']} | Best MCC: {best['val_mcc']:.4f} | "
            f"Best balanced accuracy: {best['val_balanced_accuracy']:.4f} | "
            f"Total fold time: {timedelta(seconds=round(monotonic() - _STATE['job_started']))}\n" + "=" * 80,
            flush=True,
        )


if __name__ == "__main__":
    main()
