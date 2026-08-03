"""Lightning callback for exposing training progress as JSON."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer
from torch import Tensor


class TrainingProgressCallback(Callback):
    """Write an atomic, machine-readable snapshot of training progress."""

    def __init__(self, output_path: Path) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.started_at: float | None = None

    def _metrics(self, trainer: Trainer) -> dict[str, float]:
        wanted = {"train_loss", "train_accuracy", "val_loss", "val_accuracy"}
        metrics: dict[str, float] = {}
        for name, value in trainer.callback_metrics.items():
            if name in wanted:
                metrics[name] = float(value.detach().cpu() if isinstance(value, Tensor) else value)
        return metrics

    def _write(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.output_path)

    def _snapshot(
        self,
        trainer: Trainer,
        *,
        status: str,
        completed_epochs: int,
        error: str | None = None,
    ) -> dict[str, Any]:
        elapsed = 0.0 if self.started_at is None else time.monotonic() - self.started_at
        remaining: float | None = None
        if status == "running" and completed_epochs > 0:
            seconds_per_epoch = elapsed / completed_epochs
            remaining = seconds_per_epoch * (trainer.max_epochs - completed_epochs)

        return {
            "completed_epochs": completed_epochs,
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
            "max_epochs": trainer.max_epochs,
            "metrics": self._metrics(trainer),
            "progress": round(completed_epochs / trainer.max_epochs, 4),
            "remaining_seconds": None if remaining is None else round(remaining, 2),
            "status": status,
        }

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.started_at = time.monotonic()
        self._write(self._snapshot(trainer, status="running", completed_epochs=0))

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        completed_epochs = trainer.current_epoch + 1
        self._write(
            self._snapshot(
                trainer,
                status="running",
                completed_epochs=completed_epochs,
            )
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        completed_epochs = min(trainer.current_epoch + 1, trainer.max_epochs)
        status = (
            "stopped_early"
            if trainer.should_stop and completed_epochs < trainer.max_epochs
            else "completed"
        )
        self._write(
            self._snapshot(
                trainer,
                status=status,
                completed_epochs=completed_epochs,
            )
        )

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        self._write(
            self._snapshot(
                trainer,
                status="failed",
                completed_epochs=trainer.current_epoch,
                error=f"{type(exception).__name__}: {exception}",
            )
        )
