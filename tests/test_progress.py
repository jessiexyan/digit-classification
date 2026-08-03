"""Tests for the training progress callback."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import digit_classification.progress as progress_module
from digit_classification.progress import TrainingProgressCallback


def fake_trainer(**overrides):
    values = {
        "callback_metrics": {"val_loss": torch.tensor(0.25)},
        "current_epoch": 0,
        "max_epochs": 4,
        "sanity_checking": False,
        "should_stop": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def read_snapshot(path: Path) -> dict:
    return json.loads(path.read_text())


def test_callback_reports_start_epoch_and_completion(
    monkeypatch, tmp_path: Path
) -> None:
    clock = iter([10.0, 10.0, 12.0, 14.0])
    monkeypatch.setattr(progress_module.time, "monotonic", lambda: next(clock))
    path = tmp_path / "progress.json"
    callback = TrainingProgressCallback(path)
    trainer = fake_trainer()

    callback.on_fit_start(trainer, None)
    assert read_snapshot(path)["status"] == "running"

    callback.on_validation_epoch_end(trainer, None)
    epoch_snapshot = read_snapshot(path)
    assert epoch_snapshot["completed_epochs"] == 1
    assert epoch_snapshot["progress"] == 0.25
    assert epoch_snapshot["remaining_seconds"] == 6.0
    assert epoch_snapshot["metrics"]["val_loss"] == 0.25

    trainer.current_epoch = 3
    callback.on_fit_end(trainer, None)
    final_snapshot = read_snapshot(path)
    assert final_snapshot["status"] == "completed"
    assert final_snapshot["progress"] == 1.0


def test_callback_records_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(progress_module.time, "monotonic", lambda: 10.0)
    path = tmp_path / "progress.json"
    callback = TrainingProgressCallback(path)
    trainer = fake_trainer(current_epoch=2)
    callback.on_fit_start(trainer, None)

    callback.on_exception(trainer, None, ValueError("bad batch"))

    snapshot = read_snapshot(path)
    assert snapshot["status"] == "failed"
    assert snapshot["error"] == "ValueError: bad batch"
