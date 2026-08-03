"""Tests for command-line orchestration."""

import json
from pathlib import Path

import torch
from typer.testing import CliRunner

from digit_classification import cli

runner = CliRunner()


def test_download_data_delegates_to_data_layer(monkeypatch, tmp_path: Path) -> None:
    received: list[Path] = []
    monkeypatch.setattr(cli, "download_mnist", received.append)

    result = runner.invoke(
        cli.app,
        ["download-data", "--data-dir", str(tmp_path / "data")],
    )

    assert result.exit_code == 0
    assert received == [tmp_path / "data"]
    assert "MNIST training data is available" in result.stdout


def test_train_rejects_more_than_twenty_epochs(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        [
            "train",
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--epochs",
            "21",
        ],
    )

    assert result.exit_code != 0
    assert "not in the range 1<=x<=20" in result.stderr


class FixedPredictionModel:
    def eval(self) -> "FixedPredictionModel":
        return self

    def predict_step(self, image: torch.Tensor) -> torch.Tensor:
        assert image.shape == (1, 1, 28, 28)
        return torch.tensor([[0.05, 0.15, 0.80]])


def test_predict_prints_digit_and_all_probabilities(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli.DigitClassifier,
        "load_from_checkpoint",
        lambda *args, **kwargs: FixedPredictionModel(),
    )
    monkeypatch.setattr(
        cli,
        "load_image_tensor",
        lambda path: torch.zeros(1, 28, 28),
    )

    result = runner.invoke(
        cli.app,
        [
            "predict",
            "--checkpoint-path",
            str(tmp_path / "model.ckpt"),
            "--input-path",
            str(tmp_path / "digit.png"),
        ],
    )

    assert result.exit_code == 0
    assert "Predicted digit: 8" in result.stdout
    assert "0: 0.0500" in result.stdout
    assert "5: 0.1500" in result.stdout
    assert "8: 0.8000" in result.stdout


def test_predict_can_print_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli.DigitClassifier,
        "load_from_checkpoint",
        lambda *args, **kwargs: FixedPredictionModel(),
    )
    monkeypatch.setattr(cli, "load_image_tensor", lambda path: torch.zeros(1, 28, 28))

    result = runner.invoke(
        cli.app,
        [
            "predict",
            "--checkpoint-path",
            str(tmp_path / "model.ckpt"),
            "--input-path",
            str(tmp_path / "digit.png"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["predicted_digit"] == 8
    assert payload["probabilities"] == {"0": 0.05, "5": 0.15, "8": 0.8}


def test_evaluate_loads_data_and_prints_report(monkeypatch, tmp_path: Path) -> None:
    model = object()
    monkeypatch.setattr(
        cli.DigitClassifier,
        "load_from_checkpoint",
        lambda *args, **kwargs: model,
    )

    class FakeDataModule:
        def __init__(self, data_dir: Path, seed: int) -> None:
            self.data_dir = data_dir
            assert seed == 9
            self.prepared = False

        def prepare_data(self) -> None:
            self.prepared = True

        def setup(self, stage: str) -> None:
            assert self.prepared
            assert stage == "test"

        def test_dataloader(self) -> str:
            return "evaluation-loader"

    monkeypatch.setattr(cli, "DigitDataModule", FakeDataModule)
    monkeypatch.setattr(
        cli,
        "evaluate_model",
        lambda received_model, loader: (
            "mock classification report"
            if received_model is model and loader == "evaluation-loader"
            else "wrong"
        ),
    )

    result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "--checkpoint-path",
            str(tmp_path / "model.ckpt"),
            "--data-dir",
            str(tmp_path / "data"),
            "--seed",
            "9",
        ],
    )

    assert result.exit_code == 0
    assert "mock classification report" in result.stdout


def test_train_configures_cpu_trainer_and_reports_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    class FakeCheckpoint:
        def __init__(self, **kwargs) -> None:
            calls["checkpoint_options"] = kwargs
            self.best_model_path = "/tmp/best.ckpt"

    class FakeEarlyStopping:
        def __init__(self, **kwargs) -> None:
            calls["early_stopping_options"] = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs) -> None:
            calls["trainer_options"] = kwargs

        def fit(self, model, datamodule, ckpt_path) -> None:
            calls["fit"] = (model, datamodule, ckpt_path)

    class FakeDataModule:
        def __init__(self, **kwargs) -> None:
            self.options = kwargs

    fake_model = object()
    monkeypatch.setattr(cli, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(cli, "EarlyStopping", FakeEarlyStopping)
    monkeypatch.setattr(cli, "TrainingProgressCallback", lambda path: ("progress", path))
    monkeypatch.setattr(cli.L, "Trainer", FakeTrainer)
    monkeypatch.setattr(cli.L, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "DigitDataModule", FakeDataModule)
    monkeypatch.setattr(cli, "DigitClassifier", lambda: fake_model)

    result = runner.invoke(
        cli.app,
        [
            "train",
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--epochs",
            "3",
            "--batch-size",
            "16",
            "--seed",
            "9",
            "--resume-from",
            str(tmp_path / "resume.ckpt"),
        ],
    )

    assert result.exit_code == 0
    assert calls["trainer_options"]["accelerator"] == "cpu"
    assert calls["trainer_options"]["max_epochs"] == 3
    assert len(calls["trainer_options"]["callbacks"]) == 3
    assert calls["early_stopping_options"] == {
        "monitor": "val_loss",
        "mode": "min",
        "patience": 3,
        "min_delta": 1e-3,
    }
    assert calls["fit"][0] is fake_model
    assert calls["fit"][1].options["batch_size"] == 16
    assert json.loads((tmp_path / "output" / "run_config.json").read_text()) == {
        "accelerator": "cpu",
        "batch_size": 16,
        "epochs": 3,
        "resume_from": str(tmp_path / "resume.ckpt"),
        "seed": 9,
    }
    assert calls["fit"][2] == tmp_path / "resume.ckpt"
    assert "Best checkpoint: /tmp/best.ckpt" in result.stdout


def test_check_progress_prints_snapshot(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "progress.json").write_text(
        '{"status": "running", "progress": 0.5}\n'
    )

    result = runner.invoke(
        cli.app,
        ["check-progress", "--output-dir", str(output_dir)],
    )

    assert result.exit_code == 0
    assert '"status": "running"' in result.stdout


def test_check_progress_reports_missing_snapshot(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["check-progress", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "No training progress found" in result.stderr


def test_inspect_data_prints_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "inspect_mnist",
        lambda data_dir, seed: {
            "fingerprint": "abc123",
            "has_overlap": False,
            "seed": seed,
        },
    )

    result = runner.invoke(
        cli.app,
        ["inspect-data", "--data-dir", str(tmp_path), "--seed", "9"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "fingerprint": "abc123",
        "has_overlap": False,
        "seed": 9,
    }
