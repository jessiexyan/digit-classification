"""Command-line interface for the digit classification application."""

import json
from pathlib import Path

import lightning as L
import torch
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from digit_classification.data import (
    CLASS_TO_LABEL,
    DEFAULT_SEED,
    DigitDataModule,
    download_mnist,
    inspect_mnist,
    load_image_tensor,
)
from digit_classification.evaluation import evaluate_model
from digit_classification.model import DigitClassifier
from digit_classification.progress import TrainingProgressCallback

app = typer.Typer(
    no_args_is_help=True,
    help="Train and use a classifier for the MNIST digits 0, 5, and 8.",
)


@app.command()
def download_data(
    data_dir: Path = typer.Option(..., "--data-dir", help="MNIST data directory."),
) -> None:
    """Download the MNIST training dataset."""
    download_mnist(data_dir)
    typer.echo(f"MNIST training data is available in {data_dir}.")


@app.command()
def train(
    data_dir: Path = typer.Option(..., "--data-dir", help="MNIST data directory."),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory for checkpoints and logs."
    ),
    epochs: int = typer.Option(20, "--epochs", min=1, max=20),
    batch_size: int = typer.Option(64, "--batch-size", min=1),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    resume_from: Path | None = typer.Option(
        None,
        "--resume-from",
        help="Lightning checkpoint from which to resume complete training state.",
    ),
) -> None:
    """Train the digit classifier on CPU."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "accelerator": "cpu",
        "batch_size": batch_size,
        "epochs": epochs,
        "resume_from": None if resume_from is None else str(resume_from),
        "seed": seed,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    L.seed_everything(seed, workers=True)
    data_module = DigitDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        seed=seed,
    )
    model = DigitClassifier()
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="digit-classifier-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        min_delta=1e-3,
    )
    progress_callback = TrainingProgressCallback(output_dir / "progress.json")
    trainer = L.Trainer(
        default_root_dir=output_dir,
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        deterministic=True,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            progress_callback,
        ],
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_from)
    typer.echo(f"Best checkpoint: {checkpoint_callback.best_model_path}")


@app.command()
def check_progress(
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Training output directory."
    ),
) -> None:
    """Print the latest machine-readable training progress snapshot."""
    progress_path = output_dir / "progress.json"
    if not progress_path.is_file():
        typer.echo(f"No training progress found at {progress_path}.", err=True)
        raise typer.Exit(code=1)
    typer.echo(progress_path.read_text(encoding="utf-8").rstrip())


@app.command()
def inspect_data(
    data_dir: Path = typer.Option(..., "--data-dir", help="MNIST data directory."),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
) -> None:
    """Print class counts, split sizes, overlap status, and a split fingerprint."""
    typer.echo(json.dumps(inspect_mnist(data_dir, seed=seed), indent=2, sort_keys=True))


@app.command()
def evaluate(
    checkpoint_path: Path = typer.Option(
        ..., "--checkpoint-path", help="Trained Lightning checkpoint."
    ),
    data_dir: Path = typer.Option(..., "--data-dir", help="MNIST data directory."),
    seed: int = typer.Option(
        DEFAULT_SEED,
        "--seed",
        help="Seed used to construct the training and evaluation splits.",
    ),
) -> None:
    """Evaluate a checkpoint on the held-out dataset."""
    model = DigitClassifier.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    data_module = DigitDataModule(data_dir=data_dir, seed=seed)
    data_module.prepare_data()
    data_module.setup(stage="test")
    report = evaluate_model(model, data_module.test_dataloader())
    typer.echo(report)


@app.command()
def predict(
    checkpoint_path: Path = typer.Option(
        ..., "--checkpoint-path", help="Trained Lightning checkpoint."
    ),
    input_path: Path = typer.Option(
        ..., "--input-path", help="Image containing a handwritten digit."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Print machine-readable JSON output."
    ),
) -> None:
    """Predict whether an image contains a 0, 5, or 8."""
    model = DigitClassifier.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    image = load_image_tensor(input_path).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        probabilities = model.predict_step(image)[0].cpu()

    predicted_class = int(probabilities.argmax())
    prediction = {
        "predicted_digit": CLASS_TO_LABEL[predicted_class],
        "probabilities": {
            str(CLASS_TO_LABEL[class_index]): round(probability, 6)
            for class_index, probability in enumerate(probabilities.tolist())
        },
    }
    if as_json:
        typer.echo(json.dumps(prediction, indent=2, sort_keys=True))
        return

    typer.echo(f"Predicted digit: {CLASS_TO_LABEL[predicted_class]}")
    typer.echo("Probabilities:")
    for class_index, probability in enumerate(probabilities.tolist()):
        typer.echo(f"  {CLASS_TO_LABEL[class_index]}: {probability:.4f}")


if __name__ == "__main__":
    app()
