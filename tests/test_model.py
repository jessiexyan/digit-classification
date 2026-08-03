"""Tests for the Lightning model."""

import lightning as L
import pytest
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from digit_classification.model import DEFAULT_CLASS_WEIGHTS, DigitClassifier


def test_forward_returns_three_logits_per_image() -> None:
    model = DigitClassifier()

    logits = model(torch.randn(4, 1, 28, 28))

    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()


def test_predict_step_returns_probabilities() -> None:
    model = DigitClassifier()
    images = torch.randn(4, 1, 28, 28)

    probabilities = model.predict_step(images)

    assert probabilities.shape == (4, 3)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4))


def test_predict_step_accepts_a_labeled_batch() -> None:
    model = DigitClassifier()
    batch = (torch.randn(2, 1, 28, 28), torch.tensor([0, 2]))

    probabilities = model.predict_step(batch)

    assert probabilities.shape == (2, 3)


def test_class_weights_emphasize_the_minority_class() -> None:
    model = DigitClassifier()

    assert tuple(model.class_weights.tolist()) == pytest.approx(DEFAULT_CLASS_WEIGHTS)
    assert model.class_weights[1] > model.class_weights[0]
    assert model.class_weights[1] > model.class_weights[2]


def test_configure_optimizers_returns_adam() -> None:
    model = DigitClassifier(learning_rate=0.002)

    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.002)


def test_lightning_can_train_and_validate_one_batch() -> None:
    torch.manual_seed(7)
    images = torch.randn(12, 1, 28, 28)
    targets = torch.tensor([0, 1, 2] * 4)
    loader = DataLoader(TensorDataset(images, targets), batch_size=6)
    model = DigitClassifier()
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
    )

    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

    assert "train_loss" in trainer.callback_metrics
    assert "val_loss" in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
    assert torch.isfinite(trainer.callback_metrics["val_loss"])
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, model.parameters(), strict=True)
    )
