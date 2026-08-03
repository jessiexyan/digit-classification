"""Tests for classification evaluation."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from digit_classification.evaluation import (
    collect_predictions,
    evaluate_model,
    format_classification_report,
)


class IdentityLogitModel(nn.Module):
    """Treat each synthetic input row as already-computed class logits."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def prediction_loader() -> DataLoader:
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [3.0, 2.0, 1.0],
            [0.0, 4.0, 1.0],
            [0.0, 1.0, 3.0],
            [0.0, 1.0, 5.0],
            [1.0, 0.0, 4.0],
        ]
    )
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    return DataLoader(TensorDataset(logits, targets), batch_size=2)


def test_collect_predictions_returns_expected_class_indices() -> None:
    model = IdentityLogitModel()

    true_classes, predicted_classes = collect_predictions(
        model, prediction_loader()
    )

    assert true_classes == [0, 0, 1, 1, 2, 2]
    assert predicted_classes == [0, 0, 1, 2, 2, 2]


def test_collect_predictions_restores_training_mode() -> None:
    model = IdentityLogitModel()
    model.train()

    collect_predictions(model, prediction_loader())

    assert model.training is True


def test_report_uses_original_digit_names_and_macro_metrics() -> None:
    report = format_classification_report(
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 2, 2, 2],
    )

    assert "precision" in report
    assert "recall" in report
    assert "f1-score" in report
    assert "macro avg" in report
    assert "weighted avg" in report
    assert all(digit in report for digit in ("0", "5", "8"))


def test_evaluate_model_combines_inference_and_reporting() -> None:
    report = evaluate_model(IdentityLogitModel(), prediction_loader())

    assert "accuracy" in report
    assert "0.8333" in report


def test_report_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        format_classification_report([0, 1], [0])


def test_empty_evaluation_is_rejected() -> None:
    empty_loader = DataLoader(
        TensorDataset(torch.empty(0, 3), torch.empty(0, dtype=torch.long))
    )

    with pytest.raises(ValueError, match="empty dataset"):
        collect_predictions(IdentityLogitModel(), empty_loader)
