"""Model evaluation and classification reporting."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from sklearn.metrics import classification_report
from torch import Tensor, nn

from digit_classification.data import CLASS_TO_LABEL

CLASS_INDICES = tuple(sorted(CLASS_TO_LABEL))
TARGET_NAMES = tuple(str(CLASS_TO_LABEL[index]) for index in CLASS_INDICES)


def collect_predictions(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
) -> tuple[list[int], list[int]]:
    """Run inference and return true and predicted contiguous class indices."""
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    was_training = model.training
    true_classes: list[int] = []
    predicted_classes: list[int] = []

    model.eval()
    try:
        with torch.inference_mode():
            for images, targets in batches:
                logits = model(images.to(device))
                predictions = logits.argmax(dim=1)
                true_classes.extend(targets.detach().cpu().tolist())
                predicted_classes.extend(predictions.detach().cpu().tolist())
    finally:
        model.train(was_training)

    if not true_classes:
        raise ValueError("Cannot evaluate an empty dataset.")
    return true_classes, predicted_classes


def format_classification_report(
    true_classes: list[int], predicted_classes: list[int]
) -> str:
    """Format precision, recall, F1, and support using original digit names."""
    if len(true_classes) != len(predicted_classes):
        raise ValueError("True and predicted classes must have the same length.")
    if not true_classes:
        raise ValueError("Cannot report metrics for an empty dataset.")

    return classification_report(
        true_classes,
        predicted_classes,
        labels=CLASS_INDICES,
        target_names=TARGET_NAMES,
        digits=4,
        zero_division=0,
    )


def evaluate_model(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
) -> str:
    """Evaluate a model and return a printable classification report."""
    true_classes, predicted_classes = collect_predictions(model, batches)
    return format_classification_report(true_classes, predicted_classes)
