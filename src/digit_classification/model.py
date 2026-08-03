"""PyTorch Lightning model for classifying the digits 0, 5, and 8."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Inverse-frequency weights for the training split:
# digit 0: 840, digit 5: 210, digit 8: 2,450.
DEFAULT_CLASS_WEIGHTS = (1.3889, 5.5556, 0.4762)


class DigitClassifier(L.LightningModule):
    """A compact convolutional neural network for three-class MNIST prediction."""

    def __init__(
        self,
        *,
        learning_rate: float = 1e-3,
        dropout: float = 0.25,
        class_weights: tuple[float, float, float] = DEFAULT_CLASS_WEIGHTS,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return one unnormalized score (logit) per class for each image."""
        return self.classifier(self.features(images))

    def _shared_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        images, targets = batch
        logits = self(images)
        loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        return loss, accuracy

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Calculate and record loss and accuracy for one training batch."""
        loss, accuracy = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        """Calculate and record validation metrics without updating weights."""
        loss, accuracy = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True
        )

    def predict_step(
        self,
        batch: tuple[Tensor, Tensor] | Tensor,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """Return probabilities ordered by digit label 0, 5, and 8."""
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        return torch.softmax(self(images), dim=1)

    def configure_optimizers(self) -> Any:
        """Use Adam to optimize all trainable model parameters."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
