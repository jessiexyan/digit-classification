"""Dataset curation, deterministic splitting, and loading."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
from PIL import Image, ImageOps, ImageStat
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

DIGITS = (0, 5, 8)
LABEL_TO_CLASS = {0: 0, 5: 1, 8: 2}
CLASS_TO_LABEL = {class_index: label for label, class_index in LABEL_TO_CLASS.items()}
SAMPLES_PER_DIGIT = {0: 1_200, 5: 300, 8: 3_500}
DEFAULT_SEED = 42

MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


@dataclass(frozen=True)
class DatasetSplits:
    """Original MNIST indices assigned to each non-overlapping split."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    evaluation: tuple[int, ...]


def _labels_as_ints(targets: Sequence[Any]) -> list[int]:
    """Convert a tensor or ordinary sequence of labels to Python integers."""
    return [int(label) for label in targets]


def curate_indices(
    targets: Sequence[Any],
    *,
    seed: int = DEFAULT_SEED,
    samples_per_digit: dict[int, int] | None = None,
) -> tuple[int, ...]:
    """Select the required number of examples for each digit reproducibly."""
    requested = samples_per_digit or SAMPLES_PER_DIGIT
    labels = _labels_as_ints(targets)
    selected: list[int] = []

    for digit in DIGITS:
        candidates = [index for index, label in enumerate(labels) if label == digit]
        count = requested[digit]
        if len(candidates) < count:
            raise ValueError(
                f"Digit {digit} has {len(candidates)} examples; {count} are required."
            )

        digit_rng = random.Random(seed + digit)
        selected.extend(digit_rng.sample(candidates, count))

    random.Random(seed).shuffle(selected)
    return tuple(selected)


def split_curated_indices(
    targets: Sequence[Any],
    curated_indices: Sequence[int],
    *,
    seed: int = DEFAULT_SEED,
    evaluation_fraction: float = 0.20,
    validation_fraction: float = 0.10,
) -> DatasetSplits:
    """Create deterministic stratified splits from curated MNIST indices.

    Fractions are measured against the complete curated dataset. With the
    challenge's class counts this produces 3,500 training, 500 validation, and
    1,000 evaluation examples.
    """
    if evaluation_fraction < 0 or validation_fraction < 0:
        raise ValueError("Split fractions cannot be negative.")
    if evaluation_fraction + validation_fraction >= 1:
        raise ValueError("Evaluation and validation fractions must total less than 1.")

    labels = _labels_as_ints(targets)
    by_digit: dict[int, list[int]] = {digit: [] for digit in DIGITS}
    for index in curated_indices:
        label = labels[index]
        if label not in by_digit:
            raise ValueError(f"Curated index {index} has unexpected label {label}.")
        by_digit[label].append(index)

    train: list[int] = []
    validation: list[int] = []
    evaluation: list[int] = []

    for digit, indices in by_digit.items():
        shuffled = list(indices)
        random.Random(seed + 100 + digit).shuffle(shuffled)
        evaluation_count = round(len(shuffled) * evaluation_fraction)
        validation_count = round(len(shuffled) * validation_fraction)

        evaluation.extend(shuffled[:evaluation_count])
        validation.extend(
            shuffled[evaluation_count : evaluation_count + validation_count]
        )
        train.extend(shuffled[evaluation_count + validation_count :])

    random.Random(seed + 200).shuffle(train)
    random.Random(seed + 201).shuffle(validation)
    random.Random(seed + 202).shuffle(evaluation)
    return DatasetSplits(tuple(train), tuple(validation), tuple(evaluation))


def label_counts(targets: Sequence[Any], indices: Sequence[int]) -> Counter[int]:
    """Count original digit labels at a collection of dataset indices."""
    labels = _labels_as_ints(targets)
    return Counter(labels[index] for index in indices)


class IndexedDigitDataset(Dataset):
    """View selected examples from a dataset and remap labels to 0, 1, and 2."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Any, int]:
        image, label = self.dataset[self.indices[item]]
        return image, LABEL_TO_CLASS[int(label)]


def download_mnist(data_dir: Path) -> None:
    """Download the MNIST training set into ``data_dir``."""
    MNIST(root=str(data_dir), train=True, download=True)


def load_image_tensor(input_path: Path) -> Tensor:
    """Load an external image and transform it to MNIST's tensor format."""
    with Image.open(input_path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        # MNIST uses bright digits on a dark background. Invert the common case
        # of a dark handwritten digit drawn on white paper.
        if ImageStat.Stat(image).mean[0] > 127:
            image = ImageOps.invert(image)
        image = ImageOps.pad(image, (28, 28), color=0)
        return MNIST_TRANSFORM(image)


class DigitDataModule(L.LightningDataModule):
    """Lightning data module for the reproducibly curated MNIST dataset."""

    def __init__(
        self,
        data_dir: Path,
        *,
        batch_size: int = 64,
        seed: int = DEFAULT_SEED,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.splits: DatasetSplits | None = None
        self.train_dataset: IndexedDigitDataset | None = None
        self.validation_dataset: IndexedDigitDataset | None = None
        self.evaluation_dataset: IndexedDigitDataset | None = None

    def prepare_data(self) -> None:
        download_mnist(self.data_dir)

    def setup(self, stage: str | None = None) -> None:
        dataset = MNIST(
            root=str(self.data_dir),
            train=True,
            transform=MNIST_TRANSFORM,
            download=False,
        )
        curated = curate_indices(dataset.targets, seed=self.seed)
        self.splits = split_curated_indices(dataset.targets, curated, seed=self.seed)
        self.train_dataset = IndexedDigitDataset(dataset, self.splits.train)
        self.validation_dataset = IndexedDigitDataset(
            dataset, self.splits.validation
        )
        self.evaluation_dataset = IndexedDigitDataset(
            dataset, self.splits.evaluation
        )

    def _loader(self, dataset: Dataset | None, *, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a data loader.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.validation_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.evaluation_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
