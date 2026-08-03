"""Tests for dataset curation and splitting."""

from collections import Counter
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

import digit_classification.data as data_module
from digit_classification.data import (
    DigitDataModule,
    DatasetSplits,
    IndexedDigitDataset,
    curate_indices,
    label_counts,
    inspect_mnist,
    load_image_tensor,
    split_fingerprint,
    split_curated_indices,
)


def challenge_targets() -> torch.Tensor:
    """Create shuffled labels with enough examples for challenge curation."""
    return torch.tensor(([0] * 1_300) + ([5] * 400) + ([8] * 3_600) + ([2] * 20))


def test_curate_indices_selects_exact_required_counts() -> None:
    targets = challenge_targets()

    indices = curate_indices(targets, seed=17)

    assert len(indices) == 5_000
    assert label_counts(targets, indices) == Counter({8: 3_500, 0: 1_200, 5: 300})
    assert len(set(indices)) == len(indices)


def test_curation_is_reproducible_and_seeded() -> None:
    targets = challenge_targets()

    assert curate_indices(targets, seed=17) == curate_indices(targets, seed=17)
    assert curate_indices(targets, seed=17) != curate_indices(targets, seed=18)


def test_curation_rejects_insufficient_examples() -> None:
    with pytest.raises(ValueError, match=r"Digit 0 has 10 examples; 1200"):
        curate_indices([0] * 10)


def test_split_sizes_stratification_and_non_overlap() -> None:
    targets = challenge_targets()
    curated = curate_indices(targets, seed=17)

    splits = split_curated_indices(targets, curated, seed=17)

    assert isinstance(splits, DatasetSplits)
    assert (len(splits.train), len(splits.validation), len(splits.evaluation)) == (
        3_500,
        500,
        1_000,
    )
    assert label_counts(targets, splits.train) == Counter(
        {8: 2_450, 0: 840, 5: 210}
    )
    assert label_counts(targets, splits.validation) == Counter(
        {8: 350, 0: 120, 5: 30}
    )
    assert label_counts(targets, splits.evaluation) == Counter(
        {8: 700, 0: 240, 5: 60}
    )

    train = set(splits.train)
    validation = set(splits.validation)
    evaluation = set(splits.evaluation)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(evaluation)
    assert validation.isdisjoint(evaluation)
    assert train | validation | evaluation == set(curated)


def test_split_is_reproducible() -> None:
    targets = challenge_targets()
    curated = curate_indices(targets, seed=17)

    assert split_curated_indices(targets, curated, seed=17) == split_curated_indices(
        targets, curated, seed=17
    )


def test_split_fingerprint_is_stable_and_sensitive_to_indices() -> None:
    first = DatasetSplits(train=(1, 2), validation=(3,), evaluation=(4,))
    same = DatasetSplits(train=(1, 2), validation=(3,), evaluation=(4,))
    different = DatasetSplits(train=(2, 1), validation=(3,), evaluation=(4,))

    assert split_fingerprint(first) == split_fingerprint(same)
    assert split_fingerprint(first) != split_fingerprint(different)
    assert len(split_fingerprint(first)) == 64


@pytest.mark.parametrize(
    ("evaluation_fraction", "validation_fraction", "message"),
    [
        (-0.1, 0.1, "cannot be negative"),
        (0.8, 0.2, "must total less than 1"),
    ],
)
def test_split_rejects_invalid_fractions(
    evaluation_fraction: float,
    validation_fraction: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        split_curated_indices(
            [0],
            [0],
            evaluation_fraction=evaluation_fraction,
            validation_fraction=validation_fraction,
        )


class FakeDataset(Dataset):
    def __init__(self) -> None:
        self.examples = [
            (torch.zeros(1, 28, 28), 8),
            (torch.ones(1, 28, 28), 0),
            (torch.full((1, 28, 28), 2.0), 5),
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.examples[index]


def test_indexed_dataset_selects_and_remaps_labels() -> None:
    dataset = IndexedDigitDataset(FakeDataset(), indices=[2, 0])

    first_image, first_label = dataset[0]
    second_image, second_label = dataset[1]

    assert len(dataset) == 2
    assert torch.equal(first_image, torch.full((1, 28, 28), 2.0))
    assert first_label == 1
    assert second_label == 2


def test_load_image_tensor_inverts_white_background(tmp_path: Path) -> None:
    image_path = tmp_path / "digit.png"
    image = Image.new("L", (40, 20), color=255)
    for x in range(15, 25):
        for y in range(5, 15):
            image.putpixel((x, y), 0)
    image.save(image_path)

    tensor = load_image_tensor(image_path)

    assert tensor.shape == (1, 28, 28)
    assert tensor[:, 10:18, 10:18].mean() > tensor[:, 0:4, 0:4].mean()


class FakeMNIST(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        self.targets = challenge_targets()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(1, 28, 28), int(self.targets[index])


def test_data_module_builds_expected_loaders(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(data_module, "MNIST", FakeMNIST)
    module = DigitDataModule(tmp_path, batch_size=32, seed=17)

    module.setup()

    assert len(module.train_dataloader().dataset) == 3_500
    assert len(module.val_dataloader().dataset) == 500
    assert len(module.test_dataloader().dataset) == 1_000
    assert module.predict_dataloader().dataset is module.test_dataloader().dataset


def test_inspect_mnist_reports_counts_overlap_and_fingerprint(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(data_module, "MNIST", FakeMNIST)

    summary = inspect_mnist(tmp_path, seed=17)

    assert summary["sizes"] == {
        "curated": 5_000,
        "evaluation": 1_000,
        "train": 3_500,
        "validation": 500,
    }
    assert summary["counts"]["curated"] == {"0": 1_200, "5": 300, "8": 3_500}
    assert summary["has_overlap"] is False
    assert len(summary["fingerprint"]) == 64


def test_data_module_requires_setup_before_loader(tmp_path: Path) -> None:
    module = DigitDataModule(tmp_path)

    with pytest.raises(RuntimeError, match=r"Call setup\(\)"):
        module.train_dataloader()
