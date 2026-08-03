# Digit Classification

A reproducible PyTorch Lightning classifier for the MNIST digits `0`, `5`, and
`8`. The project deliberately uses an imbalanced 5,000-image subset and exposes
data download, training, evaluation, and inference through a Typer CLI.

## Dataset

Only the 60,000-image MNIST training dataset is used. Sampling is deterministic
and stratified by digit:

| Digit | Curated | Train (70%) | Validation (10%) | Evaluation (20%) |
|---:|---:|---:|---:|---:|
| 0 | 1,200 | 840 | 120 | 240 |
| 5 | 300 | 210 | 30 | 60 |
| 8 | 3,500 | 2,450 | 350 | 700 |
| **Total** | **5,000** | **3,500** | **500** | **1,000** |

The code retains original MNIST indices rather than copying image data. Given
the same seed, it recreates the same curated sample and non-overlapping splits.
Original labels are mapped to contiguous model classes as `0 -> 0`, `5 -> 1`,
and `8 -> 2`.

## Model and imbalance strategy

The model is a small CNN built from scratch: two convolution/ReLU/max-pooling
blocks followed by a 64-unit hidden layer, dropout, and three output logits. No
pretrained or prebuilt model is used.

Training uses Adam and inverse-frequency weighted cross-entropy. The weights
are based on training counts and give digit `5` more influence because it is
the minority class. This favors minority recall, with a possible precision
tradeoff. Overall accuracy is supplemented with per-class and macro metrics so
performance on the dominant digit `8` cannot hide weak minority performance.

## Installation

Python 3.12 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Confirm the CLI is installed:

```bash
digit-classification --help
```

## Usage

Download the MNIST training data:

```bash
digit-classification download-data --data-dir data
```

Train on CPU for 1 to 20 epochs:

```bash
digit-classification train \
  --data-dir data \
  --output-dir outputs \
  --epochs 20 \
  --batch-size 64 \
  --seed 42
```

The best validation-loss checkpoint and `last.ckpt` are written under
`outputs/checkpoints/`.

Evaluate the best checkpoint on the reproducible held-out split. Replace the
example filename with the path printed by `train`; do not type angle-bracket
placeholders literally in a shell.

```bash
digit-classification evaluate \
  --checkpoint-path 'outputs/checkpoints/digit-classifier-epoch=XX-val_loss=X.XXXX.ckpt' \
  --data-dir data
```

Predict an external image:

```bash
digit-classification predict \
  --checkpoint-path 'outputs/checkpoints/digit-classifier-epoch=XX-val_loss=X.XXXX.ckpt' \
  --input-path path/to/digit.png
```

Inference converts the image to grayscale, corrects EXIF orientation, preserves
aspect ratio while padding to 28x28, and uses the same MNIST normalization as
training. A predominantly white image is inverted because MNIST contains bright
digits on a dark background. The command prints the predicted digit and the
probabilities for all three classes.

## Example evaluation

A two-epoch CPU run with seed 42 produced the following held-out results. These
numbers demonstrate the workflow rather than establish a performance target.

| Metric | Digit 0 | Digit 5 | Digit 8 | Macro average |
|---|---:|---:|---:|---:|
| Precision | 0.9827 | 0.6744 | 0.9854 | 0.8808 |
| Recall | 0.9458 | 0.9667 | 0.9614 | 0.9580 |
| F1 | 0.9639 | 0.7945 | 0.9732 | 0.9106 |

Overall accuracy was `0.9580`. The high recall and lower precision for digit
`5` are consistent with the deliberate class weighting.

## Tests

Tests use synthetic tensors and labels so they do not download MNIST or perform
a full training job. They cover exact class counts, deterministic splitting,
absence of leakage, label mapping, image preprocessing, model shapes and
probabilities, Lightning training, reporting, and CLI orchestration.

```bash
pytest
pytest --cov=digit_classification --cov-report=term-missing
```

## Project structure

```text
digit-classification/
├── README.md
├── pyproject.toml
├── src/digit_classification/
│   ├── __init__.py
│   ├── cli.py
│   ├── data.py
│   ├── evaluation.py
│   └── model.py
└── tests/
    ├── test_cli.py
    ├── test_data.py
    ├── test_evaluation.py
    └── test_model.py
```
