# Multilayer Perceptron 🧠

A **multilayer perceptron implemented from scratch in NumPy** to classify breast tumours as **malignant (M)** or **benign (B)** from the Wisconsin Diagnostic Breast Cancer dataset.

Forward pass, backpropagation, gradient descent, softmax and cross-entropy are all written by hand. No deep learning framework is used.

> 42 Paris project — Data Science / AI branch.

---

## Table of contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Split the dataset](#1-split-the-dataset)
  - [2. Train](#2-train)
  - [3. Predict](#3-predict)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)

---

## Dataset

`data/raw/data.csv` — 569 samples, no header:

| Column | Content |
|---|---|
| 0 | Sample ID |
| 1 | Diagnosis: `M` (malignant) or `B` (benign) |
| 2–31 | 30 numeric features computed from a digitized image of a cell nucleus (radius, texture, perimeter, area, smoothness, … as mean, standard error and worst value) |

Labels are one-hot encoded: `B → [1, 0]`, `M → [0, 1]`.

---

## Installation

```bash
git clone https://github.com/dzolotuk/multilayer-perceptron.git
cd multilayer-perceptron
./setup.sh
source local_env/bin/activate
```

`setup.sh` creates a virtual environment `local_env` and installs `numpy`, `pandas` and `matplotlib`.

All scripts must be run **from the repository root**.

---

## Usage

### 1. Split the dataset

```bash
python3 split.py --input data/raw/data.csv
```

Shuffles the dataset and writes:

- `data/processed/train.csv` (80 %)
- `data/processed/val.csv` (20 %)

| Option | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to the raw CSV |
| `--test_size` | `0.2` | Fraction of samples sent to the validation file |

The shuffle seed is random, so each run produces a different split.

### 2. Train

```bash
python3 train.py --dataset data/processed/train.csv --plot --metrics
```

| Option | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Path to a labelled CSV |
| `--epochs` | `60` | Maximum number of epochs |
| `--lr` | `0.01` | Learning rate |
| `--layers` | `24 24` | Hidden layer sizes, e.g. `--layers 32 16 8` |
| `--model_path` | `saved_model.npy` | Where the trained model is saved |
| `--plot` | off | Show loss and accuracy curves at the end |
| `--metrics` | off | Also log precision, recall and F1 for each epoch |

The given dataset is split internally into 80 % train / 20 % validation (fixed seed `42`). Features are standardised with the mean and standard deviation **of the training part only**, and these statistics are saved with the model.

Example output:

```
x_train shape : (364, 30)
x_val shape : (91, 30)
epoch 000 - loss: ... - acc: ... - val_loss: ... - val_acc: ...
...
> saving model 'saved_model.npy' to disk...

=== FINAL RESULT ===
train_loss : ...
val_loss   : ...
train_acc  : ...
val_acc    : ...
```

### 3. Predict

```bash
python3 predict.py --model saved_model.npy --input data/processed/val.csv
```

| Option | Description |
|---|---|
| `--model` | Path to a model saved by `train.py` |
| `--input` | CSV to predict on (same format as the dataset, including the label column) |

Prints the predicted class and confidence for each sample, then the categorical cross-entropy loss and accuracy on the file:

```
Predictions:
0: B (confidence: 0.98...)
1: M (confidence: 0.99...)
...

Loss: ...
Accuracy: ...
```

---

## How it works

### Architecture

```
Input (30 features)
   │
   ├── Dense(24) + sigmoid
   ├── Dense(24) + sigmoid        ← hidden layers, configurable with --layers
   │
   └── Dense(2)  + softmax        → P(B), P(M)
```

- **Weights:** initialised as `randn · √(2 / n_inputs)`, biases at zero.
- **Activations:** sigmoid in hidden layers (input clipped to avoid overflow), numerically stable softmax on the output.
- **Loss:** categorical cross-entropy. Combined with softmax, its gradient with respect to the output logits is simply `y_pred − y_true`.

### Training

- **Full-batch gradient descent:** one forward and one backward pass over the whole training set per epoch.
- **Backpropagation:** each `DenseLayer` stores its input and pre-activation `z` during the forward pass, then computes the gradients of its weights, bias and input in `backward()`, updating its parameters in place.
- **Early stopping:** training stops when validation loss has not improved for 5 epochs in a row.

### Saved model

`saved_model.npy` is a pickled dictionary:

```python
{
    "hidden_layers": [...],          # hidden layer sizes
    "mean": ..., "std": ...,         # normalisation statistics from training
    "layers": [{"weights": ..., "bias": ...}, ...]
}
```

`predict.py` rebuilds the network from the weight shapes, so any `--layers` configuration can be loaded without extra parameters.

---

## Project structure

```
multilayer-perceptron/
├── setup.sh                 # venv + dependencies
├── split.py                 # raw CSV → train.csv / val.csv
├── train.py                 # training entry point
├── predict.py               # inference entry point
├── data/
│   ├── raw/data.csv
│   └── processed/           # train.csv, val.csv
└── src/
    ├── core/
    │   ├── activations.py   # sigmoid, ReLU, softmax (+ derivatives)
    │   ├── layer.py         # DenseLayer: forward / backward
    │   ├── loss.py          # binary & categorical cross-entropy
    │   └── network.py       # NeuralNetwork: stacks layers
    ├── data/
    │   ├── loader.py        # CSV → X, one-hot y
    │   ├── preprocessing.py
    │   └── split.py         # train/validation split
    ├── training/
    │   └── trainer.py       # training loop, early stopping, history
    └── utils/
        ├── metrics.py       # precision, recall, F1
        └── plot.py          # loss / accuracy curves
```

---

## Tech stack

Python 3 · NumPy · pandas · matplotlib

## Author

- [Dmytro Zolotukhin](https://github.com/dzolotuk) — 42 Paris
