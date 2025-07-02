# E2E_model

This directory contains code and resources for end-to-end deep learning model experiments, including asymmetric and symmetric quantization, CNN architectures, and training scripts. The project is focused on image classification tasks using datasets like CIFAR-10 and MNIST.

## Directory Structure

- `Asymmetric.py` — Implements asymmetric quantization methods.
- `Symmetric.py` — Implements symmetric quantization methods.
- `CNNImage.py` — CNN model for image classification.
- `CNNToy.py` — Toy CNN model for experimentation.
- `CNNToyOptim.py` — Optimized version of the toy CNN.
- `Conv2dPt2e.py` — Custom 2D convolutional layer for quantization experiments.
- `PCQ.py`, `PGQ.py` — Quantization utilities and algorithms.
- `best_model.pth` — Pretrained model weights.
- `data/` — Contains datasets:
  - `cifar-10-python.tar.gz` — CIFAR-10 dataset archive.
  - `cifar-10-batches-py/` — Extracted CIFAR-10 batches.
  - `MNIST/` — MNIST dataset files.
- `README.md` — This file.

## Requirements

- Python 3.8+
- PyTorch
- NumPy

(Install dependencies using `pip install torch numpy`.)

## Usage

1. Place datasets in the `data/` directory as shown above.
2. Use the provided scripts to train or evaluate models. For example:
   ```bash
   python CNNToy.py
   ```
   or
   ```bash
   python CNNImage.py
   ```

3. Pretrained weights are available in `best_model.pth`.

## Notes

- Modify the scripts as needed for your experiments.
- See individual script files for more details on their usage and parameters.
