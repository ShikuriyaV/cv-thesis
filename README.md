# Computer Vision Thesis Preparation

A seven-day practical preparation project for computer vision and deep learning.

## Day 1 — Environment and Baseline

### Purpose

Set up a reproducible PyTorch development environment and verify the complete training workflow.

### Results

- PyTorch with CUDA successfully configured
- NVIDIA RTX 4070 GPU verified
- FashionMNIST baseline training completed
- Validation accuracy: **82.65%**
- Test accuracy: **81.64%**
- Git and GitHub repository established

### Files

- `environment_test.py`
- `practice_fashion_mnist.py`

The baseline script will be extended into a modular CNN training pipeline.

## Day 2 — CNN Training Pipeline

Implemented a basic CIFAR-10 image-classification pipeline with PyTorch.

* Built a simple CNN for image feature extraction and classification
* Created training, validation, and test DataLoaders
* Implemented training and validation loops
* Saved the model with the best validation accuracy
* Evaluated the saved model on the test set

**Results**

* Best validation accuracy: 64.92%
* Test accuracy: 64.96%

Main files:

```text
practice/
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
└── best_model.pth
```

Detailed notes are available in `notes/training_pipeline.md`.
