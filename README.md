# Computer Vision Thesis Preparation

A seven-day practical preparation project for computer vision and deep learning.

## 1 — Environment and Baseline

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

## 2 — CNN Training Pipeline

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

## 3 — ResNet18 Transfer Learning

* Built a 4-class image classifier using a pretrained ResNet18.
* Used Oxford-IIIT Pet classes: Abyssinian, Persian, Beagle, and Pug.
* Froze the ResNet18 backbone and trained only the classification head.
* Applied data augmentation and ImageNet normalization.
* Saved the model with the best validation accuracy.
* Best validation accuracy: **98.75%**


## 4 — Model Evaluation and Error Analysis

* Added test evaluation with accuracy, macro-F1, weighted-F1, and class-wise metrics.
* Generated training curves, a confusion matrix, a classification report, and correct/incorrect prediction samples.
* Achieved **97.99% test accuracy** and **97.98% macro-F1** with the best ResNet18 model.
* Identified Persian and Abyssinian as the most frequently confused classes.
* Documented overfitting, data augmentation, class imbalance, and possible model improvements.

## 5 — Video Processing

The project includes a basic frame-level video inference pipeline using OpenCV.

The pipeline can:

* Read a video and extract its FPS, total frame count, resolution, and duration;
* Sample frames at a fixed interval;
* Apply the trained ResNet18 model to individual frames;
* Save timestamps, frame IDs, predicted classes, and confidence scores to CSV.

Run the demo with:

```bash
python -m project.make_demo_video
python -m project.video_info
python -m project.infer_video
```

The prediction results are saved to:

```text
results/video_predictions.csv
```

### Limitation

This demo performs frame-level classification and does not model temporal dependencies across consecutive frames.

The current model processes every sampled frame independently. Future work could use CNN-LSTM, TCN, or Transformer-based methods for temporal modelling.
