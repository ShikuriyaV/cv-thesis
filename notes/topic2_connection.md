# Connection to Topic 2

## Current Video Processing Pipeline

This project implements a basic frame-level video classification pipeline:

```text
Video Input
→ OpenCV Frame Reading
→ Frame Sampling
→ Image Preprocessing
→ ResNet18 Classification
→ CSV Output
```

The video is opened using OpenCV. Its FPS, total frame count, width, height, and duration can be extracted. Frames are then read sequentially, and one frame is selected every fixed number of frames for inference.

Each selected frame is converted from OpenCV BGR format to RGB format, transformed using the same preprocessing pipeline as the test dataset, and passed into the trained ResNet18 model.

The prediction result contains:

* Timestamp
* Frame ID
* Predicted class
* Confidence score

The results are saved in a CSV file.

## Limitation

This demo performs frame-level classification and does not model temporal dependencies across consecutive frames.

The current model processes each selected frame independently. It does not use information from previous or subsequent frames.

Therefore, even though the input is a video, the model is still performing image classification rather than complete video understanding.

## Connection to Surgical Phase Recognition

Surgical phase recognition requires the model to understand how visual information changes over time.

A single frame may not contain enough information to distinguish between surgical phases because different phases may contain visually similar instruments, tissues, or operating-room scenes.

Temporal information from consecutive frames can help the model understand:

* What happened before the current frame
* How instruments and actions are changing
* Whether the procedure is moving from one phase to another
* Whether a prediction is temporally consistent

Future extensions may combine a CNN feature extractor with temporal models such as:

* CNN-LSTM
* Temporal Convolutional Network (TCN)
* Transformer

In these approaches, the CNN extracts spatial visual features from individual frames, while the temporal model learns relationships across a sequence of frames.

## Project Scope

This project uses a public pet image dataset rather than medical or surgical video data.

Its purpose is to demonstrate a basic practical foundation in:

* PyTorch
* Transfer learning
* Image classification
* Model evaluation
* OpenCV video processing
* Frame-level inference
* Result saving and analysis

The project does not claim to solve surgical phase recognition. It provides the basic image and video processing foundation required for further study of Topic 2.
