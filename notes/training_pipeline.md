# PyTorch Training Pipeline

## Objective

Build a basic image-classification pipeline with PyTorch using CIFAR-10.

The pipeline includes:

* Dataset and DataLoader
* CNN model
* Training loop
* Validation loop
* Training and validation loss
* Validation accuracy
* Best-model checkpoint saving
* Independent test evaluation

## Data Pipeline

CIFAR-10 contains RGB images with the shape:

```text
[C, H, W] = [3, 32, 32]
```

A batch of 64 images has the shape:

```text
[N, C, H, W] = [64, 3, 32, 32]
```

The original training data was divided into:

* Training set: 45,000 images
* Validation set: 5,000 images
* Test set: 10,000 images

`Dataset` defines how individual images and labels are accessed.

`DataLoader` groups samples into batches and provides iteration and shuffling.

## CNN Model

The model contains:

```text
Conv2d
ReLU
MaxPool2d
Conv2d
ReLU
MaxPool2d
Flatten
Linear
ReLU
Linear
```

The tensor shape changes approximately as follows:

```text
[64, 3, 32, 32]
→ [64, 16, 32, 32]
→ [64, 16, 16, 16]
→ [64, 32, 16, 16]
→ [64, 32, 8, 8]
→ [64, 2048]
→ [64, 128]
→ [64, 10]
```

The convolution layers extract visual features.

The pooling layers reduce the height and width of the feature maps.

The classification head converts the extracted features into ten class scores.

## Training Pipeline

The main training steps are:

```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

Their purposes are:

1. Clear gradients from the previous batch.
2. Perform the forward pass.
3. Compare predictions with true labels.
4. Calculate gradients through backpropagation.
5. Update the model parameters.

One epoch means that the model has processed the entire training set once.

## Validation Pipeline

Validation is performed after each training epoch.

```python
model.eval()

with torch.no_grad():
    outputs = model(images)
```

### Why use `model.eval()`?

`model.eval()` switches the model to evaluation mode.

Some layers, such as Dropout and Batch Normalization, behave differently during training and evaluation.

### Why use `torch.no_grad()`?

Validation does not update model parameters, so gradients are unnecessary.

Disabling gradient tracking reduces memory usage and unnecessary computation.

## Batch Size

Batch size is the number of samples processed before one parameter update.

For example:

```python
batch_size = 64
```

means that the model processes 64 images before calculating the loss and updating its parameters.

## Training, Validation and Test Sets

The training set is used to calculate gradients and update model parameters.

The validation set is used to monitor performance and select the best model.

The test set is used only after training to measure the final performance.

They must remain separate because evaluating the model on data used for parameter updates would not provide a reliable measure of generalization.

## Overfitting

Overfitting occurs when a model performs increasingly well on the training data but performs worse on unseen data.

A possible overfitting pattern is:

```text
Training loss decreases
Training accuracy increases
Validation loss increases
Validation accuracy decreases
```

Overfitting should be judged from the trend across multiple epochs, not from a single result.

## Best-Model Saving

The model is saved only when the validation accuracy improves:

```python
if validation_accuracy > best_val_accuracy:
    torch.save(model.state_dict(), model_path)
```

This ensures that the final checkpoint represents the best validation performance rather than simply the last training epoch.

## Results

The model was trained for five epochs.

```text
Best Validation Accuracy: 64.92%
Test Accuracy: 64.96%
Test Loss: 1.0008
```

The validation and test accuracy are very close, indicating stable performance on unseen data and no obvious overfitting in the current experiment.

## Project Files

```text
practice/
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
└── best_model.pth
```
