# Model Error Analysis

## Overall Performance

The best ResNet18 model achieved a test accuracy of **0.9799**, a macro-F1 score of **0.9798**, and a weighted-F1 score of **0.9798**.

The macro-F1 and weighted-F1 scores are almost identical because the four test classes contain a similar number of samples. Therefore, the evaluation is not strongly affected by class imbalance.

## Class-wise Performance

* Abyssinian: Precision = 0.9600, Recall = 0.9796, F1 = 0.9697
* Persian: Precision = 0.9896, Recall = 0.9500, F1 = 0.9694
* Beagle: Precision = 0.9804, Recall = 1.0000, F1 = 0.9901
* Pug: Precision = 0.9900, Recall = 0.9900, F1 = 0.9900

Persian had the lowest recall, which means that some true Persian images were classified as other breeds. Abyssinian had the lowest precision, indicating that some images from other classes were incorrectly predicted as Abyssinian.

## Confusion Analysis

Persian and Abyssinian were the most frequently confused classes.

According to the confusion matrix:

* Four Persian images were predicted as Abyssinian.
* One Abyssinian image was predicted as Persian.
* One Abyssinian image was predicted as Pug.
* One Persian image was predicted as Beagle.
* One Pug image was predicted as Beagle.

Among the five displayed incorrect samples, the following errors were observed:

1. Abyssinian predicted as Pug.
2. Abyssinian predicted as Persian.
3. Persian predicted as Abyssinian.
4. Persian predicted as Beagle.
5. Persian predicted as Abyssinian.

The frequent confusion between Persian and Abyssinian may be related to visual similarities in fur colour, facial appearance, pose, image cropping, or background information. The displayed images should be inspected more closely before identifying the main cause.

## Overfitting

There is no clear evidence of severe overfitting.

Training loss generally decreased, while validation loss also decreased throughout training. Validation accuracy reached 0.9875 and remained stable. The validation performance did not deteriorate while the training performance improved.

Validation accuracy was sometimes higher than training accuracy. This is reasonable because random cropping and horizontal flipping were applied only to the training images, making the training task more difficult.

## Effect of Data Augmentation

The training dataset used random resized cropping and horizontal flipping. These transformations likely helped the model generalize to different positions, crops, and horizontal orientations.

However, the effectiveness of data augmentation cannot be proven from only one training run. A controlled comparison should train the same model with and without augmentation while keeping the data split, learning rate, and number of epochs unchanged.

## Fine-tuning More Layers

The current model already achieved high validation and test performance while training only the final classification layer.

Therefore, it is not currently necessary to unfreeze more ResNet18 layers. Unfreezing additional layers would increase training time and could increase the risk of overfitting.

Further fine-tuning may only be useful if future experiments use more difficult classes, more varied images, or a larger dataset.

## Class Imbalance

The test classes are almost balanced, with 98 Abyssinian images and 100 images for each of the other classes.

Therefore, accuracy is not strongly dominated by one class in this experiment. In an imbalanced dataset, accuracy and weighted-F1 could remain high even if the model performed poorly on minority classes. Macro-F1 would be more useful because it gives equal importance to every class.

## Possible Improvements

Possible future improvements include:

* Inspecting all eight incorrect test images.
* Comparing training with and without data augmentation.
* Testing additional augmentation methods such as small rotations or colour jitter.
* Using confidence scores to identify uncertain predictions.
* Fine-tuning the final ResNet18 block with a smaller learning rate.
* Evaluating the model on more visually similar animal breeds.
