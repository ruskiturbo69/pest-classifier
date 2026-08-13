# Methodology

## Task

The project performs single-label image classification over the 102 categories of IP102. Each image is loaded from an `ImageFolder` split, where the parent directory is the class label.

## Model

The active model is EfficientNet-V2-S from torchvision. Its final classifier is replaced with a linear layer whose output size equals the number of classes discovered in the training split. The model can start from ImageNet weights or from an uninitialized backbone.

## Preprocessing

Training images use random resized crops, horizontal flips, and `TrivialAugmentWide`, followed by ImageNet normalization. Validation, test, and inference use deterministic resize, center crop, and the same normalization.

## Optimization

The default training configuration uses AdamW, cosine annealing, optional CUDA automatic mixed precision, a fixed random seed, and checkpoint selection by validation macro F1. Both the best and latest checkpoints are saved together with class names and configuration metadata.

## Evaluation

The primary metrics are accuracy and macro F1. Macro F1 is reported because IP102 has a long-tailed class distribution and accuracy alone can hide weak performance on minority classes. Optional test-time augmentation averages predictions for the original image, a horizontal flip, and a vertical flip.

## Limitations

- Results depend on the exact IP102 split and preprocessing configuration.
- The repository does not include the dataset or trained weights.
- A single global model may perform unevenly across the long-tailed classes.
- TTA is an inference-time technique, not a replacement for a clean held-out evaluation protocol.
