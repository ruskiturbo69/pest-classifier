# Local checkpoints

Trained weights are deliberately excluded from Git because they are large binary artifacts. Put a checkpoint such as `best_model.pth` in this directory locally, or pass an absolute/relative path to the CLI with `--checkpoint`.

The original local artifact `best_insect_model_v2.pth` is compatible with the new EfficientNet-V2-S loader as a raw 102-class state dict. It does not contain class-name metadata, so use the IP102 dataset directory with evaluation and prediction commands.
