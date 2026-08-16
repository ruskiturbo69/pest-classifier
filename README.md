# Pest Classifier

PyTorch project for classifying agricultural insect pests from the [IP102 dataset](https://github.com/xpwu95/IP102). The repository contains a reproducible training, evaluation, and inference pipeline based on EfficientNet-V2-S.

The project is focused on the 102-class IP102 benchmark. The earlier 9-class Random Forest experiment is not part of the active pipeline.

## What is included

- EfficientNet-V2-S with a 102-class classification head,
- ImageNet-style preprocessing and training augmentation,
- optional ImageNet initialization,
- CUDA mixed precision (AMP),
- AdamW optimizer and cosine learning-rate schedule,
- validation metrics including accuracy and macro F1,
- checkpoint metadata containing class names and training configuration,
- standalone evaluation with optional test-time augmentation (TTA),
- top-k inference for one image or a directory,
- lightweight smoke tests and GitHub Actions CI.

The dataset and trained weights are intentionally not committed. IP102 contains more than 75,000 images in 102 categories and should be downloaded from its official source.

## Repository layout

```text
src/pest_classifier/
  data.py       # ImageFolder loading and transforms
  model.py      # EfficientNet-V2-S and checkpoint loading
  train.py      # training CLI
  evaluate.py   # validation/test metrics and confusion matrix
  predict.py    # single-image and folder inference
tests/          # smoke tests
docs/           # dataset and methodology notes
weights/        # local checkpoint instructions; binary weights are ignored
```

## Dataset layout

The code expects the standard `ImageFolder` structure:

```text
IP102/
  train/
    0/
    1/
    ...
    101/
  val/
    0/
    ...
    101/
  test/          # optional, used by evaluate.py --split test
    0/
    ...
    101/
```

The folder names are treated as class labels. The train and validation class mappings must match exactly.

## Installation

Python 3.10 or newer is recommended. Create an environment and install the project:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

For an NVIDIA GPU, install the matching PyTorch/torchvision build from the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing the remaining packages.

## Train

The `--data-dir` argument must point to the directory containing `train/` and `val/`:

```bash
python -m pest_classifier.train \
  --data-dir /path/to/IP102 \
  --output-dir artifacts \
  --epochs 30 \
  --batch-size 32 \
  --pretrained \
  --amp
```

On Windows, start with `--num-workers 0`. Increase it only after the pipeline works reliably on the target machine.

Training writes `best_model.pth`, `last_model.pth`, `config.json`, and `history.json` to the output directory. These files are local artifacts and are ignored by Git.

## Evaluate

Evaluate the validation split without TTA:

```bash
python -m pest_classifier.evaluate \
  --data-dir /path/to/IP102 \
  --checkpoint artifacts/best_model.pth \
  --split val \
  --output-dir evaluation
```

To reproduce the original evaluation idea with horizontal and vertical flips:

```bash
python -m pest_classifier.evaluate \
  --data-dir /path/to/IP102 \
  --checkpoint artifacts/best_model.pth \
  --split val \
  --tta \
  --output-dir evaluation_tta
```

The evaluation command saves `metrics.json`, `classification_report.txt`, and `confusion_matrix.png`.

## Predict

For a single image:

```bash
python -m pest_classifier.predict image.jpg \
  --checkpoint artifacts/best_model.pth \
  --data-dir /path/to/IP102 \
  --top-k 3
```

For a directory, all supported images below it are processed recursively.

The original `best_insect_model_v2.pth` checkpoint is a raw EfficientNet-V2-S state dict and can be loaded by the new tools with `--num-classes 102`. New checkpoints include metadata and do not need that flag.

## Reference results

The existing `best_insect_model_v2.pth` checkpoint was evaluated on the IP102 validation split using EfficientNet-V2-S and three-view test-time augmentation: original image, horizontal flip, and vertical flip.

| Metric | Result |
|---|---:|
| Accuracy | 71.31% |
| Macro precision | 66.32% |
| Macro recall | 63.03% |
| Macro F1 | 64.07% |
| Weighted precision | 70.73% |
| Weighted recall | 71.31% |
| Weighted F1 | 70.67% |

Evaluation set: 7,508 images across 102 classes. These are reference results from the existing checkpoint and prior evaluation notebook; reproduce or update them with `pest_classifier.evaluate` after changing the training or evaluation configuration.

## Tests

```bash
pytest -q
```

## Dataset and citation

Please cite the original IP102 paper and follow the dataset authors' terms when downloading or redistributing the data. See [docs/DATASET.md](docs/DATASET.md) for the citation and dataset notes.

## License

The code license still needs to be selected by the author. The IP102 dataset and any pretrained backbone weights have their own terms and are not automatically covered by the code license.
