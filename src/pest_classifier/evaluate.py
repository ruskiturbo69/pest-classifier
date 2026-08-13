from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import load_datasets, make_loaders
from .model import load_model
from .utils import resolve_device, write_json


@torch.no_grad()
def evaluate_model(model, loader, device: torch.device, use_tta: bool) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        probabilities = torch.softmax(model(images), dim=1)
        if use_tta:
            probabilities = (
                probabilities
                + torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
                + torch.softmax(model(torch.flip(images, dims=[2])), dim=1)
            ) / 3.0
        y_true.extend(labels.tolist())
        y_pred.extend(probabilities.argmax(dim=1).cpu().tolist())
    return y_true, y_pred


def save_confusion_matrix(matrix: np.ndarray, classes: list[str], path: Path) -> None:
    figure_size = max(12, min(26, len(classes) // 4 + 8))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="IP102 confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an IP102 checkpoint.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", action="store_true", help="Average original, horizontal flip, and vertical flip.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    bundle = load_datasets(args.data_dir, image_size=args.image_size)
    split = bundle.val if args.split == "val" else bundle.test
    if split is None:
        raise FileNotFoundError("The requested test split does not exist.")
    _, val_loader, test_loader = make_loaders(
        bundle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    loader = val_loader if args.split == "val" else test_loader
    assert loader is not None
    model, _ = load_model(args.checkpoint, len(bundle.classes), device)
    y_true, y_pred = evaluate_model(model, loader, device, args.tta)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(bundle.classes))),
        target_names=bundle.classes,
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(bundle.classes))))
    save_confusion_matrix(matrix, bundle.classes, output_dir / "confusion_matrix.png")
    metrics = {
        "split": args.split,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "tta": args.tta,
        "samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classes": len(bundle.classes),
    }
    write_json(output_dir / "metrics.json", metrics)
    print(report)
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
