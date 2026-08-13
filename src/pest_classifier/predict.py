from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .data import build_eval_transform
from .model import load_model
from .utils import resolve_device


def _class_names(data_dir: Path | None, count: int) -> list[str]:
    if data_dir is None:
        return [str(index) for index in range(count)]
    from torchvision.datasets import ImageFolder

    classes = ImageFolder(data_dir / "train").classes
    if len(classes) != count:
        raise ValueError("The class mapping in data-dir does not match the checkpoint.")
    return classes


@torch.no_grad()
def predict_path(
    path: Path,
    model,
    transform,
    class_names: list[str],
    device: torch.device,
    top_k: int,
) -> list[dict[str, object]]:
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not files:
        raise FileNotFoundError(f"No supported images found under {path}")

    results = []
    model.eval()
    for image_path in files:
        image = Image.open(image_path).convert("RGB")
        logits = model(transform(image).unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probabilities, k=min(top_k, len(class_names)))
        results.append(
            {
                "path": str(image_path),
                "predictions": [
                    {"class": class_names[int(index)], "probability": float(value)}
                    for value, index in zip(values.cpu(), indices.cpu())
                ],
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run IP102 inference on an image or folder.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, help="Optional dataset root used to recover class names.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=102)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    num_classes = args.num_classes
    if args.data_dir is not None:
        from torchvision.datasets import ImageFolder

        num_classes = len(ImageFolder(args.data_dir / "train").classes)
    model, metadata = load_model(args.checkpoint, num_classes, device)
    classes = metadata.get("classes") if isinstance(metadata, dict) else None
    if not classes:
        classes = _class_names(args.data_dir, model.classifier[1].out_features)
    results = predict_path(
        args.path,
        model,
        build_eval_transform(args.image_size),
        list(classes),
        device,
        args.top_k,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
