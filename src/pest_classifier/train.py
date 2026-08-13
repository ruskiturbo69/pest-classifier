from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from .data import load_datasets, make_loaders
from .model import build_model
from .utils import amp_context, make_grad_scaler, resolve_device, seed_everything, write_json

log = logging.getLogger("pest_classifier.train")


def _run_train_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scaler,
    device: torch.device,
    amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)

    return total_loss / max(total_items, 1)


@torch.no_grad()
def _evaluate(model: nn.Module, loader, criterion, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_datasets(args.data_dir, image_size=args.image_size)
    if args.expected_classes and len(bundle.classes) != args.expected_classes:
        raise ValueError(
            f"Expected {args.expected_classes} classes, found {len(bundle.classes)}. "
            "Use --expected-classes 0 to disable this IP102 check."
        )

    train_loader, val_loader, _ = make_loaders(
        bundle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(len(bundle.classes), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_grad_scaler(device, args.amp)

    config = {
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "image_size": args.image_size,
        "num_classes": len(bundle.classes),
        "classes": bundle.classes,
        "model": "efficientnet_v2_s",
        "pretrained_backbone": args.pretrained,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "amp": bool(args.amp and device.type == "cuda"),
    }
    write_json(output_dir / "config.json", config)

    best_f1 = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.amp
        )
        val_metrics = _evaluate(model, val_loader, criterion, device)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        log.info(
            "epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_macro_f1=%.4f",
            epoch,
            args.epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": {**config, "epoch": epoch, "best_val_macro_f1": max(best_f1, val_metrics["macro_f1"])},
        }
        torch.save(checkpoint, output_dir / "last_model.pth")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(checkpoint, output_dir / "best_model.pth")
            log.info("saved new best checkpoint to %s", output_dir / "best_model.pth")

    write_json(output_dir / "history.json", history)
    return {"best_val_macro_f1": best_f1, "output_dir": str(output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EfficientNet-V2-S on IP102.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing train/ and val/.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 for the safest Windows setup.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device.")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights for the backbone.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision.")
    parser.add_argument("--expected-classes", type=int, default=102)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
