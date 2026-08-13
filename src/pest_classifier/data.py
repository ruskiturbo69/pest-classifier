from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DatasetBundle:
    train: datasets.ImageFolder
    val: datasets.ImageFolder
    test: datasets.ImageFolder | None

    @property
    def classes(self) -> list[str]:
        return list(self.train.classes)


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """Build the augmentation pipeline used during training."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Build a deterministic transform for validation, testing, and inference."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _load_split(path: Path, transform: transforms.Compose) -> datasets.ImageFolder:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing dataset split: {path}. Expected ImageFolder structure "
            "with train/ and val/ directories."
        )
    dataset = datasets.ImageFolder(path, transform=transform)
    if not dataset.classes:
        raise ValueError(f"Dataset split is empty: {path}")
    return dataset


def load_datasets(data_dir: str | Path, image_size: int = 224) -> DatasetBundle:
    """Load IP102 splits from ``data_dir/{train,val,test}``.

    The function validates that all available splits use the same class map.
    ``test`` is optional because some local copies of IP102 only contain train
    and validation folders.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    train = _load_split(root / "train", build_train_transform(image_size))
    eval_transform = build_eval_transform(image_size)
    val = _load_split(root / "val", eval_transform)

    if train.classes != val.classes:
        raise ValueError(
            "Class mappings differ between train and val. "
            f"train={train.classes[:5]}..., val={val.classes[:5]}..."
        )

    test_path = root / "test"
    test = _load_split(test_path, eval_transform) if test_path.is_dir() else None
    if test is not None and test.classes != train.classes:
        raise ValueError("Class mappings differ between train and test.")

    return DatasetBundle(train=train, val=val, test=test)


def make_loaders(
    bundle: DatasetBundle,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create loaders with Windows-safe defaults."""
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(bundle.train, shuffle=True, **common)
    val_loader = DataLoader(bundle.val, shuffle=False, **common)
    test_loader = (
        DataLoader(bundle.test, shuffle=False, **common) if bundle.test is not None else None
    )
    return train_loader, val_loader, test_loader
