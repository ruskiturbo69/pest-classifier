# pest_demo.py
import argparse
from pathlib import Path
import json
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature")

from pest_classifier_2_0 import extract_features, IMG_SIZE, HIST_BINS


def load_artifacts(model_dir: Path):
    model = joblib.load(model_dir / "rf_model.joblib")
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")
    with open(model_dir / "features_config.json") as f:
        config = json.load(f)
    return model, label_encoder, config


def predict_single(image_path: Path, model_dir: Path) -> str:
    model, le, config = load_artifacts(model_dir)

    feats = extract_features(
        str(image_path),
        img_size=tuple(config["IMG_SIZE"]),
        bins=config["HIST_BINS"],
    ).reshape(1, -1)

    pred = model.predict(feats)
    return le.inverse_transform(pred)[0]


def predict_folder(folder: Path, model_dir: Path):
    model, le, config = load_artifacts(model_dir)

    image_paths = [
        p for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not image_paths:
        return

    all_feats = []
    for p in image_paths:
        feats = extract_features(
            str(p),
            img_size=tuple(config["IMG_SIZE"]),
            bins=config["HIST_BINS"],
        )
        all_feats.append(feats)

    preds = model.predict(all_feats)
    labels = le.inverse_transform(preds)

    for p, label in zip(image_paths, labels):
        print(f"{p.name:25s} -> {label}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo klasyfikatora szkodników"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Ścieżka do obrazu LUB folderu z obrazami",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("ML"),
        help="Folder z zapisanym modelem i encoderem",
    )
    args = parser.parse_args()

    if args.path.is_dir():
        predict_folder(args.path, args.model_dir)
    else:
        label = predict_single(args.path, args.model_dir)
        print(f"Przewidywana klasa szkodnika: {label}")


if __name__ == "__main__":
    main()
