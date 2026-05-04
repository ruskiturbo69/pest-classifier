# pest_demo.py
import argparse
from pathlib import Path
import json
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature")

# Add the directory containing this script to sys.path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pest_classifier_3_0 import extract_features, CFG


def load_artifacts(model_dir: Path):
    model = joblib.load(model_dir / "rf_model.joblib")
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")
    with open(model_dir / "features_config.json") as f:
        config = json.load(f)
    return model, label_encoder, config


def update_global_config(config: dict):
    """
    Updates the global CFG in pest_classifier_3_0 to match the configuration
    the model was trained on, to ensure correct feature extraction.
    """
    for k, v in config.items():
        if k == "IMG_SIZE" and isinstance(v, list):
            v = tuple(v)
        elif k in ["HOG_PIXELS_PER_CELL", "HOG_CELLS_PER_BLOCK"] and isinstance(v, list):
            v = tuple(v)
        CFG[k] = v


def predict_single(image_path: Path, model_dir: Path) -> str:
    model, le, config = load_artifacts(model_dir)
    update_global_config(config)

    feats = extract_features(str(image_path)).reshape(1, -1)
    pred = model.predict(feats)
    return le.inverse_transform(pred)[0]


def predict_folder(folder: Path, model_dir: Path):
    model, le, config = load_artifacts(model_dir)
    update_global_config(config)

    image_paths = [
        p for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    for p in image_paths:
        feats = extract_features(str(p)).reshape(1, -1)
        pred = model.predict(feats)
        label = le.inverse_transform(pred)[0]
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
