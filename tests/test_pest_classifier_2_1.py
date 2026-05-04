import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import pest_classifier_2_1


def test_load_dataset(tmp_path: Path):
    """Test load_dataset to ensure it correctly parses image paths and labels."""
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Create two classes
    class1_dir = train_dir / "class_1"
    class1_dir.mkdir()
    class2_dir = train_dir / "class_2"
    class2_dir.mkdir()

    # Create dummy images and a non-image file
    (class1_dir / "img1.jpg").touch()
    (class1_dir / "img2.png").touch()
    (class1_dir / "not_an_image.txt").touch()

    (class2_dir / "img3.jpeg").touch()

    # Create an empty class dir to ensure it's handled gracefully
    (train_dir / "class_empty").mkdir()

    # Call load_dataset
    df = pest_classifier_2_1.load_dataset(tmp_path)

    assert len(df) == 3
    assert "class_1" in df["label"].values
    assert "class_2" in df["label"].values
    assert "class_empty" not in df["label"].values


def test_extract_features(tmp_path: Path):
    """Test feature extraction on a minimal dummy image."""
    img_path = tmp_path / "dummy.jpg"

    # Create a solid color 128x128 image
    img = Image.new("RGB", (128, 128), color=(255, 0, 0))
    img.save(img_path)

    # Extract features
    features = pest_classifier_2_1.extract_features(str(img_path))

    assert isinstance(features, np.ndarray)
    assert features.ndim == 1

    # mean_rgb (3), std_rgb (3), hist (HIST_BINS*3 = 24), HOG (dependent on default settings), LBP (LBP_BINS = 32)
    # The default lengths can be inferred from the pest_classifier_2_1 settings.
    # Total features should be consistent for a given size.
    assert len(features) > 0


def test_evaluate_model_basic():
    """Test evaluate_model_basic using a mock model and fixed predictions."""
    X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])  # 3 correct, 1 incorrect

    mock_model = MagicMock()
    mock_model.predict.return_value = y_pred

    metrics = pest_classifier_2_1.evaluate_model_basic("MockModel", mock_model, X_test, y_test)

    assert metrics["model"] == "MockModel"
    assert metrics["accuracy"] == 0.75  # 3/4 correct

    # We don't test exact sklearn metric calculation rules strictly here,
    # just that the functions are called and they output expected structures.
    assert "f1_macro" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics


def test_parse_args():
    """Test command line argument parsing defaults and overrides."""
    with patch.object(sys, "argv", ["script.py"]):
        args = pest_classifier_2_1.parse_args()
        assert args.root_dir == Path("dataset")
        assert args.output_dir == Path("ML")

    with patch.object(sys, "argv", ["script.py", "--root-dir", "my_data", "--output-dir", "my_output"]):
        args = pest_classifier_2_1.parse_args()
        assert args.root_dir == Path("my_data")
        assert args.output_dir == Path("my_output")
