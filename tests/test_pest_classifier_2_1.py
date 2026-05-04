import pytest
import pandas as pd
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock
from pest_classifier_2_1 import load_dataset, evaluate_model_basic

def test_load_dataset(tmp_path):
    """Test that load_dataset correctly identifies classes and images."""
    # Setup mock directory structure
    train_dir = tmp_path / "train"
    class_a_dir = train_dir / "classA"
    class_b_dir = train_dir / "classB"
    class_a_dir.mkdir(parents=True)
    class_b_dir.mkdir(parents=True)

    # Create dummy files
    (class_a_dir / "img1.jpg").touch()
    (class_a_dir / "img2.png").touch()
    (class_a_dir / "not_an_image.txt").touch()
    (class_b_dir / "img3.jpeg").touch()

    # Call function
    df = load_dataset(tmp_path)

    # Verify
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "path" in df.columns
    assert "label" in df.columns

    labels = df["label"].tolist()
    assert labels.count("classA") == 2
    assert labels.count("classB") == 1

    paths = df["path"].tolist()
    assert any("img1.jpg" in p for p in paths)
    assert any("img2.png" in p for p in paths)
    assert any("img3.jpeg" in p for p in paths)
    assert not any("not_an_image.txt" in p for p in paths)

def test_evaluate_model_basic():
    """Test that evaluate_model_basic calculates metrics correctly."""
    # Setup dummy data and mock model
    X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_test = np.array([0, 1, 0, 1])

    mock_model = MagicMock()
    # Mock predict to return perfect predictions for simplicity
    # or some specific predictions to test calculations. Let's do 3 correct, 1 wrong
    # True:  [0, 1, 0, 1]
    # Preds: [0, 1, 0, 0] -> accuracy = 0.75
    mock_model.predict.return_value = np.array([0, 1, 0, 0])

    # Call function
    result = evaluate_model_basic("MockModel", mock_model, X_test, y_test)

    # Verify
    assert isinstance(result, dict)
    assert result["model"] == "MockModel"
    assert result["accuracy"] == 0.75
    assert "f1_macro" in result
    assert "precision_macro" in result
    assert "recall_macro" in result

    # Check mock was called with X_test
    mock_model.predict.assert_called_once_with(X_test)
