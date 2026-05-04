import pytest
from pathlib import Path
import pandas as pd
from unittest.mock import MagicMock, patch
import sys

def test_predict_image_file_not_found():
    # Mocking the dependencies needed by pest_classifier_2_1
    mock_modules = [
        "joblib", "matplotlib", "matplotlib.pyplot", "numpy", "pandas",
        "seaborn", "PIL", "skimage.color", "skimage.feature",
        "sklearn.ensemble", "sklearn.linear_model", "sklearn.metrics",
        "sklearn.model_selection", "sklearn.preprocessing"
    ]

    with patch.dict(sys.modules, {mod: MagicMock() for mod in mock_modules}):
        from pest_classifier_2_1 import predict_image

        # Arrange
        dummy_path = "non_existent_image.jpg"
        mock_model = MagicMock()
        mock_label_encoder = MagicMock()

        # Act & Assert
        with pytest.raises(FileNotFoundError) as excinfo:
            predict_image(dummy_path, mock_model, mock_label_encoder)

        assert "Obraz nie istnieje" in str(excinfo.value)
        assert dummy_path in str(excinfo.value)

def test_load_dataset_edge_cases(tmp_path: Path):
    """
    Test edge cases for the load_dataset function.
    It should only load files with .jpg, .jpeg, .png extensions (case-insensitive)
    from directories under root_dir / "train".
    """
    from pest_classifier_2_1 import load_dataset

    # Create root directory structure
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Edge Case 1: Files directly in train_dir (not in class directories) should be ignored
    (train_dir / "ignore_me.jpg").touch()
    (train_dir / "ignore_me.txt").touch()

    # Edge Case 2: Valid class directory with supported image types (mixed case)
    class_a_dir = train_dir / "class_a"
    class_a_dir.mkdir()
    (class_a_dir / "img1.jpg").touch()
    (class_a_dir / "img2.PNG").touch()
    (class_a_dir / "img3.jpeg").touch()

    # Edge Case 3: Valid class directory with unsupported file types
    class_b_dir = train_dir / "class_b"
    class_b_dir.mkdir()
    (class_b_dir / "img4.JPG").touch()  # Valid
    (class_b_dir / "img_unsupported.gif").touch()  # Unsupported
    (class_b_dir / "text_file.txt").touch()  # Unsupported

    # Edge Case 4: Empty class directory
    class_empty_dir = train_dir / "class_empty"
    class_empty_dir.mkdir()

    # Run the function
    df = load_dataset(tmp_path)

    # Assertions
    # It should return a DataFrame with columns "path" and "label"
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["path", "label"]

    # It should load exactly 4 valid images: img1.jpg, img2.PNG, img3.jpeg, img4.JPG
    assert len(df) == 4

    # Extracting loaded paths
    loaded_paths = set(df["path"].tolist())

    # Expected paths
    expected_paths = {
        str(class_a_dir / "img1.jpg"),
        str(class_a_dir / "img2.PNG"),
        str(class_a_dir / "img3.jpeg"),
        str(class_b_dir / "img4.JPG")
    }

    assert loaded_paths == expected_paths

    # Verify labels correspond correctly
    # Sorting by path to make sure labels match
    df_sorted = df.sort_values(by="path").reset_index(drop=True)

    expected_data = pd.DataFrame([
        {"path": str(class_a_dir / "img1.jpg"), "label": "class_a"},
        {"path": str(class_a_dir / "img2.PNG"), "label": "class_a"},
        {"path": str(class_a_dir / "img3.jpeg"), "label": "class_a"},
        {"path": str(class_b_dir / "img4.JPG"), "label": "class_b"}
    ]).sort_values(by="path").reset_index(drop=True)

    pd.testing.assert_frame_equal(df_sorted, expected_data)
