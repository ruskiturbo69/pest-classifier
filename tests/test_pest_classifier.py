import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Use patch.dict to mock modules during the test only,
# although for top-level imports in the module under test,
# we might still need some level of sys.modules manipulation if we can't install dependencies.
# Given the environment constraints, I will keep the mocks but clean them up.

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
        # `predict_image` does not exist in pest_classifier_2_1, but we can fix the test to point to pest_demo which has it
        from pest_demo import predict_single

        # Arrange
        dummy_path = Path("non_existent_image.jpg")
        mock_model_dir = Path("mock_model_dir")

        # Act & Assert
        with pytest.raises(FileNotFoundError) as excinfo:
            predict_single(dummy_path, mock_model_dir)


def test_extract_features_file_not_found():
    # Only mock modules we don't strictly need for the test, allowing PIL and its exceptions to act naturally
    # Since we can install the dependencies and run it, we don't strictly need to mock PIL if we want to test Image.open
    # But since the previous test had mocks and comments suggested keeping them but cleaning up, we will configure the mock.

    mock_modules = [
        "joblib", "matplotlib", "matplotlib.pyplot", "numpy", "pandas",
        "seaborn", "PIL", "skimage.color", "skimage.feature",
        "sklearn.ensemble", "sklearn.linear_model", "sklearn.metrics",
        "sklearn.model_selection", "sklearn.preprocessing"
    ]

    # we need to make sure we don't shadow global sys with a local import
    with patch.dict(sys.modules, {mod: MagicMock() for mod in mock_modules}):
        sys.modules['PIL'].Image.open.side_effect = FileNotFoundError(f"[Errno 2] No such file or directory: 'non_existent_image.jpg'")

        from pest_classifier_2_1 import extract_features

        # Arrange
        dummy_path = "non_existent_image.jpg"

        # Act & Assert
        with pytest.raises(FileNotFoundError) as excinfo:
            extract_features(dummy_path)

        assert "No such file or directory: 'non_existent_image.jpg'" in str(excinfo.value)
