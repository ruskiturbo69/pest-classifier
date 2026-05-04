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
