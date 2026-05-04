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
        from pest_demo import predict_single
        from pest_classifier_2_0 import extract_features

        # Arrange
        dummy_path = Path("non_existent_image.jpg")
        mock_model_dir = Path("mock_dir")

        with patch('pest_demo.load_artifacts') as mock_load:
            mock_model = MagicMock()
            mock_le = MagicMock()
            mock_config = {"IMG_SIZE": [64, 64], "HIST_BINS": 32}
            mock_load.return_value = (mock_model, mock_le, mock_config)

            with patch('pest_demo.extract_features', side_effect=FileNotFoundError(f"Obraz nie istnieje: {dummy_path}")):
                # Act & Assert
                with pytest.raises(FileNotFoundError) as excinfo:
                    predict_single(dummy_path, mock_model_dir)

                assert "Obraz nie istnieje" in str(excinfo.value)
                assert str(dummy_path) in str(excinfo.value)
