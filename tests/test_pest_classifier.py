import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Use patch.dict to mock modules during the test only,
# although for top-level imports in the module under test,
# we might still need some level of sys.modules manipulation if we can't install dependencies.
# Given the environment constraints, I will keep the mocks but clean them up.

import sys

def test_predict_image_file_not_found():
    # Mocking the dependencies needed by pest_demo
    mock_modules = [
        "joblib", "matplotlib", "matplotlib.pyplot", "numpy", "pandas",
        "seaborn", "PIL", "skimage.color", "skimage.feature",
        "sklearn.ensemble", "sklearn.linear_model", "sklearn.metrics",
        "sklearn.model_selection", "sklearn.preprocessing"
    ]

    with patch.dict(sys.modules, {mod: MagicMock() for mod in mock_modules}):
        from pest_demo import predict_single

        # Arrange
        dummy_path = "non_existent_image.jpg"

        # We need to mock load_artifacts which is called in predict_single
        with patch('pest_demo.load_artifacts') as mock_load_artifacts, \
             patch('pest_demo.extract_features') as mock_extract_features:

            mock_model = MagicMock()
            mock_label_encoder = MagicMock()
            mock_config = {"IMG_SIZE": [64, 64], "HIST_BINS": 16}
            mock_load_artifacts.return_value = (mock_model, mock_label_encoder, mock_config)

            # extract_features in pest_classifier_2_0 raises FileNotFoundError internally when cv2/PIL fails or explicitly
            # Here we just want to ensure it raises FileNotFoundError, let's configure the mock
            mock_extract_features.side_effect = FileNotFoundError(f"Obraz nie istnieje: {dummy_path}")

            # Act & Assert
            with pytest.raises(FileNotFoundError) as excinfo:
                predict_single(Path(dummy_path), Path("dummy_model_dir"))

            assert "Obraz nie istnieje" in str(excinfo.value)
            assert dummy_path in str(excinfo.value)
