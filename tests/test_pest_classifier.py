import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

def test_predict_image_file_not_found():
    mock_modules = [
        "joblib", "matplotlib", "matplotlib.pyplot", "numpy", "pandas",
        "seaborn", "PIL", "skimage.color", "skimage.feature",
        "sklearn.ensemble", "sklearn.linear_model", "sklearn.metrics",
        "sklearn.model_selection", "sklearn.preprocessing"
    ]

    with patch.dict(sys.modules, {mod: MagicMock() for mod in mock_modules}):
        with patch("pest_demo.load_artifacts") as mock_load_artifacts:
            from pest_demo import predict_single as predict_image

            # Setup load_artifacts mock
            mock_model = MagicMock()
            mock_le = MagicMock()
            mock_config = {"IMG_SIZE": [128, 128], "HIST_BINS": 8}
            mock_load_artifacts.return_value = (mock_model, mock_le, mock_config)

            # mock the actual pest_demo.extract_features that was imported from pest_classifier_2_0
            with patch("pest_demo.extract_features") as mock_extract:
                dummy_path = "non_existent_image.jpg"
                mock_extract.side_effect = FileNotFoundError(f"Obraz nie istnieje: {dummy_path}")

                # Act & Assert
                with pytest.raises(FileNotFoundError) as excinfo:
                    predict_image(dummy_path, Path("dummy_dir"))

                assert "Obraz nie istnieje" in str(excinfo.value)
                assert dummy_path in str(excinfo.value)
