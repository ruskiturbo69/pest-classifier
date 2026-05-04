import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

def test_predict_single():
    mock_modules = [
        "joblib", "skimage", "skimage.feature", "skimage.color", "numpy", "PIL",
        "matplotlib", "matplotlib.pyplot", "pandas", "seaborn",
        "sklearn", "sklearn.ensemble", "sklearn.linear_model", "sklearn.metrics",
        "sklearn.model_selection", "sklearn.preprocessing", "scipy"
    ]
    with patch.dict(sys.modules, {mod: MagicMock() for mod in mock_modules}):
        import pest_demo

        with patch('pest_demo.load_artifacts') as mock_load_artifacts, \
             patch('pest_demo.extract_features') as mock_extract_features:

             mock_model = MagicMock()
             mock_model.predict.return_value = [1]

             mock_le = MagicMock()
             mock_le.inverse_transform.return_value = ["Moth"]

             mock_config = {"IMG_SIZE": [64, 64], "HIST_BINS": 8}

             mock_load_artifacts.return_value = (mock_model, mock_le, mock_config)

             mock_feats = MagicMock()
             mock_reshaped_feats = MagicMock()
             mock_feats.reshape.return_value = mock_reshaped_feats
             mock_extract_features.return_value = mock_feats

             image_path = Path("dummy_image.jpg")
             model_dir = Path("dummy_dir")

             result = pest_demo.predict_single(image_path, model_dir)

             assert result == "Moth"
             mock_load_artifacts.assert_called_once_with(model_dir)
             mock_extract_features.assert_called_once_with(
                 str(image_path),
                 img_size=(64, 64),
                 bins=8
             )
             mock_feats.reshape.assert_called_once_with(1, -1)
             mock_model.predict.assert_called_once_with(mock_reshaped_feats)
             mock_le.inverse_transform.assert_called_once_with([1])

def test_predict_single_file_not_found():
    mock_modules = [
        "joblib", "skimage", "skimage.feature", "skimage.color", "numpy", "PIL",
        "matplotlib", "matplotlib.pyplot", "pandas", "seaborn",
        "sklearn", "sklearn.ensemble", "sklearn.linear_model", "sklearn.metrics",
        "sklearn.model_selection", "sklearn.preprocessing", "scipy"
    ]

    with patch.dict(sys.modules, {mod: MagicMock() for mod in mock_modules}):
        from pest_demo import predict_single

        # Arrange
        dummy_path = Path("non_existent_image.jpg")
        mock_model_dir = Path("dummy_dir")

        with patch('pest_demo.load_artifacts') as mock_load_artifacts, \
             patch('pest_demo.extract_features') as mock_extract_features:

             mock_model = MagicMock()
             mock_le = MagicMock()
             mock_config = {"IMG_SIZE": [64, 64], "HIST_BINS": 8}

             mock_load_artifacts.return_value = (mock_model, mock_le, mock_config)

             # If extract_features encounters a missing file it will raise an error
             # (in our case we simulate it if we are testing that specific path)
             mock_extract_features.side_effect = FileNotFoundError(f"Obraz nie istnieje: {dummy_path}")

             # Act & Assert
             with pytest.raises(FileNotFoundError) as excinfo:
                 predict_single(dummy_path, mock_model_dir)

             assert "Obraz nie istnieje" in str(excinfo.value)
             assert str(dummy_path) in str(excinfo.value)
