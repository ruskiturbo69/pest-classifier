import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from pest_demo import predict_folder

def test_predict_folder_empty(tmp_path, capsys):
    # Setup mock for load_artifacts
    mock_model = MagicMock()
    mock_le = MagicMock()
    mock_config = {"IMG_SIZE": [64, 64], "HIST_BINS": 8}

    with patch('pest_demo.load_artifacts', return_value=(mock_model, mock_le, mock_config)):
        with patch('pest_demo.extract_features') as mock_extract:
            # Act
            predict_folder(tmp_path, Path("dummy_model_dir"))

            # Assert
            mock_extract.assert_not_called()
            mock_model.predict.assert_not_called()

            captured = capsys.readouterr()
            assert captured.out == ""
            assert captured.err == ""
