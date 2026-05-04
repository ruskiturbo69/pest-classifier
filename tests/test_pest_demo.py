import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import sys

mock_modules = [
    "joblib", "matplotlib", "matplotlib.pyplot", "numpy", "pandas",
    "seaborn", "PIL", "skimage", "skimage.color", "skimage.feature", "skimage.io", "skimage.transform",
    "sklearn", "sklearn.ensemble", "sklearn.linear_model", "sklearn.metrics",
    "sklearn.model_selection", "sklearn.preprocessing", "cv2"
]

def test_load_artifacts(tmp_path):
    with patch.dict(sys.modules, {mod: MagicMock() for mod in mock_modules}):
        import joblib
        from pest_demo import load_artifacts

        # Mock joblib.load
        joblib.load.side_effect = ["mock_model", "mock_le"]

        # Mock open
        m_open = mock_open(read_data='{"IMG_SIZE": [64, 64], "HIST_BINS": 8}')
        with patch("builtins.open", m_open):
            model, le, config = load_artifacts(tmp_path)

        assert model == "mock_model"
        assert le == "mock_le"
        assert config == {"IMG_SIZE": [64, 64], "HIST_BINS": 8}

        joblib.load.assert_any_call(tmp_path / "rf_model.joblib")
        joblib.load.assert_any_call(tmp_path / "label_encoder.joblib")
        m_open.assert_called_with(tmp_path / "features_config.json")

def test_predict_single(tmp_path):
    with patch.dict(sys.modules, {mod: MagicMock() for mod in mock_modules}):
        from pest_demo import predict_single

        mock_model = MagicMock()
        mock_model.predict.return_value = ["pred_idx"]

        mock_le = MagicMock()
        mock_le.inverse_transform.return_value = ["mock_pest"]

        mock_config = {"IMG_SIZE": [64, 64], "HIST_BINS": 8}

        mock_feats = MagicMock()
        mock_feats.reshape.return_value = "reshaped_feats"

        with patch("pest_demo.load_artifacts", return_value=(mock_model, mock_le, mock_config)), \
             patch("pest_demo.extract_features", return_value=mock_feats):

            image_path = tmp_path / "test.jpg"
            model_dir = tmp_path / "ML"

            label = predict_single(image_path, model_dir)

            assert label == "mock_pest"
            mock_model.predict.assert_called_once_with("reshaped_feats")
            mock_le.inverse_transform.assert_called_once_with(["pred_idx"])

def test_predict_folder(tmp_path, capsys):
    with patch.dict(sys.modules, {mod: MagicMock() for mod in mock_modules}):
        from pest_demo import predict_folder

        # Create some fake images
        img1 = tmp_path / "img1.jpg"
        img1.touch()
        img2 = tmp_path / "img2.png"
        img2.touch()
        # Non-image file
        txt = tmp_path / "not_img.txt"
        txt.touch()

        mock_model = MagicMock()
        mock_model.predict.return_value = ["pred_idx"]

        mock_le = MagicMock()
        mock_le.inverse_transform.return_value = ["mock_pest"]

        mock_config = {"IMG_SIZE": [64, 64], "HIST_BINS": 8}

        mock_feats = MagicMock()
        mock_feats.reshape.return_value = "reshaped_feats"

        with patch("pest_demo.load_artifacts", return_value=(mock_model, mock_le, mock_config)), \
             patch("pest_demo.extract_features", return_value=mock_feats):

            model_dir = tmp_path / "ML"
            predict_folder(tmp_path, model_dir)

            captured = capsys.readouterr()

            assert "img1.jpg" in captured.out
            assert "img2.png" in captured.out
            assert "mock_pest" in captured.out
            assert "not_img.txt" not in captured.out

def test_main_is_dir(tmp_path):
    with patch.dict(sys.modules, {mod: MagicMock() for mod in mock_modules}):
        from pest_demo import main

        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        mock_args = MagicMock()
        mock_args.path = test_dir
        mock_args.model_dir = tmp_path / "ML"

        with patch("argparse.ArgumentParser.parse_args", return_value=mock_args), \
             patch("pest_demo.predict_folder") as mock_predict_folder, \
             patch("pest_demo.predict_single") as mock_predict_single:

            main()

            mock_predict_folder.assert_called_once_with(test_dir, tmp_path / "ML")
            mock_predict_single.assert_not_called()

def test_main_is_file(tmp_path, capsys):
    with patch.dict(sys.modules, {mod: MagicMock() for mod in mock_modules}):
        from pest_demo import main

        test_file = tmp_path / "test_file.jpg"
        test_file.touch()

        mock_args = MagicMock()
        mock_args.path = test_file
        mock_args.model_dir = tmp_path / "ML"

        with patch("argparse.ArgumentParser.parse_args", return_value=mock_args), \
             patch("pest_demo.predict_folder") as mock_predict_folder, \
             patch("pest_demo.predict_single", return_value="mock_pest") as mock_predict_single:

            main()

            captured = capsys.readouterr()

            mock_predict_single.assert_called_once_with(test_file, tmp_path / "ML")
            mock_predict_folder.assert_not_called()
            assert "Przewidywana klasa szkodnika: mock_pest" in captured.out
