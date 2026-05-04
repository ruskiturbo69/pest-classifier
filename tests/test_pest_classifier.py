import pytest
from pathlib import Path

# Fix sys.path for the tests to import src.pest_demo properly
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pest_demo import predict_single

def test_predict_image_file_not_found(tmp_path):
    # Arrange
    dummy_path = Path("non_existent_image.jpg")

    # Utworzenie tymczasowego modelu
    model_dir = tmp_path / "ML"
    model_dir.mkdir()

    # Plik konfiguracyjny - to jest minimalnie potrzebne by load_artifacts nie rzucał błędu na początku
    import json
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    with open(model_dir / "features_config.json", "w") as f:
        json.dump({
            "IMG_SIZE": [128, 128],
            "HIST_BINS": 8,
            "HOG_ORIENTATIONS": 8,
            "HOG_PIXELS_PER_CELL": [16, 16],
            "HOG_CELLS_PER_BLOCK": [2, 2],
            "LBP_RADIUS": 1,
            "LBP_N_POINTS": 8,
            "LBP_BINS": 32
        }, f)

    joblib.dump(RandomForestClassifier(), model_dir / "rf_model.joblib")
    joblib.dump(LabelEncoder(), model_dir / "label_encoder.joblib")

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        # The exact exception class might be FileNotFoundError or PIL.UnidentifiedImageError
        # depending on exactly how Image.open handles it, but FileNotFoundError is typical for a completely missing file
        predict_single(dummy_path, model_dir)
