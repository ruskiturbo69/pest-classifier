import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import joblib

# ================== LOGGING ==================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ================== KONFIGURACJA ==================

IMG_SIZE: tuple[int, int] = (128, 128)
HIST_BINS: int = 8
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
N_ESTIMATORS: int = 300
CV_FOLDS: int = 5

# HOG parameters
HOG_ORIENTATIONS: int = 8
HOG_PIXELS_PER_CELL: tuple[int, int] = (16, 16)
HOG_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)

# LBP parameters
LBP_RADIUS: int = 1
LBP_N_POINTS: int = 8       # typically 8 * radius
LBP_BINS: int = 32          # histogram bins for LBP distribution


# ================== DANE I CECHY ==================

def load_dataset(root_dir: Path) -> pd.DataFrame:
    train_dir = root_dir / "train"
    image_paths: list[str] = []
    labels: list[str] = []

    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue
        for fpath in class_dir.iterdir():
            if fpath.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_paths.append(str(fpath))
                labels.append(class_dir.name)

    df = pd.DataFrame({"path": image_paths, "label": labels})
    log.info("Załadowano %d obrazów z %d klas.", len(df), df["label"].nunique())
    return df


def extract_features(
    image_path: str,
    img_size: tuple[int, int] = IMG_SIZE,
    bins: int = HIST_BINS,
    hog_orientations: int = HOG_ORIENTATIONS,
    hog_pixels_per_cell: tuple[int, int] = HOG_PIXELS_PER_CELL,
    hog_cells_per_block: tuple[int, int] = HOG_CELLS_PER_BLOCK,
    lbp_radius: int = LBP_RADIUS,
    lbp_n_points: int = LBP_N_POINTS,
    lbp_bins: int = LBP_BINS,
) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(img_size)
    img_np = np.array(img)

    # ---- colour features (original) ----
    mean_rgb = img_np.mean(axis=(0, 1))
    std_rgb = img_np.std(axis=(0, 1))

    hist_features: list[float] = []
    for channel in range(3):
        hist, _ = np.histogram(
            img_np[:, :, channel],
            bins=bins,
            range=(0, 255),
            density=True,
        )
        hist_features.extend(hist)

    # ---- grayscale for texture / shape features ----
    gray = rgb2gray(img_np)   # float64 in [0, 1], shape (H, W)

    # ---- HOG — shape & edge gradients ----
    hog_features = hog(
        gray,
        orientations=hog_orientations,
        pixels_per_cell=hog_pixels_per_cell,
        cells_per_block=hog_cells_per_block,
        feature_vector=True,
    )

    # ---- LBP — local texture patterns ----
    lbp = local_binary_pattern(gray, P=lbp_n_points, R=lbp_radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=lbp_bins, range=(0, lbp_bins), density=True)

    return np.concatenate([
        mean_rgb,        # 3
        std_rgb,         # 3
        hist_features,   # bins * 3 = 24
        hog_features,    # ~288 for 128x128 with default cell/block settings
        lbp_hist,        # lbp_bins = 32
    ])


def extract_features_for_df(
    df: pd.DataFrame,
    img_size: tuple[int, int] = IMG_SIZE,
    bins: int = HIST_BINS,
    hog_orientations: int = HOG_ORIENTATIONS,
    hog_pixels_per_cell: tuple[int, int] = HOG_PIXELS_PER_CELL,
    hog_cells_per_block: tuple[int, int] = HOG_CELLS_PER_BLOCK,
    lbp_radius: int = LBP_RADIUS,
    lbp_n_points: int = LBP_N_POINTS,
    lbp_bins: int = LBP_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    log.info("Ekstrakcja cech (równolegle)…")

    def _extract(path: str) -> np.ndarray:
        return extract_features(
            path, img_size, bins,
            hog_orientations, hog_pixels_per_cell, hog_cells_per_block,
            lbp_radius, lbp_n_points, lbp_bins,
        )

    with ThreadPoolExecutor() as executor:
        features = list(executor.map(_extract, df["path"]))

    X = np.vstack(features)
    y = df["label"].values
    log.info("Kształt macierzy cech: %s", X.shape)
    return X, y


# ================== MODELE ==================

def train_rf_model(
    X: np.ndarray,
    y_enc: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    n_estimators: int = N_ESTIMATORS,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc,
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    log.info("Trenowanie Random Forest (%d drzew)…", n_estimators)
    rf.fit(X_train, y_train)
    return rf, X_train, X_test, y_train, y_test


def evaluate_on_test(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: Path,
) -> None:
    y_pred = model.predict(X_test)

    log.info("=== Wyniki na zbiorze testowym ===")
    log.info("Accuracy : %.4f", accuracy_score(y_test, y_pred))
    log.info("F1 macro : %.4f", f1_score(y_test, y_pred, average="macro"))
    print(
        classification_report(
            y_test, y_pred, target_names=label_encoder.classes_
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cmap="Blues",
    )
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywista klasa")
    plt.title("Macierz pomyłek – Random Forest")
    plt.tight_layout()
    _save_fig(output_dir / "confusion_matrix_rf.png")


def plot_feature_importance(
    model: RandomForestClassifier,
    output_dir: Path,
    bins: int = HIST_BINS,
    top_k: int = 15,
) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    feat_names: list[str] = []
    for ch in ["R", "G", "B"]:
        feat_names.append(f"mean_{ch}")
    for ch in ["R", "G", "B"]:
        feat_names.append(f"std_{ch}")
    for ch in ["R", "G", "B"]:
        for i in range(bins):
            feat_names.append(f"hist_{ch}_{i}")

    # HOG feature count: orientations * cells_per_block^2 * n_blocks_per_axis^2
    # For 128x128, 16px cells, 2-cell blocks: (128/16 - 1)^2 * 4 * 8 = 49 * 32 = 1568 ... just label by index
    n_hog = model.n_features_in_ - (len(feat_names) + LBP_BINS)
    for i in range(n_hog):
        feat_names.append(f"hog_{i}")
    for i in range(LBP_BINS):
        feat_names.append(f"lbp_{i}")

    top_names = np.array(feat_names)[indices][:top_k]
    top_vals = importances[indices][:top_k]

    imp_df = pd.DataFrame({"feature": top_names, "importance": top_vals})

    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df, x="importance", y="feature")
    plt.title("Najważniejsze cechy – Random Forest")
    plt.tight_layout()
    _save_fig(output_dir / "feature_importance_rf.png")


def run_cross_validation(
    model: RandomForestClassifier,
    X: np.ndarray,
    y_enc: np.ndarray,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> None:
    """Stratified k-fold CV for both Random Forest and Logistic Regression."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Random Forest — evaluated on training data only to avoid test-set leakage
    scores_rf = cross_val_score(model, X, y_enc, cv=cv, scoring="accuracy")
    log.info(
        "=== CV (RF, stratified): %.3f +/- %.3f ===",
        scores_rf.mean(),
        scores_rf.std(),
    )

    # Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log_reg = LogisticRegression(max_iter=1000)
    scores_lr = cross_val_score(log_reg, X_scaled, y_enc, cv=cv, scoring="accuracy")
    log.info(
        "=== CV (LogReg, stratified): %.3f +/- %.3f ===",
        scores_lr.mean(),
        scores_lr.std(),
    )


# ================== ZAPIS / PREDYKCJA ==================

def save_artifacts(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    scaler: StandardScaler,
    output_dir: Path,
    img_size: tuple[int, int] = IMG_SIZE,
    hist_bins: int = HIST_BINS,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "rf_model.joblib")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")  # saved for LogReg inference

    config = {
        "IMG_SIZE": img_size,
        "HIST_BINS": hist_bins,
        "TEST_SIZE": TEST_SIZE,
        "RANDOM_STATE": RANDOM_STATE,
        "N_ESTIMATORS": N_ESTIMATORS,
        "CV_FOLDS": CV_FOLDS,
        "HOG_ORIENTATIONS": HOG_ORIENTATIONS,
        "HOG_PIXELS_PER_CELL": HOG_PIXELS_PER_CELL,
        "HOG_CELLS_PER_BLOCK": HOG_CELLS_PER_BLOCK,
        "LBP_RADIUS": LBP_RADIUS,
        "LBP_N_POINTS": LBP_N_POINTS,
        "LBP_BINS": LBP_BINS,
    }
    with open(output_dir / "features_config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("Artefakty zapisane w: %s", output_dir)


def predict_image(
    path_to_image: str | Path,
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    img_size: tuple[int, int] = IMG_SIZE,
    bins: int = HIST_BINS,
) -> str:
    path_to_image = Path(path_to_image)
    if not path_to_image.exists():
        raise FileNotFoundError(f"Obraz nie istnieje: {path_to_image}")

    feats = extract_features(str(path_to_image), img_size, bins).reshape(1, -1)
    label = label_encoder.inverse_transform(model.predict(feats))[0]
    return label


# ================== HELPERS ==================

def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    log.info("Wykres zapisany: %s", path)


def _plot_class_counts(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["label", "count"]

    plt.figure(figsize=(8, 4))
    sns.barplot(data=counts, x="label", y="count")
    plt.xticks(rotation=45)
    plt.title("Liczba obrazów w każdej klasie")
    plt.tight_layout()
    _save_fig(output_dir / "class_counts.png")


# ================== PIPELINE ==================

def run_training(root_dir: Path, output_dir: Path) -> None:
    df = load_dataset(root_dir)
    _plot_class_counts(df, output_dir)

    X, y = extract_features_for_df(df)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    rf, X_train, X_test, y_train, y_test = train_rf_model(X, y_enc)

    run_evaluation(rf, X_test, y_test, le, X, y_enc, output_dir)

    # Fit scaler on full X so it's usable at inference time
    scaler = StandardScaler().fit(X)
    save_artifacts(rf, le, scaler, output_dir)


def run_evaluation(
    rf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    X_full: np.ndarray,
    y_enc_full: np.ndarray,
    output_dir: Path,
) -> None:
    evaluate_on_test(rf, X_test, y_test, le, output_dir)
    plot_feature_importance(rf, output_dir, bins=HIST_BINS, top_k=15)
    run_cross_validation(rf, X_full, y_enc_full, CV_FOLDS, RANDOM_STATE)


# ================== ENTRY POINT ==================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pest image classifier – Random Forest")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("dataset"),
        help="Katalog główny datasetu (musi zawierać podkatalog 'train/').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ML"),
        help="Katalog wyjściowy na modele i wykresy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(root_dir=args.root_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
