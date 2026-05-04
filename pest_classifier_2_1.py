import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # backend bez GUI, brak ostrzeżeń z tkinter
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
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
LBP_N_POINTS: int = 8  # typically 8 * radius
LBP_BINS: int = 32  # histogram bins for LBP distribution

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

    # colour features
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

    # grayscale for texture / shape features
    gray = rgb2gray(img_np)

    # HOG — shape & edge gradients
    hog_features = hog(
        gray,
        orientations=hog_orientations,
        pixels_per_cell=hog_pixels_per_cell,
        cells_per_block=hog_cells_per_block,
        feature_vector=True,
    )

    # LBP — local texture patterns
    lbp = local_binary_pattern(gray, P=lbp_n_points, R=lbp_radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=lbp_bins, range=(0, lbp_bins), density=True)

    return np.concatenate(
        [
            mean_rgb,  # 3
            std_rgb,  # 3
            hist_features,  # bins * 3 = 24
            hog_features,
            lbp_hist,  # 32
        ]
    )


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
            path,
            img_size,
            bins,
            hog_orientations,
            hog_pixels_per_cell,
            hog_cells_per_block,
            lbp_radius,
            lbp_n_points,
            lbp_bins,
        )

    with ThreadPoolExecutor() as executor:
        features = list(executor.map(_extract, df["path"]))

    X = np.vstack(features)
    y = df["label"].values
    log.info("Kształt macierzy cech: %s", X.shape)
    return X, y


# ================== MODELE I EWALUACJA ==================


def evaluate_model_basic(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Zwraca słownik z podstawowymi metrykami dla danego modelu."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    return {
        "model": name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
    }


def evaluate_on_test(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: Path,
) -> None:
    y_pred = model.predict(X_test)

    log.info("=== Wyniki na zbiorze testowym (Random Forest) ===")
    log.info("Accuracy : %.4f", accuracy_score(y_test, y_pred))
    log.info("F1 macro : %.4f", f1_score(y_test, y_pred, average="macro"))
    print(
        classification_report(
            y_test, y_pred, target_names=label_encoder.classes_
        )
    )

    # macierz pomyłek
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

    # predictions.csv do analizy błędów
    classes = label_encoder.classes_
    y_true_labels = classes[y_test]
    y_pred_labels = classes[y_pred]

    preds_df = pd.DataFrame(
        {
            "y_true_index": y_test,
            "y_true_label": y_true_labels,
            "y_pred_index": y_pred,
            "y_pred_label": y_pred_labels,
        }
    )
    preds_path = output_dir / "predictions_rf_test.csv"
    preds_df.to_csv(preds_path, index=False)
    log.info("Zapisano predykcje RF na zbiorze testowym: %s", preds_path)


def plot_feature_importance(
    model: RandomForestClassifier,
    output_dir: Path,
    bins: int = HIST_BINS,
    top_k: int = 15,
) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    channels = ["R", "G", "B"]
    feat_names = [f"mean_{ch}" for ch in channels]
    feat_names.extend([f"std_{ch}" for ch in channels])
    feat_names.extend([f"hist_{ch}_{i}" for ch in channels for i in range(bins)])

    # HOG feature count: reszta po odjęciu kolorów i LBP
    n_hog = model.n_features_in_ - (len(feat_names) + LBP_BINS)
    feat_names.extend([f"hog_{i}" for i in range(n_hog)])
    feat_names.extend([f"lbp_{i}" for i in range(LBP_BINS)])

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
    """Stratified k‑fold CV dla RF i logistycznej."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scores_rf = cross_val_score(model, X, y_enc, cv=cv, scoring="accuracy")
    log.info(
        "=== CV (RF, stratified): %.3f +/- %.3f ===",
        scores_rf.mean(),
        scores_rf.std(),
    )

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
    joblib.dump(scaler, output_dir / "scaler.joblib")

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


def generate_simple_report(
    output_dir: Path,
    dataset_info: dict,
    results_csv: Path,
) -> None:
    """Prosty raport w Markdown z metrykami i ścieżkami do wykresów."""
    report_path = output_dir / "report_experiment.md"

    results_df = pd.read_csv(results_csv)

    lines: list[str] = []
    lines.append("# Pest Classifier – experiment report\n")
    lines.append("## Dataset summary\n")
    lines.append(f"- Number of images: {dataset_info['n_images']}")
    lines.append(f"- Number of classes: {dataset_info['n_classes']}\n")

    lines.append("## Test results (per model)\n")
    lines.append(results_df.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Figures\n")
    lines.append("- Class counts: `class_counts.png`")
    lines.append("- Confusion matrix (RF): `confusion_matrix_rf.png`")
    lines.append("- Feature importance (RF): `feature_importance_rf.png`\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Zapisano raport eksperymentu: %s", report_path)


# ================== PIPELINE ==================


def run_training(root_dir: Path, output_dir: Path) -> None:
    df = load_dataset(root_dir)

    dataset_info = {
        "n_images": len(df),
        "n_classes": df["label"].nunique(),
    }

    _plot_class_counts(df, output_dir)

    X, y = extract_features_for_df(df)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    log.info("Trenowanie Random Forest (%d drzew)…", N_ESTIMATORS)
    rf.fit(X_train, y_train)

    # Logistic Regression baseline
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    log_reg = LogisticRegression(max_iter=1000)
    log.info("Trenowanie Logistic Regression (baseline)…")
    log_reg.fit(X_train_scaled, y_train)

    # porównanie modeli na teście
    results: list[dict] = []
    results.append(evaluate_model_basic("RandomForest", rf, X_test, y_test))
    results.append(
        evaluate_model_basic("LogisticRegression", log_reg, X_test_scaled, y_test)
    )

    results_df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results_test_models.csv"
    results_df.to_csv(results_path, index=False)
    log.info("Zapisano wyniki modeli: %s", results_path)

    # raport markdown
    generate_simple_report(
        output_dir=output_dir,
        dataset_info=dataset_info,
        results_csv=results_path,
    )

    # szczegółowa ewaluacja RF
    evaluate_on_test(rf, X_test, y_test, le, output_dir)
    plot_feature_importance(rf, output_dir, bins=HIST_BINS, top_k=15)
    run_cross_validation(rf, X, y_enc, CV_FOLDS, RANDOM_STATE)

    # scaler na pełnym X (do inference)
    scaler_full = StandardScaler().fit(X)
    save_artifacts(rf, le, scaler_full, output_dir)


# ================== ENTRY POINT ==================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pest image classifier – Random Forest"
    )
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
