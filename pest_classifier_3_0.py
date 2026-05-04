import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

CFG = {
    "IMG_SIZE": (128, 128),
    "HIST_BINS": 8,
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "N_ESTIMATORS": 300,
    "CV_FOLDS": 5,
    "HOG_ORIENTATIONS": 8,
    "HOG_PIXELS_PER_CELL": (16, 16),
    "HOG_CELLS_PER_BLOCK": (2, 2),
    "LBP_RADIUS": 1,
    "LBP_N_POINTS": 8,
    "LBP_BINS": 32
}

def extract_features(img_path):
    """
    Extracts visual features from an image given its path.

    Args:
        img_path (str or Path): Path to the image file.

    Returns:
        np.ndarray: A 1D numpy array containing concatenated mean, std,
        color histograms, HOG, and LBP features.
    """
    img = np.array(Image.open(img_path).convert("RGB").resize(CFG["IMG_SIZE"]))
    gray = rgb2gray(img)

    # Basic color statistics
    mean_vals = img.mean(axis=(0, 1))
    std_vals = img.std(axis=(0, 1))

    # Color histograms for each channel
    hist_r = np.histogram(img[:, :, 0], bins=CFG["HIST_BINS"], range=(0, 255), density=True)[0]
    hist_g = np.histogram(img[:, :, 1], bins=CFG["HIST_BINS"], range=(0, 255), density=True)[0]
    hist_b = np.histogram(img[:, :, 2], bins=CFG["HIST_BINS"], range=(0, 255), density=True)[0]
    hist_vals = np.concatenate([hist_r, hist_g, hist_b])

    # HOG features
    hog_features = hog(
        gray,
        orientations=CFG["HOG_ORIENTATIONS"],
        pixels_per_cell=CFG["HOG_PIXELS_PER_CELL"],
        cells_per_block=CFG["HOG_CELLS_PER_BLOCK"],
        feature_vector=True
    )

    # LBP features
    lbp = local_binary_pattern(gray, P=CFG["LBP_N_POINTS"], R=CFG["LBP_RADIUS"], method="uniform")
    lbp_features = np.histogram(lbp, bins=CFG["LBP_BINS"], range=(0, CFG["LBP_BINS"]), density=True)[0]

    # Concatenate all features into a single 1D array
    return np.concatenate([mean_vals, std_vals, hist_vals, hog_features, lbp_features])

def _save_fig(path):
    """
    Helper function to save and close the current matplotlib figure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    log.info("Wykres zapisany: %s", path)

def load_dataset_metadata(root_dir):
    """
    Loads dataset metadata from the given root directory.

    Args:
        root_dir (Path): Root directory containing the dataset.

    Returns:
        pd.DataFrame: DataFrame containing paths and labels for each image.
    """
    train_dir = root_dir / "train"
    image_paths = []

    for p in train_dir.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_paths.append({"path": str(p), "label": p.parent.name})

    df = pd.DataFrame(image_paths)
    log.info("Załadowano %d obrazów z %d klas.", len(df), df["label"].nunique())
    return df

def plot_class_distribution(df, output_dir):
    """
    Plots and saves the class distribution of the dataset.

    Args:
        df (pd.DataFrame): Dataset metadata.
        output_dir (Path): Directory to save the plot.
    """
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="label")
    plt.xticks(rotation=45)
    plt.title("Liczba obrazów w każdej klasie")
    plt.tight_layout()
    _save_fig(output_dir / "class_counts.png")

def extract_all_features(df):
    """
    Extracts features for all images in the dataset concurrently.

    Args:
        df (pd.DataFrame): Dataset metadata containing image paths.

    Returns:
        np.ndarray: Feature matrix X.
    """
    with ThreadPoolExecutor() as executor:
        features = list(executor.map(extract_features, df["path"]))
    return np.vstack(features)

def train_and_evaluate_models(X_tr, X_te, y_tr, y_te, output_dir, n_classes, n_samples):
    """
    Trains and evaluates Random Forest and Logistic Regression models.

    Args:
        X_tr (np.ndarray): Training features.
        X_te (np.ndarray): Testing features.
        y_tr (np.ndarray): Training labels.
        y_te (np.ndarray): Testing labels.
        output_dir (Path): Directory to save evaluation results.
        n_classes (int): Number of unique classes in the dataset.
        n_samples (int): Total number of samples in the dataset.

    Returns:
        RandomForestClassifier: The trained Random Forest model.
    """
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=CFG["N_ESTIMATORS"],
        random_state=CFG["RANDOM_STATE"],
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)

    # Train Logistic Regression
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_tr_scaled, y_tr)

    def evaluate_model(name, model, X_test):
        preds = model.predict(X_test)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_te, preds, average="macro", zero_division=0
        )
        return {
            "model": name,
            "accuracy": accuracy_score(y_te, preds),
            "f1_macro": f1_score(y_te, preds, average="macro"),
            "precision_macro": precision,
            "recall_macro": recall
        }

    results = [
        evaluate_model("RandomForest", rf, X_te),
        evaluate_model("LogisticRegression", lr, X_te_scaled)
    ]
    res_df = pd.DataFrame(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_dir / "results_test_models.csv", index=False)

    # Generate evaluation report
    report_content = f"""# Pest Classifier – experiment report
## Dataset summary
- Number of images: {n_samples}
- Number of classes: {n_classes}

## Test results (per model)
{res_df.to_markdown(index=False)}

## Figures
- Class counts: `class_counts.png`
- Confusion matrix (RF): `confusion_matrix_rf.png`
- Feature importance (RF): `feature_importance_rf.png`
"""
    with open(output_dir / "report_experiment.md", "w") as f:
        f.write(report_content)

    return rf

def plot_model_results(rf, X_te, y_te, le, output_dir):
    """
    Plots and saves confusion matrix and feature importances for the Random Forest model.

    Args:
        rf (RandomForestClassifier): Trained Random Forest model.
        X_te (np.ndarray): Testing features.
        y_te (np.ndarray): Testing labels.
        le (LabelEncoder): Fitted label encoder.
        output_dir (Path): Directory to save plots and predictions.
    """
    preds = rf.predict(X_te)
    print(classification_report(y_te, preds, target_names=le.classes_))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_te, preds),
        annot=True,
        fmt="d",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        cmap="Blues"
    )
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywista klasa")
    plt.title("Macierz pomyłek – Random Forest")
    plt.tight_layout()
    _save_fig(output_dir / "confusion_matrix_rf.png")

    # Save predictions
    preds_df = pd.DataFrame({
        "y_true_index": y_te,
        "y_true_label": le.classes_[y_te],
        "y_pred_index": preds,
        "y_pred_label": le.classes_[preds]
    })
    preds_df.to_csv(output_dir / "predictions_rf_test.csv", index=False)

    # Plot feature importances
    f_names = (
        [f"{stat}_{ch}" for stat in ["mean", "std"] for ch in "RGB"] +
        [f"hist_{ch}_{i}" for ch in "RGB" for i in range(CFG["HIST_BINS"])] +
        [f"hog_{i}" for i in range(rf.n_features_in_ - 6 - 3 * CFG["HIST_BINS"] - CFG["LBP_BINS"])] +
        [f"lbp_{i}" for i in range(CFG["LBP_BINS"])]
    )

    top_indices = np.argsort(rf.feature_importances_)[::-1][:15]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=rf.feature_importances_[top_indices], y=np.array(f_names)[top_indices])
    plt.title("Najważniejsze cechy – Random Forest")
    plt.tight_layout()
    _save_fig(output_dir / "feature_importance_rf.png")

def run_cross_validation(rf, X, y_enc):
    """
    Runs cross-validation for Random Forest and Logistic Regression models.

    Args:
        rf (RandomForestClassifier): Initialized Random Forest model.
        X (np.ndarray): Full feature matrix.
        y_enc (np.ndarray): Full encoded labels.
    """
    cv = StratifiedKFold(n_splits=CFG["CV_FOLDS"], shuffle=True, random_state=CFG["RANDOM_STATE"])

    # CV for Random Forest
    rf_scores = cross_val_score(rf, X, y_enc, cv=cv, scoring="accuracy")
    log.info("CV (RF): %.3f +/- %.3f", rf_scores.mean(), rf_scores.std())

    # CV for Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr_scores = cross_val_score(lr, X_scaled, y_enc, cv=cv, scoring="accuracy")
    log.info("CV (LogReg): %.3f +/- %.3f", lr_scores.mean(), lr_scores.std())

def save_artifacts(rf, le, X, output_dir):
    """
    Saves the trained model, label encoder, scaler, and feature configuration.

    Args:
        rf (RandomForestClassifier): Trained Random Forest model.
        le (LabelEncoder): Fitted label encoder.
        X (np.ndarray): Full feature matrix (used to fit the scaler).
        output_dir (Path): Directory to save the artifacts.
    """
    joblib.dump(rf, output_dir / "rf_model.joblib")
    joblib.dump(le, output_dir / "label_encoder.joblib")

    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, output_dir / "scaler.joblib")

    with open(output_dir / "features_config.json", "w") as f:
        json.dump(CFG, f, indent=2)

def run_training(root_dir, output_dir):
    """
    Main pipeline for training the pest classifier.

    Args:
        root_dir (Path): Directory containing the dataset.
        output_dir (Path): Directory to save models and outputs.
    """
    # 1. Load dataset metadata
    df = load_dataset_metadata(root_dir)

    # 2. Plot class distribution
    plot_class_distribution(df, output_dir)

    # 3. Extract features
    X = extract_all_features(df)

    # 4. Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(df["label"])

    # 5. Split dataset
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=CFG["TEST_SIZE"], random_state=CFG["RANDOM_STATE"], stratify=y_enc
    )

    # 6. Train and evaluate models
    rf = train_and_evaluate_models(
        X_tr, X_te, y_tr, y_te, output_dir, n_classes=df["label"].nunique(), n_samples=len(df)
    )

    # 7. Plot and save Random Forest results
    plot_model_results(rf, X_te, y_te, le, output_dir)

    # 8. Run cross-validation
    run_cross_validation(rf, X, y_enc)

    # 9. Save artifacts
    save_artifacts(rf, le, X, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pest image classifier – Random Forest")
    parser.add_argument("--root-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("ML"))
    args = parser.parse_args()

    run_training(args.root_dir, args.output_dir)
