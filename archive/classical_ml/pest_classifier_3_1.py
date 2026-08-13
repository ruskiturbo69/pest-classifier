import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import random
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
    "RANDOM_STATE": 42,
    "N_ESTIMATORS": 300,
    "HOG_ORIENTATIONS": 8,
    "HOG_PIXELS_PER_CELL": (16, 16),
    "HOG_CELLS_PER_BLOCK": (2, 2),
    "LBP_RADIUS": 1,
    "LBP_N_POINTS": 8,
    "LBP_BINS": 32,
    "MAX_SAMPLES_PER_CLASS_TRAIN": 100,  # Bezpiecznik RAM dla treningu
    "MAX_SAMPLES_PER_CLASS_TEST": 50     # Bezpiecznik RAM dla testu
}

def extract_features(img_path):
    """
    Ekstrakcja cech wizualnych (statystyki, Histogramy, HOG, LBP).
    """
    try:
        img = np.array(Image.open(img_path).convert("RGB").resize(CFG["IMG_SIZE"]))
    except Exception as e:
        log.warning(f"Nie można wczytać obrazu {img_path}: {e}")
        # Zwracamy zera w przypadku uszkodzonego pliku, by nie przerywać puli wątków
        expected_size = (6 + 3 * CFG["HIST_BINS"] + 
                         (CFG["IMG_SIZE"][0] // CFG["HOG_PIXELS_PER_CELL"][0] - 1) * (CFG["IMG_SIZE"][1] // CFG["HOG_PIXELS_PER_CELL"][1] - 1) * CFG["HOG_CELLS_PER_BLOCK"][0] * CFG["HOG_CELLS_PER_BLOCK"][1] * CFG["HOG_ORIENTATIONS"] + CFG["LBP_BINS"])
        return np.zeros(expected_size)

    gray = rgb2gray(img)

    # Statystyki bazowe
    mean_vals = img.mean(axis=(0, 1))
    std_vals = img.std(axis=(0, 1))

    # Histogramy
    hist_r = np.histogram(img[:, :, 0], bins=CFG["HIST_BINS"], range=(0, 255), density=True)[0]
    hist_g = np.histogram(img[:, :, 1], bins=CFG["HIST_BINS"], range=(0, 255), density=True)[0]
    hist_b = np.histogram(img[:, :, 2], bins=CFG["HIST_BINS"], range=(0, 255), density=True)[0]
    hist_vals = np.concatenate([hist_r, hist_g, hist_b])

    # HOG
    hog_features = hog(
        gray,
        orientations=CFG["HOG_ORIENTATIONS"],
        pixels_per_cell=CFG["HOG_PIXELS_PER_CELL"],
        cells_per_block=CFG["HOG_CELLS_PER_BLOCK"],
        feature_vector=True
    )

    # LBP
    lbp = local_binary_pattern(gray, P=CFG["LBP_N_POINTS"], R=CFG["LBP_RADIUS"], method="uniform")
    lbp_features = np.histogram(lbp, bins=CFG["LBP_BINS"], range=(0, CFG["LBP_BINS"]), density=True)[0]

    return np.concatenate([mean_vals, std_vals, hist_vals, hog_features, lbp_features])

def _save_fig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    log.info("Wykres zapisany: %s", path)

def load_ip102_metadata(split_dir: Path, max_samples_per_class: int = None) -> pd.DataFrame:
    """
    Ładowanie danych ze struktury hierarchicznej (ImageFolder).
    Bezpiecznik OOM aplikowany od razu na poziomie odczytu I/O z dysku.
    """
    if not split_dir.exists():
        raise FileNotFoundError(f"Krytyczny błąd: Nie znaleziono folderu {split_dir}")

    data = []
    # Iterujemy po folderach klas wewnątrz np. "train" (czyli foldery 0, 1, 2... 101)
    for class_dir in split_dir.iterdir():
        if class_dir.is_dir():
            label = class_dir.name # Pobiera nazwę folderu jako etykietę klasy
            
            # Pobieranie wszystkich ścieżek do zdjęć w danym folderze
            img_paths = [p for p in class_dir.glob("*") if p.suffix.lower() in {'.jpg', '.png', '.jpeg'}]
            
            # Subsampling (Bezpiecznik pamięci)
            if max_samples_per_class is not None and len(img_paths) > max_samples_per_class:
                random.seed(CFG["RANDOM_STATE"])
                # Sortujemy by mieć pewność, że random.sample na każdej maszynie zadziała tak samo
                img_paths = random.sample(sorted(img_paths), max_samples_per_class)
                
            for img_path in img_paths:
                data.append({"path": str(img_path), "label": label})
                
    df = pd.DataFrame(data)
    log.info(f"Wczytano folder {split_dir.name}: {len(df)} obrazów, {df['label'].nunique()} klas.")
    return df

def plot_class_distribution(df, output_dir, prefix="train"):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="label", order=df['label'].value_counts().index)
    plt.xticks(rotation=90, fontsize=6)
    plt.title(f"Liczba obrazów w każdej klasie ({prefix})")
    plt.tight_layout()
    _save_fig(output_dir / f"class_counts_{prefix}.png")

def extract_all_features(df):
    with ThreadPoolExecutor() as executor:
        features = list(executor.map(extract_features, df["path"]))
    return np.vstack(features)

def train_and_evaluate_models(X_tr, X_te, y_tr, y_te, output_dir, n_classes, n_samples):
    log.info("Rozpoczęto trening Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=CFG["N_ESTIMATORS"],
        random_state=CFG["RANDOM_STATE"],
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)

    log.info("Rozpoczęto skalowanie i trening Logistic Regression...")
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    lr = LogisticRegression(max_iter=1000, n_jobs=-1)
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

    report_content = f"""# Zgłoszenie z eksperymentu - Wersja 3.1 (IP102)
## Podsumowanie zbioru treningowego
- Liczba przetworzonych obrazów (z limitem): {n_samples}
- Liczba klas: {n_classes}

## Wyniki testów na wydzielonym zbiorze IP102
{res_df.to_markdown(index=False)}
"""
    with open(output_dir / "report_experiment.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    return rf

def plot_model_results(rf, X_te, y_te, le, output_dir):
    preds = rf.predict(X_te)
    
    # Zapis macierzy pomyłek
    plt.figure(figsize=(24, 20))
    sns.heatmap(
        confusion_matrix(y_te, preds),
        annot=False, # Zbyt dużo klas na adnotacje tekstowe w kafelkach
        cmap="Blues"
    )
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywista klasa")
    plt.title("Macierz pomyłek – Random Forest (IP102)")
    plt.tight_layout()
    _save_fig(output_dir / "confusion_matrix_rf.png")

    # Zapis najważniejszych cech
    f_names = (
        [f"{stat}_{ch}" for stat in ["mean", "std"] for ch in "RGB"] +
        [f"hist_{ch}_{i}" for ch in "RGB" for i in range(CFG["HIST_BINS"])] +
        [f"hog_{i}" for i in range(rf.n_features_in_ - 6 - 3 * CFG["HIST_BINS"] - CFG["LBP_BINS"])] +
        [f"lbp_{i}" for i in range(CFG["LBP_BINS"])]
    )

    top_indices = np.argsort(rf.feature_importances_)[::-1][:20]
    plt.figure(figsize=(10, 8))
    sns.barplot(x=rf.feature_importances_[top_indices], y=np.array(f_names)[top_indices])
    plt.title("20 Najważniejszych cech – Random Forest")
    plt.tight_layout()
    _save_fig(output_dir / "feature_importance_rf.png")

def save_artifacts(rf, le, X_tr, output_dir):
    joblib.dump(rf, output_dir / "rf_model.joblib")
    joblib.dump(le, output_dir / "label_encoder.joblib")

    scaler = StandardScaler()
    scaler.fit(X_tr)
    joblib.dump(scaler, output_dir / "scaler.joblib")

    with open(output_dir / "features_config.json", "w") as f:
        json.dump(CFG, f, indent=2)

def run_training(root_dir: Path, output_dir: Path):
    # Ścieżki teraz wskazują na Twoje fizyczne foldery
    train_dir = root_dir / "classification" / "train"
    test_dir = root_dir / "classification" / "test"
    
    log.info("--- ETAP 1: Ładowanie metadanych (Hierarchia Folderów) ---")
    df_train = load_ip102_metadata(train_dir, max_samples_per_class=CFG["MAX_SAMPLES_PER_CLASS_TRAIN"])
    df_test = load_ip102_metadata(test_dir, max_samples_per_class=CFG["MAX_SAMPLES_PER_CLASS_TEST"])

    plot_class_distribution(df_train, output_dir, prefix="train")
    
    log.info("--- ETAP 2: Ekstrakcja cech ---")
    log.info("Przetwarzanie zbioru TRENINGOWEGO...")
    X_tr = extract_all_features(df_train)
    y_tr_raw = df_train['label'].values
    
    log.info("Przetwarzanie zbioru TESTOWEGO...")
    X_te = extract_all_features(df_test)
    y_te_raw = df_test['label'].values

    log.info("--- ETAP 3: Kodowanie Etykiet ---")
    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_raw)
    
    # Zabezpieczenie przed klasami w teście, których nie było w treningu (częste przy agresywnym subsamplingu)
    known_classes = set(le.classes_)
    valid_test_indices = [i for i, label in enumerate(y_te_raw) if label in known_classes]
    
    if len(valid_test_indices) < len(y_te_raw):
        log.warning(f"Odrzucono {len(y_te_raw) - len(valid_test_indices)} próbek testowych z powodu nieznanych klas.")
        
    X_te = X_te[valid_test_indices]
    y_te_raw = y_te_raw[valid_test_indices]
    y_te = le.transform(y_te_raw)

    log.info("--- ETAP 4: Trening i Ewaluacja ---")
    rf = train_and_evaluate_models(
        X_tr, X_te, y_tr, y_te, output_dir, n_classes=len(le.classes_), n_samples=len(df_train)
    )
    results_path = output_dir / "results_test_models.csv"
    if results_path.exists():
        final_results = pd.read_csv(results_path)
        print("\n" + "="*50)
        print("FINALNE METRYKI MODELU (BASELINE ML)")
        print("="*50)
        print(final_results.to_string(index=False))
        print("="*50 + "\n")
    log.info("--- ETAP 5: Generowanie raportów i zapis artefaktów ---")
    try:
        plot_model_results(rf, X_te, y_te, le, output_dir)
    except Exception as e:
        log.error("Nie można było zapisać wykresów ewaluacji: %s", str(e))

    save_artifacts(rf, le, X_tr, output_dir)
    log.info("Pipeline klasycznego ML zakończony sukcesem. Model i logi zapisane.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pest image classifier (IP102) – Random Forest")
    # Domyślna ścieżka - zmień na swój folder z rozpakowanym Kaggle IP102
    parser.add_argument("--root-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("ML_IP102_Results"))
    args = parser.parse_args()

    run_training(args.root_dir, args.output_dir)
