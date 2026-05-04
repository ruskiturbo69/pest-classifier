import argparse, json, logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Słownik konfiguracyjny (CFG) - trzyma wszystkie hiperparametry w jednym miejscu.
CFG = {
    "IMG_SIZE": (128, 128), "HIST_BINS": 8, "TEST_SIZE": 0.2, "RANDOM_STATE": 42,
    "N_ESTIMATORS": 300, "CV_FOLDS": 5, "HOG_ORIENTATIONS": 8,
    "HOG_PIXELS_PER_CELL": (16, 16), "HOG_CELLS_PER_BLOCK": (2, 2),
    "LBP_RADIUS": 1, "LBP_N_POINTS": 8, "LBP_BINS": 32
}

def extract_features(img_path):
    # Wczytanie obrazu, konwersja do RGB i zmiana rozmiaru
    img = np.array(Image.open(img_path).convert("RGB").resize(CFG["IMG_SIZE"]))
    # Skala szarości dla HOG i LBP
    gray = rgb2gray(img)
    
    # Równoczesne wyliczenie i spłaszczenie wszystkich cech do jednego wektora 1D:
    return np.concatenate([
        img.mean(axis=(0, 1)), # Średnia z kanałów RGB
        img.std(axis=(0, 1)),  # Odchylenie standardowe z kanałów RGB
        # Histogramy kolorów dla każdego z 3 kanałów (wyliczone w pętli wewnątrz listy)
        np.concatenate([np.histogram(img[:, :, c], bins=CFG["HIST_BINS"], range=(0, 255), density=True)[0] for c in range(3)]),
        # Cechy HOG (analiza gradientów krawędzi)
        hog(gray, orientations=CFG["HOG_ORIENTATIONS"], pixels_per_cell=CFG["HOG_PIXELS_PER_CELL"], cells_per_block=CFG["HOG_CELLS_PER_BLOCK"], feature_vector=True),
        # Cechy LBP (analiza mikro-tekstury w postaci histogramu)
        np.histogram(local_binary_pattern(gray, P=CFG["LBP_N_POINTS"], R=CFG["LBP_RADIUS"], method="uniform"), bins=CFG["LBP_BINS"], range=(0, CFG["LBP_BINS"]), density=True)[0]
    ])

def _save_fig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path); plt.close(); log.info("Wykres zapisany: %s", path)

def run_training(train_dir, test_dir, output_dir):
    df_train = pd.DataFrame([{"path": str(p), "label": p.parent.name} for p in train_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if df_train.empty:
        log.error("Nie znaleziono obrazów treningowych w %s", train_dir)
        return
    log.info("Załadowano %d obrazów treningowych z %d klas.", len(df_train), df_train["label"].nunique())

    df_test = pd.DataFrame([{"path": str(p), "label": p.parent.name} for p in test_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if df_test.empty:
        log.error("Nie znaleziono obrazów testowych w %s", test_dir)
        return
    log.info("Załadowano %d obrazów testowych z %d klas.", len(df_test), df_test["label"].nunique())

    # Wizualizacja balansu klas dla zbioru treningowego
    plt.figure(figsize=(8, 4)); sns.countplot(data=df_train, x="label"); plt.xticks(rotation=45); plt.title("Liczba obrazów treningowych w każdej klasie"); plt.tight_layout(); _save_fig(output_dir / "class_counts.png")

    # Wielowątkowa ekstrakcja cech
    with ThreadPoolExecutor() as executor:
        X_tr = np.vstack(list(executor.map(extract_features, df_train["path"])))
        X_te = np.vstack(list(executor.map(extract_features, df_test["path"])))
        
    le = LabelEncoder()
    le.fit(np.concatenate([df_train["label"], df_test["label"]]))
    y_tr = le.transform(df_train["label"])
    y_te = le.transform(df_test["label"])

    # Inicjalizacja i trening Random Forest (ponownie walrus := przypisuje wytrenowany model do zmiennej rf)
    (rf := RandomForestClassifier(n_estimators=CFG["N_ESTIMATORS"], random_state=CFG["RANDOM_STATE"], n_jobs=-1)).fit(X_tr, y_tr)
    # To samo dla Logistic Regression, z użyciem pipeline do skalowania
    (lr := make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))).fit(X_tr, y_tr)

    # Funkcja pomocnicza ewaluująca model i zwracająca słownik z metrykami
    def eval_mod(name, mod, Xt):
        pr, rc, _, _ = precision_recall_fscore_support(y_te, p := mod.predict(Xt), average="macro", zero_division=0)
        return {"model": name, "accuracy": accuracy_score(y_te, p), "f1_macro": f1_score(y_te, p, average="macro"), "precision_macro": pr, "recall_macro": rc}

    # Złożenie wyników RF i LR do DataFrame i zapis do CSV
    res = pd.DataFrame([eval_mod("RandomForest", rf, X_te), eval_mod("LogisticRegression", lr, X_te)])
    output_dir.mkdir(parents=True, exist_ok=True); res.to_csv(out_res := output_dir / "results_test_models.csv", index=False)

    # Zapis prostego raportu w Markdown
    with open(output_dir / "report_experiment.md", "w", encoding="utf-8") as f:
        f.write(f"# Pest Classifier – experiment report\n## Dataset summary\n- Number of train images: {len(df_train)}\n- Number of test images: {len(df_test)}\n- Number of classes: {len(le.classes_)}\n\n## Test results (per model)\n{res.to_markdown(index=False)}\n\n## Figures\n- Class counts: `class_counts.png`\n- Confusion matrix (RF): `confusion_matrix_rf.png`\n- Feature importance (RF): `feature_importance_rf.png`\n")

    # Predykcja na zbiorze testowym, raport klasyfikacji i macierz pomyłek
    p = rf.predict(X_te)
    print(classification_report(y_te, p, labels=range(len(le.classes_)), target_names=le.classes_))
    plt.figure(figsize=(8, 6)); sns.heatmap(confusion_matrix(y_te, p, labels=range(len(le.classes_))), annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues"); plt.xlabel("Predykcja"); plt.ylabel("Rzeczywista klasa"); plt.title("Macierz pomyłek – Random Forest"); plt.tight_layout(); _save_fig(output_dir / "confusion_matrix_rf.png")
    pd.DataFrame({"y_true_index": y_te, "y_true_label": le.classes_[y_te], "y_pred_index": p, "y_pred_label": le.classes_[p]}).to_csv(output_dir / "predictions_rf_test.csv", index=False)

    # Dynamiczne generowanie nazw cech dla Feature Importance - łączenie list cech (RGB, hist, HOG, LBP)
    f_names = [f"{stat}_{ch}" for stat in ["mean", "std"] for ch in "RGB"] + [f"hist_{ch}_{i}" for ch in "RGB" for i in range(CFG["HIST_BINS"])] + [f"hog_{i}" for i in range(rf.n_features_in_ - 6 - 3*CFG["HIST_BINS"] - CFG["LBP_BINS"])] + [f"lbp_{i}" for i in range(CFG["LBP_BINS"])]
    top_i = np.argsort(rf.feature_importances_)[::-1][:15]
    plt.figure(figsize=(8, 6)); sns.barplot(x=rf.feature_importances_[top_i], y=np.array(f_names)[top_i]); plt.title("Najważniejsze cechy – Random Forest"); plt.tight_layout(); _save_fig(output_dir / "feature_importance_rf.png")

    # Stratyfikowana walidacja krzyżowa (Stratified K-Fold)
    cv = StratifiedKFold(n_splits=CFG["CV_FOLDS"], shuffle=True, random_state=CFG["RANDOM_STATE"])
    log.info("CV (RF): %.3f +/- %.3f", (s := cross_val_score(rf, X_tr, y_tr, cv=cv, scoring="accuracy")).mean(), s.std())
    log.info("CV (LogReg): %.3f +/- %.3f", (s := cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X_tr, y_tr, cv=cv, scoring="accuracy")).mean(), s.std())

    # Zapisanie wytrenowanego modelu, encodera, scalera i parametrów konfiguracyjnych
    joblib.dump(rf, output_dir / "rf_model.joblib"); joblib.dump(le, output_dir / "label_encoder.joblib"); joblib.dump(StandardScaler().fit(X_tr), output_dir / "scaler.joblib")
    with open(output_dir / "features_config.json", "w") as f: json.dump(CFG, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pest image classifier – Random Forest")
    p.add_argument("--train-dir", type=Path, default=Path(r"C:\Users\kwiac\Desktop\Studia\wlasne\maszynowe\insects\pest\pest\train"))
    p.add_argument("--test-dir", type=Path, default=Path(r"C:\Users\kwiac\Desktop\Studia\wlasne\maszynowe\insects\pest\pest\test"))
    p.add_argument("--output-dir", type=Path, default=Path("ML"))
    args = p.parse_args()
    run_training(args.train_dir, args.test_dir, args.output_dir)