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
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CFG = {
    "IMG_SIZE": (128, 128), "HIST_BINS": 8, "TEST_SIZE": 0.2, "RANDOM_STATE": 42,
    "N_ESTIMATORS": 300, "CV_FOLDS": 5, "HOG_ORIENTATIONS": 8,
    "HOG_PIXELS_PER_CELL": (16, 16), "HOG_CELLS_PER_BLOCK": (2, 2),
    "LBP_RADIUS": 1, "LBP_N_POINTS": 8, "LBP_BINS": 32
}

def extract_features(img_path):
    img = np.array(Image.open(img_path).convert("RGB").resize(CFG["IMG_SIZE"]))
    gray = rgb2gray(img)
    return np.concatenate([
        img.mean(axis=(0, 1)), img.std(axis=(0, 1)),
        np.concatenate([np.histogram(img[:, :, c], bins=CFG["HIST_BINS"], range=(0, 255), density=True)[0] for c in range(3)]),
        hog(gray, orientations=CFG["HOG_ORIENTATIONS"], pixels_per_cell=CFG["HOG_PIXELS_PER_CELL"], cells_per_block=CFG["HOG_CELLS_PER_BLOCK"], feature_vector=True),
        np.histogram(local_binary_pattern(gray, P=CFG["LBP_N_POINTS"], R=CFG["LBP_RADIUS"], method="uniform"), bins=CFG["LBP_BINS"], range=(0, CFG["LBP_BINS"]), density=True)[0]
    ])

def _save_fig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path); plt.close(); log.info("Wykres zapisany: %s", path)

def run_training(root_dir, output_dir):
    df = pd.DataFrame([{"path": str(p), "label": p.parent.name} for p in (root_dir / "train").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    log.info("Załadowano %d obrazów z %d klas.", len(df), df["label"].nunique())

    plt.figure(figsize=(8, 4)); sns.countplot(data=df, x="label"); plt.xticks(rotation=45); plt.title("Liczba obrazów w każdej klasie"); plt.tight_layout(); _save_fig(output_dir / "class_counts.png")

    with ThreadPoolExecutor() as executor: X = np.vstack(list(executor.map(extract_features, df["path"])))
    y_enc = (le := LabelEncoder()).fit_transform(df["label"])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=CFG["TEST_SIZE"], random_state=CFG["RANDOM_STATE"], stratify=y_enc)

    (rf := RandomForestClassifier(n_estimators=CFG["N_ESTIMATORS"], random_state=CFG["RANDOM_STATE"], n_jobs=-1)).fit(X_tr, y_tr)
    (lr := LogisticRegression(max_iter=1000)).fit(StandardScaler().fit_transform(X_tr), y_tr)

    def eval_mod(name, mod, Xt):
        pr, rc, _, _ = precision_recall_fscore_support(y_te, p := mod.predict(Xt), average="macro", zero_division=0)
        return {"model": name, "accuracy": accuracy_score(y_te, p), "f1_macro": f1_score(y_te, p, average="macro"), "precision_macro": pr, "recall_macro": rc}

    res = pd.DataFrame([eval_mod("RandomForest", rf, X_te), eval_mod("LogisticRegression", lr, StandardScaler().fit(X_tr).transform(X_te))])
    output_dir.mkdir(parents=True, exist_ok=True); res.to_csv(out_res := output_dir / "results_test_models.csv", index=False)

    with open(output_dir / "report_experiment.md", "w") as f:
        f.write(f"# Pest Classifier – experiment report\n## Dataset summary\n- Number of images: {len(df)}\n- Number of classes: {df['label'].nunique()}\n\n## Test results (per model)\n{res.to_markdown(index=False)}\n\n## Figures\n- Class counts: `class_counts.png`\n- Confusion matrix (RF): `confusion_matrix_rf.png`\n- Feature importance (RF): `feature_importance_rf.png`\n")

    p = rf.predict(X_te)
    print(classification_report(y_te, p, target_names=le.classes_))
    plt.figure(figsize=(8, 6)); sns.heatmap(confusion_matrix(y_te, p), annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues"); plt.xlabel("Predykcja"); plt.ylabel("Rzeczywista klasa"); plt.title("Macierz pomyłek – Random Forest"); plt.tight_layout(); _save_fig(output_dir / "confusion_matrix_rf.png")
    pd.DataFrame({"y_true_index": y_te, "y_true_label": le.classes_[y_te], "y_pred_index": p, "y_pred_label": le.classes_[p]}).to_csv(output_dir / "predictions_rf_test.csv", index=False)

    f_names = [f"{stat}_{ch}" for stat in ["mean", "std"] for ch in "RGB"] + [f"hist_{ch}_{i}" for ch in "RGB" for i in range(CFG["HIST_BINS"])] + [f"hog_{i}" for i in range(rf.n_features_in_ - 6 - 3*CFG["HIST_BINS"] - CFG["LBP_BINS"])] + [f"lbp_{i}" for i in range(CFG["LBP_BINS"])]
    top_i = np.argsort(rf.feature_importances_)[::-1][:15]
    plt.figure(figsize=(8, 6)); sns.barplot(x=rf.feature_importances_[top_i], y=np.array(f_names)[top_i]); plt.title("Najważniejsze cechy – Random Forest"); plt.tight_layout(); _save_fig(output_dir / "feature_importance_rf.png")

    cv = StratifiedKFold(n_splits=CFG["CV_FOLDS"], shuffle=True, random_state=CFG["RANDOM_STATE"])
    log.info("CV (RF): %.3f +/- %.3f", (s := cross_val_score(rf, X, y_enc, cv=cv, scoring="accuracy")).mean(), s.std())
    log.info("CV (LogReg): %.3f +/- %.3f", (s := cross_val_score(LogisticRegression(max_iter=1000), StandardScaler().fit_transform(X), y_enc, cv=cv, scoring="accuracy")).mean(), s.std())

    joblib.dump(rf, output_dir / "rf_model.joblib"); joblib.dump(le, output_dir / "label_encoder.joblib"); joblib.dump(StandardScaler().fit(X), output_dir / "scaler.joblib")
    with open(output_dir / "features_config.json", "w") as f: json.dump(CFG, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pest image classifier – Random Forest")
    p.add_argument("--root-dir", type=Path, default=Path("dataset"))
    p.add_argument("--output-dir", type=Path, default=Path("ML"))
    args = p.parse_args()
    run_training(args.root_dir, args.output_dir)
