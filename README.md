# Pest Classifier (Random Forest + HOG + LBP)

Educational machine learning project for automatic classification of 9 agricultural pest species from RGB images.

The pipeline extracts a rich set of features from each image (color, shape and texture) and trains a Random Forest classifier. A Logistic Regression model is used as a baseline to show how much can be achieved with good feature engineering alone.

---

## Technologies

- Python 3.11 / 3.14  
- scikit-learn (`RandomForestClassifier`, `LogisticRegression`)  
- scikit-image (HOG, Local Binary Patterns)  
- NumPy, pandas, matplotlib, seaborn, Pillow  
- joblib (model persistence)

---

## Dataset

The code assumes a directory structure with separate folders for each pest class under a common training directory, for example:

- `aphids`  
- `armyworm`  
- `beetle`  
- `bollworm`  
- `grasshopper`  
- `mites`  
- `mosquito`  
- `sawfly`  
- `stem_borer`  

The dataset itself is **not** included in this repository. Only a few small example images are stored in the `demo_jpg` folder for demonstration purposes.

---

## Project structure

- `pest_classifier_2_0.py` – main training pipeline (feature extraction, train/validation split, cross-validation, saving artifacts)  
- `pest_demo.py` – simple CLI demo: prediction for a single image or all images in a folder  
- `demo_jpg/` – a few example images for quick testing  
- `requirements.txt` – Python dependencies  

During training the script saves model artifacts to the output directory (for example `ML/`):

- `rf_model.joblib` – trained Random Forest classifier  
- `label_encoder.joblib` – `LabelEncoder` for class names  
- `features_config.json` – configuration of the feature extractor (image size, number of bins, HOG and LBP parameters)

---

## Installation

1. (Optional) Create and activate a virtual environment.  
2. Install dependencies:

   ```bash
   pip install -r requirements.txt

---

## Training

To train the model, run:

   ```bash
   python pest_classifier_2_0.py
   ```
The script will:
- load the training dataset,
- extract color + HOG + LBP features,
- train a Random Forest classifier,
- evaluate it on a held-out test set,
- run stratified cross-validation,
- save model artifacts and plots (class counts, confusion matrix, feature importance).

---

## Demo: predicting pests from images
After training, you can test the model on individual images or on a folder with images using pest_demo.py and the saved model directory (for example ML/).
The demo uses neutral file names (jpg_1.jpg, jpg_2.jpg, …) to show that predictions are not hard-coded and the model genuinely infers the class from image content.

---

## Results

- 9 pest classes, 2232 training images in total.
- Feature vector: color statistics, RGB histograms, HOG and LBP (around 1600 features per image).
- Random Forest (300 trees) and Logistic Regression both reach around 99.5–100% accuracy and macro‑F1 in 5‑fold stratified cross‑validation.
- On a held‑out test set the Random Forest model makes only 1–2 mistakes out of 447 images.

These results are generated with the extended training script (`pest_classifier_3_0.py`), which also:
- compares Random Forest and Logistic Regression on the same feature set,
- saves test metrics to `results_test_models.csv`,
- produces an automatic experiment report (`report_experiment.md`) and artifacts for analysis (`predictions_rf_test.csv`, `confusion_matrix_rf.png`, `feature_importance_rf.png`).

This demonstrates that carefully engineered hand‑crafted features can be competitive with more complex models for this type of well‑controlled data.

---

## Versions and experiment report

- `pest_classifier_2_0.py` – initial version of the training pipeline (hand‑crafted features + Random Forest, single model evaluation).
- `pest_classifier_2_1.py` – extended version with:
  - comparison of Random Forest and Logistic Regression on the same feature set,
  - test metrics saved to `results_test_models.csv`,
  - automatic experiment report `report_experiment.md` (dataset summary, per‑model metrics, figure list),
  - additional artifacts for analysis: `predictions_rf_test.csv`, `confusion_matrix_rf.png`, `feature_importance_rf.png`.

The latest experiments and results shown in this README are based on `pest_classifier_2_1.py`.

---

## Possible extensions
- Replace hand-crafted features with a CNN backbone (transfer learning, for example ResNet or EfficientNet).
- Add more pest species and more challenging real-world conditions (lighting, background, camera type).
- Deploy as a simple web or mobile application for farmers or integrate with UAV and field robots.

---

## Author
- **Szymon Kwiatkowski** – MSc student (Unmanned Systems and AI), Poznań University of Life Sciences, ML and computer vision enthusiast.  
- GitHub: [@ruskiturbo69](https://github.com/ruskiturbo69)
