# Klasyfikator Szkodników (Pest Classifier)
*(Scroll down for English version)*

Projekt edukacyjny uczenia maszynowego służący do automatycznej klasyfikacji szkodników rolniczych (9 gatunków) na podstawie obrazów RGB.

Oryginalnie aplikacja została stworzona jako projekt zaliczeniowy z dziedziny ML w trakcie studiów magisterskich oraz jako projekt do portfolio.

Model wykorzystuje klasyfikator Losowych Lasów (Random Forest) oraz ekstrakcję ręcznie definiowanych cech, m.in. HOG, LBP (Local Binary Patterns) oraz charakterystyki kolorów, demonstrując jak wysokie wyniki można osiągnąć za pomocą solidnej inżynierii cech, w porównaniu z klasyczną metodą bazową (Regresja Logistyczna).

---

## Technologie

- Python 3.11 / 3.12+
- scikit-learn (`RandomForestClassifier`, `LogisticRegression`)  
- scikit-image (HOG, Local Binary Patterns)  
- NumPy, pandas, matplotlib, seaborn, Pillow  
- joblib (zapis i odczyt modelu)

---

## Zbiór danych

Zbiór wymaga folderów z przypisanymi gatunkami wewnątrz katalogu głównego (domyślnie: `dataset/train/`), np.:
- `aphids`  
- `armyworm`  
- `beetle`  
- `bollworm`  
- `grasshopper`  
- `mites`  
- `mosquito`  
- `sawfly`  
- `stem_borer`  

Ze względów rozmiarowych pełen zbiór danych **nie** znajduje się w tym repozytorium. W katalogu `demo_jpg/` i w `dataset/train/` dołączono jedynie kilka zdjęć demonstracyjnych ułatwiających przetestowanie kodu.

---

## Struktura projektu

Główny kod został zrefaktoryzowany i znajduje się obecnie w folderze `src/`:

- `src/pest_classifier_3_0.py` – uproszczony i poprawiony główny skrypt treningowy, zredukowana ilość kodu przy zachowaniu pełnej funkcjonalności, posiada odporność na brakujące / zbyt małe klasy.
- `src/pest_demo.py` – skrypt demonstracyjny CLI: klasyfikacja dla jednego zdjęcia lub całego folderu.

Poprzednie, archiwalne wersje (pokazujące rozwój projektu) przeniesiono do folderu `legacy/`:
- `legacy/pest_classifier_2_1.py` – wcześniejsza, rozszerzona wersja rurociągu (porównania modeli, obszerny raport).
- `legacy/pest_classifier_2_0.py` – oryginalna, pierwsza wersja.
- `legacy/pest_classifier_condensed.py` – skrócona, eksperymentalna wersja.

W trakcie procesu treningowego artefakty (modele, konfiguracja, wykresy) zapisywane są do katalogu (np. `ML/`):
- `rf_model.joblib` – wytrenowany klasyfikator Random Forest.
- `label_encoder.joblib` – `LabelEncoder` przechowujący nazwy klas.
- `features_config.json` – konfiguracja parametrów ekstrakcji cech (wielkość obrazu, parametry HOG/LBP).

---

## Instalacja

1. Opcjonalnie: stwórz i aktywuj środowisko wirtualne.
2. Zainstaluj wymagane pakiety komendą:

   ```bash
   pip install -r requirements.txt
   ```

---

## Trening

Aby rozpocząć trenowanie modelu:

   ```bash
   python src/pest_classifier_3_0.py
   ```

Skrypt wykonuje m.in.:
- Załadowanie danych z obrazów,
- Równoległą ekstrakcję cech (kolor + HOG + LBP),
- Wytrenowanie klasyfikatora Random Forest,
- Przeprowadzenie walidacji krzyżowej (cross-validation) uwzględniającej wielkość klas,
- Zapisanie artefaktów modelu oraz wygenerowanie wykresów w folderze wyjściowym (`ML/`).

---

## Demo (Inference)
Po wytrenowaniu, możesz przetestować model używając `pest_demo.py`.

Przewidywanie na pojedynczym pliku:
```bash
python src/pest_demo.py demo_jpg/jpg_1.jpg --model-dir ML
```

Przewidywanie na folderze:
```bash
python src/pest_demo.py demo_jpg/ --model-dir ML
```

---

## Autor
- **Szymon Kwiatkowski** – Student MSc (Unmanned Systems and AI), Poznań University of Life Sciences. Pasjonat Machine Learning i Computer Vision.
- GitHub: [@ruskiturbo69](https://github.com/ruskiturbo69)

---
---

# Pest Classifier (English)

Educational machine learning project for the automatic classification of 9 agricultural pest species from RGB images.

The application was originally created as an academic master's degree project and as a machine learning portfolio piece.

The model uses a Random Forest classifier alongside handcrafted feature extraction, including HOG, LBP (Local Binary Patterns), and color features, demonstrating how high accuracy can be achieved with solid feature engineering compared to a basic Logistic Regression baseline.

---

## Technologies

- Python 3.11 / 3.12+
- scikit-learn (`RandomForestClassifier`, `LogisticRegression`)
- scikit-image (HOG, Local Binary Patterns)
- NumPy, pandas, matplotlib, seaborn, Pillow
- joblib (model persistence)

---

## Dataset

The dataset requires folders assigned to specific pest species located in the main directory (default: `dataset/train/`), e.g.:
- `aphids`
- `armyworm`
- `beetle`
- `bollworm`
- `grasshopper`
- `mites`
- `mosquito`
- `sawfly`
- `stem_borer`

For storage constraints, the entire dataset is **not** included in this repository. Only a few demonstration images are kept in `demo_jpg/` and `dataset/train/` to help test the code quickly.

---

## Project Structure

The main codebase has been refactored and is now located in the `src/` directory:

- `src/pest_classifier_3_0.py` – simplified and optimized main training script, reducing code bloat while retaining full functionality. It now also gracefully handles small or missing datasets.
- `src/pest_demo.py` – CLI demonstration script: classification for a single image or an entire directory.

Older, archival versions (showing the evolution of the code) have been moved to the `legacy/` directory:
- `legacy/pest_classifier_2_1.py` – an earlier extended pipeline version (model comparison, detailed reporting).
- `legacy/pest_classifier_2_0.py` – the original first iteration.
- `legacy/pest_classifier_condensed.py` – a condensed, experimental version.

During training, artifacts (models, configuration, plots) are saved to the output directory (e.g., `ML/`):
- `rf_model.joblib` – trained Random Forest classifier.
- `label_encoder.joblib` – `LabelEncoder` storing the class names.
- `features_config.json` – configuration parameters for feature extraction (image size, HOG/LBP parameters).

---

## Installation

1. Optionally: create and activate a virtual environment.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Training

To train the model, simply run:

   ```bash
   python src/pest_classifier_3_0.py
   ```

The script will handle:
- Loading image data,
- Concurrent feature extraction (Color + HOG + LBP),
- Training a Random Forest classifier,
- Running Stratified Cross-Validation (adapted to class size limits),
- Saving model artifacts and outputting analytic plots in the target folder (`ML/`).

---

## Demo (Inference)
After training the model, you can test it out using `pest_demo.py`.

Predicting a single image:
```bash
python src/pest_demo.py demo_jpg/jpg_1.jpg --model-dir ML
```

Predicting an entire directory:
```bash
python src/pest_demo.py demo_jpg/ --model-dir ML
```

---

## Author
- **Szymon Kwiatkowski** – MSc student (Unmanned Systems and AI), Poznań University of Life Sciences. ML and Computer Vision enthusiast.
- GitHub: [@ruskiturbo69](https://github.com/ruskiturbo69)
