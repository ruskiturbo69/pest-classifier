# Methodology / Metodologia

This document provides an academic explanation of the methodologies, libraries, feature engineering techniques, machine learning algorithms, and evaluation metrics utilized in the Pest Classifier project. The explanation is provided in both English and Polish.

---

## English Academic Explanation

### 1. Software Libraries and Frameworks
The implementation of the proposed classification pipeline heavily relies on robust open-source libraries:
- **`numpy` and `pandas`**: Utilized for efficient multi-dimensional array operations, mathematical computations, and structured data manipulation.
- **`Pillow` (PIL) and `scikit-image`**: Employed as the primary image processing frameworks. `scikit-image` is specifically chosen for its advanced implementations of feature extraction algorithms.
- **`scikit-learn`**: Serves as the core machine learning library, providing standardized APIs for model training, data scaling, cross-validation, and performance evaluation.
- **`matplotlib` and `seaborn`**: Used for generating high-quality academic visualizations, such as class distribution plots, feature importance bar charts, and confusion matrices.
- **`joblib`**: Applied for the efficient serialization of trained models, label encoders, and scalers, facilitating seamless inference.
- **`concurrent.futures`**: Leveraged to implement multithreaded parallel processing during the computationally expensive feature extraction phase.

### 2. Feature Engineering Strategy
Rather than utilizing deep neural networks to implicitly learn representations, this project employs a hand-crafted feature engineering approach to extract distinct visual characteristics from the agricultural pest images:
- **Color Statistics**: The pipeline calculates the mean, standard deviation, and histograms across the Red, Green, and Blue (RGB) channels. These features encapsulate the global color distribution, which often acts as a strong discriminative factor between different biological species.
- **Histogram of Oriented Gradients (HOG)**: Extracted from grayscale conversions of the images, HOG captures local object appearances and shapes by analyzing the distribution of intensity gradients and edge directions. This method is highly robust to variations in lighting and slight positional shifts.
- **Local Binary Patterns (LBP)**: Implemented to describe the micro-texture of the pest surface. LBP operates by thresholding the neighborhood of each pixel and treating the result as a binary number, effectively capturing patterns such as spots, lines, and flat areas irrespective of global illumination changes.

### 3. Machine Learning Algorithms
- **Random Forest Classifier**: Selected as the primary predictive model. As an ensemble learning method constructed from numerous decision trees, Random Forest naturally mitigates the risk of overfitting through bootstrap aggregating (bagging) and random subspace methods. Furthermore, it inherently provides a feature importance metric, which is crucial for interpreting the model's decision-making process in biological applications.
- **Logistic Regression**: Deployed as a baseline model. The purpose of evaluating a linear classifier like Logistic Regression is to demonstrate the discriminative quality of the extracted feature space; high performance from a linear baseline indicates that the engineered features (HOG, LBP, and color statistics) are sufficiently separating the classes in a high-dimensional space.

### 4. Evaluation Metrics and Validation Protocol
- **Stratified K-Fold Cross-Validation**: Employed to ensure a robust, unbiased estimation of model generalization capability. Stratification guarantees that the proportional representation of each pest class is maintained across all training and validation folds.
- **Evaluation Metrics**: The models are assessed using Accuracy, Precision, Recall, and the Macro F1-score. The Macro F1-score is particularly important as it calculates metrics for each class independently and finds their unweighted mean, thereby heavily penalizing poor performance on minority classes.
- **Confusion Matrix**: Utilized to provide a granular analysis of inter-class misclassifications, enabling the identification of morphologically similar pest species that the model struggles to differentiate.

---

## Metodologia (Polish Academic Explanation)

### 1. Wykorzystane Biblioteki i Narzędzia
Implementacja proponowanego potoku klasyfikacyjnego w dużej mierze opiera się na sprawdzonych bibliotekach open-source:
- **`numpy` oraz `pandas`**: Wykorzystywane do wydajnych operacji na wielowymiarowych tablicach, obliczeń matematycznych oraz manipulacji ustrukturyzowanymi danymi.
- **`Pillow` (PIL) oraz `scikit-image`**: Zastosowane jako główne ramy przetwarzania obrazów. Biblioteka `scikit-image` została wybrana ze względu na zaawansowane implementacje algorytmów ekstrakcji cech.
- **`scikit-learn`**: Pełni funkcję głównej biblioteki uczenia maszynowego, dostarczając ustandaryzowane interfejsy API do trenowania modeli, skalowania danych, walidacji krzyżowej i oceny wydajności.
- **`matplotlib` oraz `seaborn`**: Służą do generowania wysokiej jakości wizualizacji, takich jak wykresy rozkładu klas, wykresy istotności cech oraz macierze pomyłek.
- **`joblib`**: Zastosowane do wydajnej serializacji wytrenowanych modeli, koderów etykiet oraz skalerów, co umożliwia ich późniejsze wykorzystanie w procesie wnioskowania (inference).
- **`concurrent.futures`**: Wykorzystane do zaimplementowania wielowątkowego przetwarzania równoległego podczas kosztownego obliczeniowo etapu ekstrakcji cech.

### 2. Strategia Inżynierii Cech (Feature Engineering)
Zamiast polegać na głębokich sieciach neuronowych do ukrytego uczenia się reprezentacji, w projekcie zastosowano podejście oparte na ręcznie zaprojektowanych cechach (hand-crafted features) w celu wyodrębnienia kluczowych wizualnych charakterystyk szkodników rolniczych:
- **Statystyki Kolorów**: Potok oblicza średnią, odchylenie standardowe oraz histogramy dla kanałów Czerwonego, Zielonego i Niebieskiego (RGB). Cechy te rejestrują globalny rozkład barw, który często stanowi silny czynnik dyskryminujący poszczególne gatunki biologiczne.
- **Histogram Zorientowanych Gradientów (HOG)**: Ekstrahowany z obrazów przekształconych do skali szarości, algorytm HOG rejestruje lokalny wygląd obiektów i ich kształt poprzez analizę rozkładu gradientów intensywności oraz kierunków krawędzi. Metoda ta jest wysoce odporna na zmiany oświetlenia oraz niewielkie przesunięcia przestrzenne.
- **Lokalne Wzorce Binarne (LBP)**: Zaimplementowane w celu opisu mikro-tekstury powierzchni szkodnika. Algorytm LBP działa poprzez progowanie sąsiedztwa każdego piksela i traktowanie wyniku jako liczby binarnej, co skutecznie wychwytuje wzorce takie jak plamy, linie oraz płaskie obszary niezależnie od globalnych zmian oświetlenia.

### 3. Algorytmy Uczenia Maszynowego
- **Klasyfikator Lasu Losowego (Random Forest)**: Wybrany jako główny model predykcyjny. Jako metoda uczenia zespołowego (ensemble learning) zbudowana z wielu drzew decyzyjnych, Las Losowy naturalnie ogranicza ryzyko przeuczenia (overfitting) poprzez zastosowanie agregacji próbkowania (bagging) oraz metody losowych podprzestrzeni cech. Co więcej, model ten z założenia dostarcza metryki istotności cech, co jest kluczowe dla interpretacji procesu decyzyjnego w zastosowaniach biologicznych.
- **Regresja Logistyczna (Logistic Regression)**: Zastosowana jako model referencyjny (baseline). Celem oceny liniowego klasyfikatora, jakim jest Regresja Logistyczna, jest wykazanie jakości dyskryminacyjnej wyekstrahowanej przestrzeni cech; wysoka wydajność bazowego modelu liniowego wskazuje, że zaprojektowane cechy (HOG, LBP i statystyki kolorów) w wystarczającym stopniu separują klasy w wielowymiarowej przestrzeni.

### 4. Metryki Ewaluacyjne i Protokół Walidacji
- **Stratyfikowana Walidacja Krzyżowa (Stratified K-Fold Cross-Validation)**: Zastosowana w celu zapewnienia rzetelnej i nieobciążonej oceny zdolności modelu do generalizacji. Stratyfikacja gwarantuje zachowanie proporcjonalnej reprezentacji każdej klasy szkodnika we wszystkich podziałach treningowych i walidacyjnych.
- **Metryki Ewaluacyjne**: Modele oceniane są za pomocą Dokładności (Accuracy), Precyzji (Precision), Czułości (Recall) oraz wartości Makro F1-score. Metryka Makro F1-score jest szczególnie istotna, ponieważ oblicza wyniki dla każdej klasy niezależnie, a następnie wyciąga ich średnią arytmetyczną, surowo karząc tym samym słabą skuteczność na klasach mniejszościowych.
- **Macierz Pomyłek (Confusion Matrix)**: Wykorzystywana do szczegółowej analizy błędnych klasyfikacji między poszczególnymi klasami, umożliwiając identyfikację morfologicznie podobnych gatunków szkodników, z rozróżnieniem których model ma największe trudności.
