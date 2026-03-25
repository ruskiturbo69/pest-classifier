Pest Classifier (Random Forest + HOG + LBP)
Educational machine learning project for automatic classification of 9 agricultural pest species from RGB images.

The pipeline extracts a rich set of features from each image (color, shape and texture) and trains a Random Forest classifier. A Logistic Regression model is used as a baseline to show how much can be achieved with good feature engineering alone.

Technologies
Python 3.11 / 3.14

scikit-learn (RandomForestClassifier, LogisticRegression)

scikit-image (HOG, Local Binary Patterns)

NumPy, pandas, matplotlib, seaborn, Pillow

joblib (model persistence)

Dataset
The code assumes a directory structure with separate folders for each pest class under a common training directory (for example: aphids, armyworm, beetle, bollworm, grasshopper, mites, mosquito, sawfly, stem_borer).

The dataset itself is not included in this repository. Only a few small example images are stored in the demo_jpg folder for demonstration purposes.

Project structure
pest_classifier_2_0.py – main training pipeline (feature extraction, train/validation split, cross-validation, saving artifacts)

pest_demo.py – simple CLI demo: prediction for a single image or all images in a folder

demo_jpg – a few example images for quick testing

requirements.txt – Python dependencies

During training the script saves model artifacts to the output directory (for example ML):

rf_model.joblib – trained Random Forest classifier

label_encoder.joblib – LabelEncoder for class names

features_config.json – configuration of the feature extractor (image size, number of bins, HOG and LBP parameters)

Installation
Create and activate a virtual environment (optional but recommended), then install dependencies with pip install -r requirements.txt.

Training
Run the main training script (python pest_classifier_2_0.py).
This will load the training dataset, extract color + HOG + LBP features, train a Random Forest classifier, evaluate it on a held-out test set, run stratified cross-validation, and save model artifacts and plots (class counts, confusion matrix, feature importance).

Demo: predicting pests from images
After training, you can test the model on individual images or on a folder with images using pest_demo.py and the saved model directory (for example ML).

The demo uses neutral file names (jpg_1.jpg, jpg_2.jpg, …) to show that predictions are not hard-coded and the model genuinely infers the class from image content.

Results
9 pest classes, 2232 training images in total

Feature vector: color statistics, RGB histograms, HOG and LBP (around 1600 features per image)

Random Forest (300 trees) and Logistic Regression both reach around 99.5% accuracy and macro-F1 in 5-fold stratified cross-validation

On a held-out test set the model makes only 1–2 mistakes out of 447 images

This demonstrates that carefully engineered hand-crafted features can be competitive with more complex models for this type of well-controlled data.

Possible extensions
Replace hand-crafted features with a CNN backbone (transfer learning, for example ResNet or EfficientNet)

Add more pest species and more challenging real-world conditions (lighting, background, camera type)

Deploy as a simple web or mobile application for farmers or integrate with UAV and field robots

Klasyfikator szkodników (Random Forest + HOG + LBP)
Projekt edukacyjny z zakresu uczenia maszynowego dotyczący automatycznej klasyfikacji 9 gatunków szkodników roślin uprawnych na podstawie obrazów RGB.

Pipeline wyznacza z każdego obrazu bogaty zestaw cech (kolor, kształt i tekstura), a następnie trenuje klasyfikator Random Forest. Dodatkowo wykorzystana jest regresja logistyczna jako model bazowy, aby pokazać, ile można osiągnąć wyłącznie dzięki dobrej inżynierii cech.

Technologie
Python 3.11 / 3.14

scikit-learn (RandomForestClassifier, LogisticRegression)

scikit-image (HOG, Local Binary Patterns)

NumPy, pandas, matplotlib, seaborn, Pillow

joblib (zapis i wczytywanie modeli)

Zbiór danych
Kod zakłada strukturę katalogów z osobnymi folderami dla każdej klasy szkodników w ramach katalogu treningowego (na przykład: aphids, armyworm, beetle, bollworm, grasshopper, mites, mosquito, sawfly, stem_borer).

Sam zbiór danych nie jest dołączony do repozytorium. W folderze demo_jpg znajdują się jedynie przykładowe obrazy do szybkiego przetestowania działania modelu.

Struktura projektu
pest_classifier_2_0.py – główny pipeline treningowy (ekstrakcja cech, podział na zbiór treningowy i walidacyjny, walidacja krzyżowa, zapisywanie artefaktów)

pest_demo.py – proste demo w trybie CLI: predykcja dla pojedynczego obrazu lub całego folderu

demo_jpg – kilka przykładowych obrazów do szybkiego testu

requirements.txt – lista zależności Pythona

W trakcie treningu skrypt zapisuje artefakty modelu do katalogu wyjściowego (np. ML):

rf_model.joblib – wytrenowany klasyfikator Random Forest

label_encoder.joblib – obiekt LabelEncoder z nazwami klas

features_config.json – konfiguracja ekstraktora cech (rozmiar obrazu, liczba binów, parametry HOG i LBP)

Instalacja
Zaleca się utworzenie i aktywację wirtualnego środowiska, a następnie instalację zależności komendą pip install -r requirements.txt.

Trening
Uruchom główny skrypt treningowy (python pest_classifier_2_0.py).
Skrypt wczyta zbiór treningowy, wyznaczy cechy koloru, kształtu (HOG) i tekstury (LBP), wytrenuje klasyfikator Random Forest, oceni go na wydzielonym zbiorze testowym, wykona walidację krzyżową oraz zapisze artefakty modelu i wykresy (liczność klas, macierz pomyłek, ważność cech).

Demo: predykcja szkodników ze zdjęć
Po treningu można przetestować model na pojedynczych obrazach lub na całym folderze ze zdjęciami, korzystając ze skryptu pest_demo.py i katalogu z zapisanym modelem (np. ML).

Demo używa neutralnych nazw plików (jpg_1.jpg, jpg_2.jpg, …), aby pokazać, że predykcje nie są zakodowane „na sztywno” i model faktycznie wnioskuje klasę na podstawie treści obrazu.

Wyniki
9 klas szkodników, 2232 obrazów w zbiorze treningowym

Wektor cech: statystyki koloru, histogramy RGB, HOG i LBP (około 1600 cech na obraz)

Random Forest (300 drzew) oraz regresja logistyczna osiągają około 99,5% accuracy i macro-F1 w 5-krotnej walidacji krzyżowej

Na wydzielonym zbiorze testowym model popełnia zaledwie 1–2 błędy na 447 obrazów

Pokazuje to, że starannie zaprojektowane, ręcznie wyznaczane cechy mogą konkurować z bardziej złożonymi modelami w przypadku dobrze kontrolowanych danych.

Możliwe rozszerzenia
Zastąpienie cech ręcznych modelem CNN (transfer learning, np. ResNet lub EfficientNet)

Dodanie kolejnych gatunków szkodników oraz bardziej wymagających warunków (różne oświetlenie, tła, typy kamer)

Udostępnienie modelu w formie prostej aplikacji webowej lub mobilnej dla rolników, bądź integracja z UAV i robotami polowymi