import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Konfiguracja sprzętu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Wczytanie Twojego modelu (Pamiętamy, że to wersja LARGE!)
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 102)
model.load_state_dict(torch.load('best_insect_model.pth'))
model.to(device)
model.eval()

# 3. Transformacje
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. ŚCIEŻKA DO FOLDERU (Pamiętaj o 'r' na początku!)
# Podmień na własną ścieżkę do folderu z 10 zdjęciami
folder_path = 'data/demo'

print(f"Rozpoczynam analizę w folderze: {folder_path}")
print("-" * 60)
print(f"{'NAZWA PLIKU':<25} | {'KLASA (ID)':<10} | {'PEWNOŚĆ MODELU'}")
print("-" * 60)

# 5. Pętla przez wszystkie pliki w folderze
for filename in os.listdir(folder_path):
    # Sprawdzamy, czy plik jest zdjęciem
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        img_path = os.path.join(folder_path, filename)
        
        try:
            # Wczytywanie i predykcja
            img = Image.open(img_path).convert('RGB')
            img_t = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_t)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Bierzemy 3 najlepsze wyniki
                top3_prob, top3_pred = torch.topk(probabilities, 3)
                
            # Wyświetlanie wyników Top-3 (BEZ STAREGO KODU)
            print(f"{filename:<25} | GŁÓWNY TYP: ID {top3_pred[0].item():<4} ({top3_prob[0].item()*100:.1f}%)")
            print(f"{'':<25} | ALTERNATYWY: ID {top3_pred[1].item()} ({top3_prob[1].item()*100:.1f}%), ID {top3_pred[2].item()} ({top3_prob[2].item()*100:.1f}%)")
            
        except Exception as e:
            print(f"{filename:<25} | BŁĄD ODCZYTU: {e}")
                
            # Wyświetlanie wyników w ładnej tabelce
            confidence_pct = conf.item() * 100
            print(f"{filename:<25} | ID: {pred.item():<6} | {confidence_pct:>6.2f} %")
            
        except Exception as e:
            print(f"{filename:<25} | BŁĄD ODCZYTU: {e}")

print("-" * 60)
print("Analiza zakończona!")
