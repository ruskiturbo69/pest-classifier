import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

# --- KONFIGURACJA ---
DATA_DIR = "dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 
EPOCHS = 20
IMG_SIZE = 224

# --- 1. TRANSFORMACJE (Data Augmentation) ---
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. ŁADOWANIE DANYCH ---
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=8, pin_memory=True)

# --- 3. MODEL (Transfer Learning) ---
# Używamy MobileNetV3-Large - szybki, lekki i skuteczny
model = models.mobilenet_v3_large(weights='DEFAULT')
# Podmiana końcówki na 102 klasy IP102
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 102)
model = model.to(DEVICE)

# --- 4. OPTYMALIZACJA ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. PĘTLA TRENINGOWA ---
def train():
    print(f"🚀 Startujemy na: {DEVICE} ({torch.cuda.get_device_name(0)})")
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Ewaluacja po każdej epoce
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | Val F1: {f1:.4f}")

        # Zapisujemy najlepszy model (Checkpointing)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_insect_model.pth")
            print("Nowy rekord F1! Model zapisany.")

if __name__ == "__main__":
    train()
