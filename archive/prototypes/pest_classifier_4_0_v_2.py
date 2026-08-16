import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 

# --- KONFIGURACJA ---
DATA_DIR = "dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 
EPOCHS = 30     
IMG_SIZE = 224

# --- 1. TRANSFORMACJE ---
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.TrivialAugmentWide(),
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

# --- 3. MODEL (Przesiadka na EfficientNet-V2-S) ---
# Używamy EfficientNet-V2-Small - świetny stosunek precyzji do obciążenia VRAM
model = models.efficientnet_v2_s(weights='DEFAULT')
# EfficientNet ma inną warstwę klasyfikacyjną (indeks 1 zamiast 3)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 102)
model = model.to(DEVICE)

# --- 4. OPTYMALIZACJA (Scheduler + AMP) ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Płynne zmniejszanie Learning Rate w miarę upływu epok
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
# Skaler do 16-bitowych obliczeń (oszczędza VRAM na RTX)
scaler = torch.amp.GradScaler('cuda')

# --- 5. PĘTLA TRENINGOWA ---
def train():
    print(f"🚀 Startujemy na: {DEVICE} ({torch.cuda.get_device_name(0)}) | EfficientNet-V2-S")
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            # non_blocking=True dla szybszego transferu asynchronicznego
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Kontekst AMP dla połowicznej precyzji (FP16)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

        # Aktualizacja schedulera po całej epoce
        scheduler.step()

        # Ewaluacja po każdej epoce
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        f1 = f1_score(y_true, y_pred, average='macro')
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f} | Loss: {running_loss/len(train_loader):.4f} | Val F1: {f1:.4f}")

        # Zapisujemy najlepszy model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_insect_model_v2.pth")
            print("✅ Nowy rekord F1! Model zapisany.")

if __name__ == "__main__":
    train()
