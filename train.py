# ==========================================================
# 📘 DRIVER DROWSINESS DETECTION — TRAINING SCRIPT
# Model: DriverStateCNN
# Author: Shubham Patel (NIT Raipur)
# ==========================================================

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# ⚙️ CONFIGURATION
# ==========================================================
TRAIN_DIR = "train"
VAL_DIR = "valid"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"✅ Using device: {DEVICE}")

# ==========================================================
# 🧩 CUSTOM DATASET
# ==========================================================
class DriverDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images, self.labels = [], []
        annot_path = os.path.join(folder_path, "_annotations_cleaned.txt")

        if not os.path.exists(annot_path):
            raise FileNotFoundError(f"❌ Missing annotation file: {annot_path}")

        with open(annot_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                img_name = parts[0]
                bbox_info = parts[1].split(",")
                if len(bbox_info) < 5:
                    continue
                try:
                    x1, y1, x2, y2, cls = map(int, bbox_info)
                except ValueError:
                    continue

                img_path = os.path.join(folder_path, img_name)
                if os.path.exists(img_path):
                    self.images.append((img_path, (x1, y1, x2, y2, cls)))
                    self.labels.append(cls)

        if len(self.images) == 0:
            raise ValueError(f"❌ No valid images found in {folder_path}")

        # Automatically detect number of classes
        self.num_classes = len(set(self.labels))
        print(f"📊 Loaded {len(self.images)} samples from {folder_path}")
        print(f"📁 Detected Classes: {sorted(list(set(self.labels)))}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, (x1, y1, x2, y2, label) = self.images[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Error loading {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop region (bounding box)
        h, w, _ = image.shape
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            cropped = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        else:
            cropped = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

        if self.transform:
            cropped = self.transform(cropped)

        # sanity check: label in valid range
        if label < 0 or label >= self.num_classes:
            raise ValueError(f"❌ Invalid label {label} found in {img_path}")

        return cropped, label


# ==========================================================
# 🧠 MODEL DEFINITION
# ==========================================================
class DriverStateCNN(nn.Module):
    def __init__(self, num_classes):
        super(DriverStateCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ==========================================================
# 🔁 TRAINING & VALIDATION FUNCTIONS
# ==========================================================
def train_one_epoch(model, loader, optimizer, criterion, num_classes):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        if (labels < 0).any() or (labels >= num_classes).any():
            raise ValueError(f"Invalid label found in batch: {labels}")

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


def validate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


# ==========================================================
# 🚀 MAIN
# ==========================================================
def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    print("\n📂 Loading datasets...")
    train_ds = DriverDataset(TRAIN_DIR, transform=transform)
    val_ds = DriverDataset(VAL_DIR, transform=transform)

    NUM_CLASSES = max(train_ds.num_classes, val_ds.num_classes)
    print(f"\n🧮 Auto-detected NUM_CLASSES = {NUM_CLASSES}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DriverStateCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\n📅 Epoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, NUM_CLASSES)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"🔹 Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"🔹 Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("💾 Model saved!")

    print("\n✅ Training complete. Best Val Accuracy: {:.2f}%".format(best_acc * 100))


if __name__ == "__main__":
    main()
