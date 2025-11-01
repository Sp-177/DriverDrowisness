import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
from colorama import Fore, Style


# ==========================================
# CONFIGURATION
# ==========================================
DATASET_ROOT = "D:/CSE/Project/DriverDrowisness/dataset"  # ⚠️ change to your dataset path
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 6
MODEL_SAVE_PATH = "driver_drowsiness_final.pth"

CLASS_NAMES = [
    "DangerousDriving",
    "Distracted",
    "Drinking",
    "SafeDriving",
    "SleepyDriving",
    "Yawn"
]

# ==========================================
# DATASET CLASS
# ==========================================
class DriverDrowsinessDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.annotation_file = os.path.join(folder_path, "_annotations.txt")
        self.transform = transform
        data_rows = []

        try:
            with open(self.annotation_file, "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2:
                        continue
                    filename, coords = parts
                    try:
                        x1, y1, x2, y2, cls = map(int, coords.split(","))
                        data_rows.append([filename, x1, y1, x2, y2, cls])
                    except ValueError:
                        continue
        except Exception as e:
            print(Fore.RED + f"[ERROR] Could not read {self.annotation_file}: {e}" + Style.RESET_ALL)

        df = pd.DataFrame(data_rows, columns=["filename", "x1", "y1", "x2", "y2", "class"])
        df = df[df["filename"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
        self.data = df.reset_index(drop=True)

        print(Fore.CYAN + f"📁 Loaded {len(self.data)} samples from {folder_path}" + Style.RESET_ALL)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.folder_path, row["filename"])
        label = int(row["class"]) if str(row["class"]).isdigit() else 0

        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("Image not readable")

            h, w, _ = image.shape
            x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                cropped = image

            image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        except Exception as e:
            print(Fore.YELLOW + f"[WARN] Problem loading {img_path}: {e}" + Style.RESET_ALL)
            dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            image = Image.fromarray(dummy)

        if self.transform:
            image = self.transform(image)
        return image, label


# ==========================================
# MODEL DEFINITION (Light CNN)
# ==========================================
class DrowsinessCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(DrowsinessCNN, self).__init__()
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
            nn.Dropout(0.4),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ==========================================
# TRAINING FUNCTION
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(Fore.GREEN + f"🚀 Training on: {device}" + Style.RESET_ALL)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_set = DriverDrowsinessDataset(os.path.join(DATASET_ROOT, "train"), transform)
    valid_set = DriverDrowsinessDataset(os.path.join(DATASET_ROOT, "valid"), transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(Fore.CYAN + f"🧩 Train samples: {len(train_set)} | Validation samples: {len(valid_set)}" + Style.RESET_ALL)

    model = DrowsinessCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        print(Fore.MAGENTA + f"\n📦 Epoch {epoch+1}/{EPOCHS}" + Style.RESET_ALL)
        for images, labels in tqdm(train_loader, desc="Training", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        print(Fore.GREEN + f"✅ Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%" + Style.RESET_ALL)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(Fore.BLUE + f"🔹 Validation Accuracy: {val_acc:.2f}%" + Style.RESET_ALL)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(Fore.CYAN + f"\n💾 Model saved to {MODEL_SAVE_PATH}" + Style.RESET_ALL)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    train_model()
