# ==========================================================
# 📘 DRIVER DROWSINESS DETECTION — FULL TRAINING SCRIPT
# Model: DrwoisnessCNN_Pro
# Author: Shubham Patel (NIT Raipur)
# ==========================================================

import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from colorama import Fore, Style
from sklearn.metrics import classification_report, confusion_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================================
# GLOBAL CONFIGURATION
# ==========================================================
DATASET_ROOT = "dataset"
TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
VAL_PATH = os.path.join(DATASET_ROOT, "valid")
MODEL_SAVE_PATH = "models/DrwoisnessCNN_Pro.pth"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_CLASSES = 6
PATIENCE = 5
GRAD_CLIP = 1.0
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# ==========================================================
# 🧹 DATASET PREPROCESSING
# ==========================================================
def validate_annotation_line(line, folder):
    parts = line.strip().split(" ", 1)
    if len(parts) != 2:
        return None
    filename, coords = parts
    img_path = os.path.join(folder, filename)
    if not os.path.exists(img_path):
        return None
    try:
        x1, y1, x2, y2, cls = map(int, coords.split(","))
        if x2 <= x1 or y2 <= y1:
            return None
        return line.strip()
    except Exception:
        return None


def preprocess_dataset(dataset_root, max_workers=8):
    splits = ["train", "valid", "test"]
    print(Fore.CYAN + "\n🚀 Starting dataset preprocessing...\n" + Style.RESET_ALL)
    for split in splits:
        folder = os.path.join(dataset_root, split)
        annotation_path = os.path.join(folder, "_annotations.txt")
        cleaned_path = os.path.join(folder, "_annotations_cleaned.txt")

        if not os.path.exists(annotation_path):
            print(Fore.YELLOW + f"[WARN] Missing annotation file: {annotation_path}" + Style.RESET_ALL)
            continue

        backup_path = annotation_path.replace(".txt", "_backup.txt")
        if not os.path.exists(backup_path):
            os.rename(annotation_path, backup_path)
            print(Fore.BLUE + f"[INFO] Backup created: {backup_path}" + Style.RESET_ALL)

        with open(backup_path, "r") as f:
            lines = f.readlines()

        valid_rows = []
        total_lines = len(lines)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(validate_annotation_line, line, folder) for line in lines]
            for future in tqdm(as_completed(futures), total=total_lines, desc=f"Validating {split}"):
                result = future.result()
                if result:
                    valid_rows.append(result)

        with open(cleaned_path, "w") as cf:
            cf.write("\n".join(valid_rows))

        invalid_count = total_lines - len(valid_rows)
        print(
            Fore.GREEN + f"\n✅ [{split.upper()}] Cleaned successfully!" + Style.RESET_ALL
        )
        print(Fore.WHITE + f"   Total lines     : {total_lines}")
        print(Fore.WHITE + f"   Valid entries   : {len(valid_rows)}")
        print(Fore.WHITE + f"   Invalid removed : {invalid_count}")
        print(Fore.WHITE + f"   Output file     : {cleaned_path}\n" + Style.RESET_ALL)
    print(Fore.MAGENTA + "🎯 All splits processed successfully.\n" + Style.RESET_ALL)

# ==========================================================
# 📁 CUSTOM DATASET LOADER
# ==========================================================
class DriverDrowsinessDataset(Dataset):
    def __init__(self, folder_path, transform=None, target_size=(128, 128)):
        self.folder_path = folder_path
        self.annotation_file = os.path.join(folder_path, "_annotations_cleaned.txt")
        self.transform = transform
        self.target_size = target_size
        self.data = self._load_annotations()
        print(Fore.CYAN + f"📁 Loaded {len(self.data)} samples from {folder_path}" + Style.RESET_ALL)

    def _load_annotations(self):
        rows = []
        if not os.path.exists(self.annotation_file):
            print(Fore.RED + f"[ERROR] Annotation file not found: {self.annotation_file}" + Style.RESET_ALL)
            return pd.DataFrame(columns=["filename", "x1", "y1", "x2", "y2", "class"])
        with open(self.annotation_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                filename, coords = parts
                try:
                    x1, y1, x2, y2, cls = map(int, coords.split(","))
                    rows.append([filename.strip(), x1, y1, x2, y2, cls])
                except ValueError:
                    continue
        return pd.DataFrame(rows, columns=["filename", "x1", "y1", "x2", "y2", "class"]).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def _safe_crop(self, image, x1, y1, x2, y2):
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return image
        return image[y1:y2, x1:x2]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.folder_path, row["filename"])
        label = int(row["class"])
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Unreadable: {row['filename']}")
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cropped = self._safe_crop(image, row["x1"], row["y1"], row["x2"], row["y2"])
            cropped = cv2.resize(cropped, self.target_size)
            image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        except Exception:
            image = Image.fromarray(np.zeros((*self.target_size, 3), dtype=np.uint8))
        if self.transform:
            image = self.transform(image)
        return image, label

# ==========================================================
# 🧠 CNN MODEL WITH CBAM + SE ATTENTION
# ==========================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = SEBlock(channels, reduction)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.channel_att(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return x * spatial


class DrowsinessCNN_Pro(nn.Module):
    def __init__(self, num_classes=6, dropout=0.4):
        super().__init__()
        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, 32),
            CBAM(32),
            conv_block(32, 64),
            CBAM(64),
            conv_block(64, 128),
            CBAM(128),
            conv_block(128, 256),
            CBAM(256)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x)

# ==========================================================
# TRAINING + EVALUATION PIPELINE
# ==========================================================
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return 100 * correct / total, val_loss / len(loader)


def plot_confusion_matrix(labels, preds, num_classes):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — DrwoisnessCNN_Pro ({num_classes} Classes)")
    plt.tight_layout()
    plt.show()


def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(Fore.GREEN + f"🚀 Training on device: {device}" + Style.RESET_ALL)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_set = DriverDrowsinessDataset(TRAIN_PATH, transform)
    val_set = DriverDrowsinessDataset(VAL_PATH, transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = DrowsinessCNN_Pro(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                              steps_per_epoch=len(train_loader), epochs=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_acc, epochs_no_improve = 0.0, 0

    for epoch in range(EPOCHS):
        print(Fore.MAGENTA + f"\n📦 Epoch {epoch + 1}/{EPOCHS}" + Style.RESET_ALL)
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc="Training", unit="batch"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(Fore.GREEN + f"✅ Train Loss: {running_loss/len(train_loader):.4f} | Accuracy: {train_acc:.2f}%" + Style.RESET_ALL)

        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        print(Fore.CYAN + f"🔹 Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%" + Style.RESET_ALL)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(Fore.YELLOW + f"💾 Model saved — Best Accuracy: {best_val_acc:.2f}%" + Style.RESET_ALL)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(Fore.RED + f"🛑 Early stopping after {PATIENCE} epochs without improvement." + Style.RESET_ALL)
                break

    print(Fore.CYAN + f"\n🏁 Training finished — Best Validation Accuracy: {best_val_acc:.2f}%" + Style.RESET_ALL)

    # Final evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(Fore.GREEN + "\n📈 Classification Report:" + Style.RESET_ALL)
    print(classification_report(all_labels, all_preds, digits=3))
    plot_confusion_matrix(all_labels, all_preds, NUM_CLASSES)

# ==========================================================
# MAIN ENTRY
# ==========================================================
if __name__ == "__main__":
    preprocess_dataset(DATASET_ROOT)
    train_and_evaluate()
