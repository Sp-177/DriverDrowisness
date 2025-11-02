import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style

# Add parent directory to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.cnn_model import DrowsinessCNN
from scripts.dataset_loader import DriverDrowsinessDataset


# ==========================================================
# CONFIGURATION
# ==========================================================
DATASET_ROOT = "dataset/train"
VALIDATION_ROOT = "dataset/valid"
MODEL_SAVE_PATH = "models/DrwoisnessCNN_Pro.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_CLASSES = 6
PATIENCE = 5
GRAD_CLIP = 1.0  # gradient clipping threshold
SEED = 42

# ==========================================================
# SET SEED FOR REPRODUCIBILITY
# ==========================================================
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


# ==========================================================
# TRAIN + EVALUATE FUNCTION
# ==========================================================
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(Fore.GREEN + f"🚀 Training on device: {device}" + Style.RESET_ALL)

    # ---------------- Transforms ----------------
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # ---------------- Data ----------------
    train_set = DriverDrowsinessDataset(DATASET_ROOT, transform)
    val_set = DriverDrowsinessDataset(VALIDATION_ROOT, transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---------------- Model, Loss, Optimizer ----------------
    model = DrowsinessCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                              steps_per_epoch=len(train_loader), epochs=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_acc = 0.0
    epochs_no_improve = 0

    # ==========================================================
    # TRAINING LOOP
    # ==========================================================
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

        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(Fore.GREEN + f"✅ Train Loss: {avg_loss:.4f} | Accuracy: {train_acc:.2f}%" + Style.RESET_ALL)

        # ---------------- Validation ----------------
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        print(Fore.CYAN + f"🔹 Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%" + Style.RESET_ALL)

        # ---------------- Save Best ----------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(Fore.YELLOW + f"💾 Model improved & saved — Best Accuracy: {best_val_acc:.2f}%" + Style.RESET_ALL)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # ---------------- Early Stopping ----------------
        if epochs_no_improve >= PATIENCE:
            print(Fore.RED + f"🛑 Early stopping after {PATIENCE} epochs without improvement." + Style.RESET_ALL)
            break

    print(Fore.CYAN + f"\n🏁 Training completed — Best Validation Accuracy: {best_val_acc:.2f}%" + Style.RESET_ALL)

    # ==========================================================
    # FINAL EVALUATION
    # ==========================================================
    print(Fore.YELLOW + "\n📊 Generating Final Evaluation Report..." + Style.RESET_ALL)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Evaluating", unit="batch"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(Fore.GREEN + "\n📈 Classification Report:" + Style.RESET_ALL)
    print(classification_report(all_labels, all_preds, digits=3))

    plot_confusion_matrix(all_labels, all_preds, num_classes=NUM_CLASSES)


# ==========================================================
# EVALUATION HELPER
# ==========================================================
def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    val_acc = 100 * correct / total
    return val_acc, val_loss / len(loader)


# ==========================================================
# CONFUSION MATRIX VISUALIZATION
# ==========================================================
def plot_confusion_matrix(labels, preds, num_classes):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — DrwoisnessCNN_Pro ({num_classes} Classes)")
    plt.tight_layout()
    plt.show()


# ==========================================================
# MAIN ENTRY
# ==========================================================
if __name__ == "__main__":
    train_and_evaluate()
