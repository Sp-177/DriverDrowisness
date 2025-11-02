import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from colorama import Fore, Style

# Add parent directory to import scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.cnn_model import DrowsinessCNN_Best
from scripts.dataset_loader import DriverDrowsinessDataset


# ==========================================================
# CONFIGURATION
# ==========================================================
DATASET_ROOT = "dataset/train"
VALIDATION_ROOT = "dataset/valid"
MODEL_SAVE_PATH = "models/driver_drowsiness_final.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
NUM_CLASSES = 6
PATIENCE = 5  # early stopping patience


# ==========================================================
# TRAIN FUNCTION
# ==========================================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(Fore.GREEN + f"🚀 Training started on: {device}" + Style.RESET_ALL)

    # ---------------- Data Augmentation ----------------
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # ---------------- Dataset & Dataloader ----------------
    train_set = DriverDrowsinessDataset(DATASET_ROOT, transform)
    val_set = DriverDrowsinessDataset(VALIDATION_ROOT, transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---------------- Model, Loss, Optimizer ----------------
    model = DrowsinessCNN_Best(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
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
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(Fore.GREEN + f"✅ Train Loss: {avg_loss:.4f} | Accuracy: {train_acc:.2f}%" + Style.RESET_ALL)

        # ---------------- Validation ----------------
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)
        print(Fore.CYAN + f"🔹 Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%" + Style.RESET_ALL)

        # ---------------- Checkpoint ----------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(Fore.YELLOW + f"💾 Model saved — Best Accuracy: {best_val_acc:.2f}%" + Style.RESET_ALL)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # ---------------- Early Stopping ----------------
        if epochs_no_improve >= PATIENCE:
            print(Fore.RED + f"🛑 Early stopping triggered after {PATIENCE} epochs without improvement." + Style.RESET_ALL)
            break

    print(Fore.CYAN + f"\n🏁 Training complete — Best Validation Accuracy: {best_val_acc:.2f}%" + Style.RESET_ALL)
    print(Fore.YELLOW + f"📦 Model weights saved at: {MODEL_SAVE_PATH}" + Style.RESET_ALL)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    train_model()
