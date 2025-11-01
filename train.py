import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from colorama import Fore, Style

from models.cnn_model import DrowsinessCNN
from scripts.dataset_loader import DriverDrowsinessDataset
from scripts.evaluate import evaluate_model

# ==========================================================
# CONFIG
# ==========================================================
DATASET_ROOT = "dataset/train"
VALIDATION_ROOT = "dataset/valid"
MODEL_SAVE_PATH = "models/driver_drowsiness_final.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 6


# ==========================================================
# TRAIN FUNCTION
# ==========================================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(Fore.GREEN + f"🚀 Training on {device}" + Style.RESET_ALL)

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load dataset
    train_set = DriverDrowsinessDataset(DATASET_ROOT, transform)
    val_set = DriverDrowsinessDataset(VALIDATION_ROOT, transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model, loss, optimizer
    model = DrowsinessCNN(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    # ==========================================================
    # EPOCH LOOP
    # ==========================================================
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0

        print(Fore.MAGENTA + f"\n📦 Epoch {epoch+1}/{EPOCHS}" + Style.RESET_ALL)
        for imgs, labels in tqdm(train_loader, desc="Training", unit="batch"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(Fore.GREEN + f"✅ Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%" + Style.RESET_ALL)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(Fore.CYAN + f"🔹 Validation Accuracy: {val_acc:.2f}%" + Style.RESET_ALL)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(Fore.YELLOW + f"💾 Model saved to {MODEL_SAVE_PATH} (Best Acc: {best_val_acc:.2f}%)" + Style.RESET_ALL)

    print(Fore.CYAN + f"\n🏁 Training complete. Best Validation Accuracy: {best_val_acc:.2f}%" + Style.RESET_ALL)
    evaluate_model(MODEL_SAVE_PATH, dataset_root="dataset/test")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    train_model()
