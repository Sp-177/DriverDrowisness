import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from colorama import Fore, Style
from models.cnn_model import DrowsinessCNN
from scripts.dataset_loader import DriverDrowsinessDataset

DATASET_ROOT = "dataset"
IMG_SIZE, BATCH_SIZE, EPOCHS, LR, NUM_CLASSES = 128, 32, 20, 1e-3, 6
MODEL_SAVE_PATH = "models/driver_drowsiness_final.pth"

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(Fore.GREEN + f"🚀 Using {device}" + Style.RESET_ALL)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_set = DriverDrowsinessDataset(f"{DATASET_ROOT}/train", transform)
    valid_set = DriverDrowsinessDataset(f"{DATASET_ROOT}/valid", transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DrowsinessCNN(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train(); running_loss=correct=total=0
        print(Fore.MAGENTA + f"\n📦 Epoch {epoch+1}/{EPOCHS}" + Style.RESET_ALL)
        for imgs, lbls in tqdm(train_loader, desc="Training", unit="batch"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward(); optimizer.step()
            running_loss += loss.item()
            _, pred = out.max(1); total += lbls.size(0); correct += pred.eq(lbls).sum().item()
        print(Fore.GREEN + f"✅ Loss: {running_loss/len(train_loader):.4f}, Train Acc: {100*correct/total:.2f}%" + Style.RESET_ALL)

        # Validation
        model.eval(); val_correct=val_total=0
        with torch.no_grad():
            for imgs, lbls in valid_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                _, pred = out.max(1)
                val_total += lbls.size(0); val_correct += pred.eq(lbls).sum().item()
        print(Fore.BLUE + f"🔹 Validation Accuracy: {100*val_correct/val_total:.2f}%" + Style.RESET_ALL)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(Fore.CYAN + f"💾 Model saved at {MODEL_SAVE_PATH}" + Style.RESET_ALL)

if __name__ == "__main__":
    train_model()
