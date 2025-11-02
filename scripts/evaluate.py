import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from colorama import Fore, Style

# ==========================================================
# IMPORTS (Aligned with Training Script)
# ==========================================================
from scripts.cnn_model import DrowsinessCNN_Best
from scripts.dataset_loader import DriverDrowsinessDataset

# ==========================================================
# EVALUATION FUNCTION
# ==========================================================
def evaluate_model(model_path, dataset_root="dataset/test", img_size=128, num_classes=6, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(Fore.GREEN + f"\n🚀 Evaluating Model on {device}" + Style.RESET_ALL)

    # -------------------------
    # Data Preprocessing
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = DriverDrowsinessDataset(dataset_root, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------
    # Load Trained Model
    # -------------------------
    if not os.path.exists(model_path):
        print(Fore.RED + f"❌ Model file not found at {model_path}" + Style.RESET_ALL)
        return

    model = DrowsinessCNN_Best(num_classes=num_classes, img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(Fore.YELLOW + "📦 Model loaded successfully. Starting evaluation..." + Style.RESET_ALL)

    # -------------------------
    # Evaluation Loop
    # -------------------------
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="🔍 Evaluating", unit="batch"):
            imgs, lbls = imgs.to(device), lbls.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    # -------------------------
    # Metrics & Results
    # -------------------------
    print(Fore.CYAN + "\n📊 Evaluation Report:" + Style.RESET_ALL)
    print(classification_report(all_labels, all_preds, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print(Fore.MAGENTA + "\n🔹 Confusion Matrix:" + Style.RESET_ALL)
    print(cm)

    # -------------------------
    # Accuracy Calculation
    # -------------------------
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    print(Fore.GREEN + f"\n✅ Overall Accuracy: {accuracy:.2f}%" + Style.RESET_ALL)

    print(Fore.BLUE + "\n🏁 Evaluation Completed Successfully." + Style.RESET_ALL)

    return accuracy, cm


# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    MODEL_PATH = "models/driver_drowsiness_final.pth"
    TEST_DATA_ROOT = "dataset/test"

    evaluate_model(
        model_path=MODEL_PATH,
        dataset_root=TEST_DATA_ROOT,
        img_size=128,
        num_classes=6,
        batch_size=32
    )
