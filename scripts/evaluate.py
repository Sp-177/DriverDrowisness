import os
import json
import time
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from colorama import Fore, Style

# ==========================================================
# 🔹 Imports (Match Your Folder Structure)
# ==========================================================
from scripts.cnn_model import DrowsinessCNN
from scripts.dataset_loader import DriverDrowsinessDataset

# ==========================================================
# 🔹 Evaluation Function
# ==========================================================
def evaluate_model_pro(
    model_path,
    dataset_root="dataset/test",
    img_size=128,
    num_classes=6,
    batch_size=32,
    export_json=True,
    save_misclassified=False
):
    start_time = time.time()
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
    # Load Model
    # -------------------------
    if not os.path.exists(model_path):
        print(Fore.RED + f"❌ Model file not found at {model_path}" + Style.RESET_ALL)
        return

    model = DrowsinessCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(Fore.YELLOW + "📦 Model loaded successfully. Starting evaluation..." + Style.RESET_ALL)

    # -------------------------
    # Evaluation Loop
    # -------------------------
    all_preds, all_labels = [], []
    misclassified = []

    with torch.no_grad():
        for imgs, lbls, paths in tqdm(loader, desc="🔍 Evaluating", unit="batch"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

            if save_misclassified:
                for i in range(len(preds)):
                    if preds[i] != lbls[i]:
                        misclassified.append({
                            "image": paths[i],
                            "true_label": int(lbls[i].cpu().numpy()),
                            "pred_label": int(preds[i].cpu().numpy())
                        })

    # -------------------------
    # Metrics & Results
    # -------------------------
    print(Fore.CYAN + "\n📊 Evaluation Report:" + Style.RESET_ALL)
    print(classification_report(all_labels, all_preds, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print(Fore.MAGENTA + "\n🔹 Confusion Matrix:" + Style.RESET_ALL)
    print(cm)

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    print(Fore.GREEN + f"\n✅ Overall Accuracy: {accuracy:.2f}%" + Style.RESET_ALL)

    # -------------------------
    # Timing & Export Summary
    # -------------------------
    end_time = time.time()
    eval_time = end_time - start_time
    print(Fore.BLUE + f"\n🕒 Evaluation Time: {eval_time:.2f} seconds" + Style.RESET_ALL)

    results = {
        "model_path": model_path,
        "dataset": dataset_root,
        "device": str(device),
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "evaluation_time_sec": round(eval_time, 2)
    }

    if save_misclassified:
        results["misclassified"] = misclassified[:20]  # log top 20 only

    if export_json:
        os.makedirs("results", exist_ok=True)
        json_path = os.path.join("results", "evaluation_report.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(Fore.CYAN + f"\n💾 Results saved to {json_path}" + Style.RESET_ALL)

    print(Fore.BLUE + "\n🏁 Evaluation Completed Successfully." + Style.RESET_ALL)
    return accuracy, cm


# ==========================================================
# 🔹 Main Execution
# ==========================================================
if __name__ == "__main__":
    MODEL_PATH = "models/driver_drowsiness_pro.pth"
    TEST_DATA_ROOT = "dataset/test"

    evaluate_model_pro(
        model_path=MODEL_PATH,
        dataset_root=TEST_DATA_ROOT,
        img_size=128,
        num_classes=6,
        batch_size=32,
        export_json=True,
        save_misclassified=True
    )
