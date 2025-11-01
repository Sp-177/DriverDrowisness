import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from models.cnn_model import DrowsinessCNN
from dataset_loader import DriverDrowsinessDataset
from colorama import Fore, Style

def evaluate_model(model_path, dataset_root="dataset/test", img_size=128, num_classes=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(Fore.GREEN + f"🚀 Evaluating on {device}" + Style.RESET_ALL)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = DriverDrowsinessDataset(dataset_root, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = DrowsinessCNN(num_classes=num_classes, img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    print(Fore.BLUE + "\n📊 Evaluation Metrics:" + Style.RESET_ALL)
    print(classification_report(all_labels, all_preds, digits=4))
    print(Fore.MAGENTA + "🔹 Confusion Matrix:" + Style.RESET_ALL)
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate_model("models/driver_drowsiness_final.pth")
