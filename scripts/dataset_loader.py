import os, cv2, numpy as np, pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from colorama import Fore, Style
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cnn_model import DrowsinessCNN


class DriverDrowsinessDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.annotation_file = os.path.join(folder_path, "_annotations.txt")
        self.transform = transform
        rows = []

        try:
            with open(self.annotation_file, "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2: continue
                    filename, coords = parts
                    try:
                        x1, y1, x2, y2, cls = map(int, coords.split(","))
                        rows.append([filename, x1, y1, x2, y2, cls])
                    except ValueError: continue
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to read {self.annotation_file}: {e}" + Style.RESET_ALL)

        df = pd.DataFrame(rows, columns=["filename", "x1", "y1", "x2", "y2", "class"])
        self.data = df[df["filename"].apply(lambda x: isinstance(x, str) and x.strip() != "")].reset_index(drop=True)
        print(Fore.CYAN + f"📁 Loaded {len(self.data)} samples from {folder_path}" + Style.RESET_ALL)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.folder_path, row["filename"])
        label = int(row["class"])
        try:
            image = cv2.imread(img_path)
            if image is None: raise ValueError("Unreadable image")
            h, w, _ = image.shape
            x1, y1, x2, y2 = [max(0, v) for v in (row["x1"], row["y1"], row["x2"], row["y2"])]
            cropped = image[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else image
            image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(Fore.YELLOW + f"[WARN] {img_path}: {e}" + Style.RESET_ALL)
            image = Image.fromarray(np.zeros((128,128,3), dtype=np.uint8))
        if self.transform: image = self.transform(image)
        return image, label
