import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from colorama import Fore, Style

# Add parent path for flexible import handling
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class DriverDrowsinessDataset(Dataset):
    """
    Custom PyTorch Dataset for Driver Drowsiness Detection.

    Expects a folder structure:
    ├── dataset/
    │   ├── _annotations.txt
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...

    Each line in _annotations.txt:
    filename x1,y1,x2,y2,class
    """

    def __init__(self, folder_path, transform=None, target_size=(128, 128)):
        self.folder_path = folder_path
        self.annotation_file = os.path.join(folder_path, "_annotations.txt")
        self.transform = transform
        self.target_size = target_size

        self.data = self._load_annotations()
        print(
            Fore.CYAN
            + f"📁 Loaded {len(self.data)} valid samples from {folder_path}"
            + Style.RESET_ALL
        )

    # ----------------------------------------------------------------------
    def _load_annotations(self):
        """Reads annotation file and returns a clean pandas DataFrame."""
        rows = []
        if not os.path.exists(self.annotation_file):
            print(
                Fore.RED
                + f"[ERROR] Annotation file not found: {self.annotation_file}"
                + Style.RESET_ALL
            )
            return pd.DataFrame(columns=["filename", "x1", "y1", "x2", "y2", "class"])

        try:
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
                        print(
                            Fore.YELLOW
                            + f"[WARN] Invalid coordinate format in line: {line.strip()}"
                            + Style.RESET_ALL
                        )
                        continue
        except Exception as e:
            print(
                Fore.RED
                + f"[ERROR] Failed to read {self.annotation_file}: {e}"
                + Style.RESET_ALL
            )

        df = pd.DataFrame(rows, columns=["filename", "x1", "y1", "x2", "y2", "class"])
        df = df[df["filename"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
        return df.reset_index(drop=True)

    # ----------------------------------------------------------------------
    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.data)

    # ----------------------------------------------------------------------
    def _safe_crop(self, image, x1, y1, x2, y2):
        """Safely crop image using coordinates (with boundary clipping)."""
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return image
        return image[y1:y2, x1:x2]

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        """Loads one image and its label."""
        row = self.data.iloc[idx]
        img_path = os.path.join(self.folder_path, row["filename"])
        label = int(row["class"])

        try:
            image = cv2.imread(img_path)

            if image is None:
                raise FileNotFoundError(f"Unreadable or missing image: {row['filename']}")

            # Convert grayscale → RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Crop ROI if bounding box valid
            cropped = self._safe_crop(
                image, row["x1"], row["y1"], row["x2"], row["y2"]
            )

            # Resize to target size if needed
            if self.target_size:
                cropped = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)

            # Convert to PIL for PyTorch transforms
            image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        except Exception as e:
            print(Fore.YELLOW + f"[WARN] {img_path}: {e}" + Style.RESET_ALL)
            # Return blank fallback image on failure
            image = Image.fromarray(np.zeros((*self.target_size, 3), dtype=np.uint8))

        # Apply user-defined transform
        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------------------------------------------------
# ✅ Example usage:
if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    dataset = DriverDrowsinessDataset(
        folder_path="D:/CSE/Project/DriverDrowsiness/dataset",
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"Total Samples: {len(dataset)}")
    for imgs, labels in dataloader:
        print(f"Batch shape: {imgs.shape}, Labels: {labels[:8].tolist()}")
        break
