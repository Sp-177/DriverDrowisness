import os
import cv2
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def preprocess_dataset(dataset_root):
    """
    Validate and clean all annotation files.
    - Checks missing images
    - Removes invalid boxes
    - Creates 'cleaned_annotations.txt' for each split
    """
    splits = ["train", "valid", "test"]
    for split in splits:
        folder = os.path.join(dataset_root, split)
        annotation_path = os.path.join(folder, "_annotations.txt")
        cleaned_path = os.path.join(folder, "_annotations_cleaned.txt")

        if not os.path.exists(annotation_path):
            print(Fore.YELLOW + f"[WARN] Missing annotation file: {annotation_path}" + Style.RESET_ALL)
            continue

        valid_rows = []
        with open(annotation_path, "r") as f:
            for line in tqdm(f, desc=f"Checking {split}"):
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                filename, coords = parts
                img_path = os.path.join(folder, filename)
                if not os.path.exists(img_path):
                    print(Fore.RED + f"[MISSING] {img_path}" + Style.RESET_ALL)
                    continue
                try:
                    x1, y1, x2, y2, cls = map(int, coords.split(","))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    valid_rows.append(line.strip())
                except Exception:
                    continue

        with open(cleaned_path, "w") as cf:
            cf.write("\n".join(valid_rows))

        print(Fore.GREEN + f"✅ Cleaned {split}: {len(valid_rows)} valid entries saved to {cleaned_path}" + Style.RESET_ALL)


if __name__ == "__main__":
    preprocess_dataset("dataset")
