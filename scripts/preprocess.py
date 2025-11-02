import os
import cv2
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def validate_annotation_line(line, folder):
    """
    Validate a single annotation line.
    Returns valid line if okay, otherwise None.
    """
    parts = line.strip().split(" ", 1)
    if len(parts) != 2:
        return None

    filename, coords = parts
    img_path = os.path.join(folder, filename)

    # Check if image exists
    if not os.path.exists(img_path):
        return None

    try:
        x1, y1, x2, y2, cls = map(int, coords.split(","))
        if x2 <= x1 or y2 <= y1:
            return None
        return line.strip()
    except Exception:
        return None


def preprocess_dataset(dataset_root, max_workers=8):
    """
    Validate and clean all annotation files efficiently.
    - Checks missing images
    - Removes invalid boxes
    - Creates '_annotations_cleaned.txt' for each split
    - Multithreaded validation for faster execution
    """

    splits = ["train", "valid", "test"]
    print(Fore.CYAN + "\n🚀 Starting dataset preprocessing...\n" + Style.RESET_ALL)

    for split in splits:
        folder = os.path.join(dataset_root, split)
        annotation_path = os.path.join(folder, "_annotations.txt")
        cleaned_path = os.path.join(folder, "_annotations_cleaned.txt")

        if not os.path.exists(annotation_path):
            print(Fore.YELLOW + f"[WARN] Missing annotation file: {annotation_path}" + Style.RESET_ALL)
            continue

        # Backup original
        backup_path = annotation_path.replace(".txt", "_backup.txt")
        if not os.path.exists(backup_path):
            os.rename(annotation_path, backup_path)
            print(Fore.BLUE + f"[INFO] Backup created: {backup_path}" + Style.RESET_ALL)

        with open(backup_path, "r") as f:
            lines = f.readlines()

        valid_rows = []
        total_lines = len(lines)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(validate_annotation_line, line, folder) for line in lines]

            for future in tqdm(as_completed(futures), total=total_lines, desc=f"Validating {split}"):
                result = future.result()
                if result:
                    valid_rows.append(result)

        with open(cleaned_path, "w") as cf:
            cf.write("\n".join(valid_rows))

        invalid_count = total_lines - len(valid_rows)
        print(
            Fore.GREEN
            + f"\n✅ [{split.upper()}] Cleaned successfully!"
            + Style.RESET_ALL
        )
        print(Fore.WHITE + f"   Total lines     : {total_lines}")
        print(Fore.WHITE + f"   Valid entries   : {len(valid_rows)}")
        print(Fore.WHITE + f"   Invalid removed : {invalid_count}")
        print(Fore.WHITE + f"   Output file     : {cleaned_path}\n" + Style.RESET_ALL)

    print(Fore.MAGENTA + "🎯 All splits processed successfully.\n" + Style.RESET_ALL)


if __name__ == "__main__":
    dataset_path = "dataset"
    preprocess_dataset(dataset_path)
