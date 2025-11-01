import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ==============================
# CONFIG
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
MODEL_PATH = "checkpoints/face_model.pth"
VALID_DIR = "dataset/valid"
ANNOTATION_PATH = os.path.join(VALID_DIR, "_annotations.txt")

class_names = [
    "DangerousDriving",
    "Distracted",
    "Drinking",
    "SafeDriving",
    "SleepyDriving",
    "Yawn"
]

# ==============================
# LOAD MODEL
# ==============================
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# LOAD VALIDATION SAMPLES
# ==============================
images, labels = [], []
with open(ANNOTATION_PATH, 'r') as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    if len(parts) < 2:
        continue
    img_name = parts[0]
    coords = parts[1].split(',')
    if len(coords) != 5:
        continue
    x_min, y_min, x_max, y_max, class_id = map(float, coords)
    img_path = os.path.join(VALID_DIR, img_name)
    if not os.path.exists(img_path):
        continue

    image = Image.open(img_path).convert("RGB")
    image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
    image = transform(image)
    images.append(image)
    labels.append(int(class_id))

# ==============================
# VALIDATION LOOP
# ==============================
print(f"🔍 Validating {len(images)} images...")

X = torch.stack(images).to(DEVICE)
y_true = torch.tensor(labels).to(DEVICE)

with torch.no_grad():
    outputs = model(X)
    preds = torch.argmax(outputs, dim=1)

y_true_cpu = y_true.cpu().numpy()
y_pred_cpu = preds.cpu().numpy()

# ==============================
# RESULTS
# ==============================
print("\n✅ Validation Complete!\n")
print(classification_report(y_true_cpu, y_pred_cpu, target_names=class_names))
print("Confusion Matrix:\n", confusion_matrix(y_true_cpu, y_pred_cpu))
