import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from colorama import Fore, Style

# =======================================================
# CONFIG
# =======================================================
MODEL_PATH = "models/driver_drowsiness_final.pth"
IMG_SIZE = 128
CLASS_NAMES = [
    "DangerousDriving",
    "Distracted",
    "Drinking",
    "SafeDriving",
    "SleepyDriving",
    "Yawn"
]

# =======================================================
# MODEL
# =======================================================
class DrowsinessCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(DrowsinessCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# =======================================================
# LOAD MODEL
# =======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrowsinessCNN(num_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(Fore.GREEN + f"✅ Model loaded from {MODEL_PATH}" + Style.RESET_ALL)

# =======================================================
# TRANSFORMS
# =======================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# =======================================================
# FACE DETECTOR
# =======================================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# =======================================================
# CAMERA
# =======================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(Fore.RED + "❌ Could not open webcam!" + Style.RESET_ALL)
    exit()

print(Fore.CYAN + "🎥 Starting detection... Press 'q' to quit." + Style.RESET_ALL)

# =======================================================
# LOOP
# =======================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print(Fore.YELLOW + "[WARN] Frame not captured!" + Style.RESET_ALL)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Crop face
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        # Convert and preprocess
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = outputs.max(1)
            label = CLASS_NAMES[predicted.item()]

        # Color code based on alert
        if label in ["SleepyDriving", "Yawn"]:
            color = (0, 0, 255)   # 🔴 red warning
        elif label in ["Drinking", "Distracted"]:
            color = (0, 165, 255) # 🟠 orange warning
        else:
            color = (0, 255, 0)   # 🟢 safe

        # Draw box + label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(Fore.BLUE + "🛑 Camera closed." + Style.RESET_ALL)
