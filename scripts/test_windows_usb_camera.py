import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from colorama import Fore, Style
import time
import winsound  # for buzzer alert on Windows

# =======================================================
# CONFIG
# =======================================================
MODEL_PATH = "models/DrwoisnessCNN_Pro.pth"
IMG_SIZE = 128
CLASS_NAMES = [
    "DangerousDriving",
    "Distracted",
    "Drinking",
    "SafeDriving",
    "SleepyDriving",
    "Yawn"
]

ALERT_FREQ = 1200  # Hz
ALERT_DURATION = 600  # ms

# =======================================================
# MODEL DEFINITION
# =======================================================
class DrwoisnessCNN_Pro(nn.Module):
    def __init__(self, num_classes=6):
        super(DrwoisnessCNN_Pro, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * (IMG_SIZE // 16) * (IMG_SIZE // 16), 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# =======================================================
# LOAD MODEL
# =======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrwoisnessCNN_Pro(num_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(Fore.GREEN + f"✅ Model loaded successfully from {MODEL_PATH}" + Style.RESET_ALL)

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
# CAMERA INIT
# =======================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print(Fore.RED + "❌ Could not open webcam. Check USB connection." + Style.RESET_ALL)
    exit()

print(Fore.CYAN + "🎥 Starting Driver Drowsiness Detection (Pro Version)" + Style.RESET_ALL)
print(Fore.YELLOW + "Press 'Q' to quit." + Style.RESET_ALL)

# =======================================================
# REAL-TIME LOOP
# =======================================================
prev_label = ""
alert_active = False
alert_cooldown = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print(Fore.RED + "[ERROR] Frame capture failed!" + Style.RESET_ALL)
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        # Preprocess
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
            label = CLASS_NAMES[predicted.item()]
            conf = confidence.item()

        # Determine color and alert
        if label in ["SleepyDriving", "Yawn"]:
            color = (0, 0, 255)
            status_text = "⚠️ DROWSY - Take a Break!"
            if not alert_active and time.time() - alert_cooldown > 3:
                winsound.Beep(ALERT_FREQ, ALERT_DURATION)
                alert_active = True
                alert_cooldown = time.time()
        elif label in ["Distracted", "Drinking"]:
            color = (0, 165, 255)
            status_text = "⚠️ DISTRACTED - Stay Focused!"
            alert_active = False
        else:
            color = (0, 255, 0)
            status_text = "✅ SAFE DRIVING"
            alert_active = False

        # Draw UI
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.rectangle(frame, (10, 20), (400, 50), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imshow("Driver Drowsiness Detection (Pro)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(Fore.BLUE + "🛑 Camera closed. Detection stopped." + Style.RESET_ALL)
