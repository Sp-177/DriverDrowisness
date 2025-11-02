# ==========================================================
# 🎥 DRIVER DROWSINESS REALTIME DETECTION — TEST SCRIPT (Fixed)
# Author: Shubham Patel (NIT Raipur)
# Model: best_model.pth (trained CNN)
# ==========================================================

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------
# 🧠 Define the SAME CNN Model used in training
# -----------------------------
class DrwoisnessCNN_Pro(nn.Module):
    def __init__(self, num_classes=6):
        super(DrwoisnessCNN_Pro, self).__init__()
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

        # 👇 FIXED to match training: 128*16*16 -> 256
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# -----------------------------
# 🔧 Load Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 6  # update if your dataset uses 5 or 6
model = DrwoisnessCNN_Pro(num_classes=num_classes).to(device)

# Safe load with warning fix
weights = torch.load("best_model.pth", map_location=device, weights_only=True)
model.load_state_dict(weights)
model.eval()

# -----------------------------
# 🧾 Class Labels
# -----------------------------
class_names = ["DangerousDriving", "Distracted", "Drinking", "SafeDriving", "SleepyDriving", "Yawn"]

# -----------------------------
# 🔄 Transform for input image
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # match training size
    transforms.ToTensor(),
])

# -----------------------------
# 📸 Start Webcam Feed
# -----------------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("✅ Real-time Driver Drowsiness Detection started...")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        try:
            img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img)
                _, preds = torch.max(outputs, 1)
                label = class_names[preds.item()]

            color = (0, 255, 0) if label == "SafeDriving" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except Exception as e:
            print("⚠️ Error:", e)
            continue

    cv2.imshow("Driver Drowsiness Detection - Shubham", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
