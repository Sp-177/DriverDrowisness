# ==========================================================
# 🎥 REAL-TIME DRIVER DROWSINESS DETECTION (USB CAMERA)
# Model: DrwoisnessCNN_Pro
# Author: Shubham Patel (NIT Raipur)
# ==========================================================

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from time import time

# ==========================================================
# 🧠 MODEL DEFINITION (exactly same as training)
# ==========================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = SEBlock(channels, reduction)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.channel_att(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return x * spatial


class DrowsinessCNN_Pro(nn.Module):
    def __init__(self, num_classes=6, dropout=0.4):
        super().__init__()
        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, 32),
            CBAM(32),
            conv_block(32, 64),
            CBAM(64),
            conv_block(64, 128),
            CBAM(128),
            conv_block(128, 256),
            CBAM(256)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x)

# ==========================================================
# ⚙️ SETUP
# ==========================================================
MODEL_PATH = "models/DrwoisnessCNN_Pro.pth"   # trained model path
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = DrowsinessCNN_Pro(num_classes=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"✅ Model loaded successfully on {DEVICE}")

# EXACT CLASS NAMES (6)
CLASS_NAMES = [
    "DangerousDriving","Distracted",

"Drinking",
"SafeDriving",
"SleepyDriving",
"Yawn "
]

# Image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==========================================================
# 🎥 REAL-TIME INFERENCE LOOP
# ==========================================================
cap = cv2.VideoCapture(0)   # 0 = USB webcam
if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

prev_time = time()
fps = 0

print("🎬 Starting real-time detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        if roi_color.size == 0:
            continue

        # Preprocess
        img = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(DEVICE)

        # Prediction
        with torch.no_grad():
            outputs = model(img)
            _, pred = torch.max(outputs, 1)
            label = CLASS_NAMES[pred.item()]

        # Visual color logic
        if label.lower().startswith("neutral"):
            color = (0, 255, 0)
        elif "sleep" in label.lower() or "yawn" in label.lower():
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        # Draw bounding box & text
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # FPS
    curr_time = time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Driver Drowsiness Detection — DrwoisnessCNN_Pro", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Detection stopped.")
