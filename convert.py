# ==========================================================
# 🧩 DRIVER DROWSINESS DETECTION — ONNX EXPORT SCRIPT
# Model: DrwoisnessCNN_Pro (trained in PyTorch)
# Author: Shubham Patel (NIT Raipur)
# ==========================================================

import torch
import torch.nn as nn

# ==========================================================
# 🧠 MODEL DEFINITION (must match training exactly)
# ==========================================================
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

        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ==========================================================
# ⚙️ CONFIGURATION
# ==========================================================
MODEL_PATH = "best_model.pth"
ONNX_PATH = "driver_drowsiness.onnx"
IMG_SIZE = 128
NUM_CLASSES = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 🔄 LOAD MODEL
# ==========================================================
print(f"🔹 Loading model from: {MODEL_PATH}")
model = DrwoisnessCNN_Pro(num_classes=NUM_CLASSES).to(device)

# Safe weight loading
weights = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(weights)
model.eval()

# ==========================================================
# 🧪 DUMMY INPUT
# ==========================================================
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

# ==========================================================
# 🔁 EXPORT TO ONNX
# ==========================================================
print("⚙️ Converting model to ONNX format...")

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"✅ Successfully exported model to {ONNX_PATH}")

# ==========================================================
# 🧾 OPTIONAL: VERIFY EXPORT
# ==========================================================
try:
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("🧠 ONNX model structure verified successfully!")
except Exception as e:
    print("⚠️ ONNX verification skipped or failed:", e)
