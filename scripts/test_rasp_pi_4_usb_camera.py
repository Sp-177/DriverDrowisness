import cv2
import torch
import numpy as np
import onnxruntime as ort
from models.cnn_model import DrowsinessCNN
from torchvision import transforms

# -----------------------
# CONFIG
# -----------------------
IMG_SIZE = 128
NUM_CLASSES = 6
LABELS = ["Awake", "Sleepy", "Drowsy", "Yawning", "Blinking", "Closed Eyes"]
MODEL_PATH_PTH = "models/driver_drowsiness_final.pth"
MODEL_PATH_ONNX = "models/driver_drowsiness.onnx"

# -----------------------
# LOAD MODEL
# -----------------------
use_onnx = False
model = None

try:
    # Try ONNX first
    ort_session = ort.InferenceSession(MODEL_PATH_ONNX)
    use_onnx = True
    print("✅ Loaded ONNX model.")
except Exception as e:
    print("⚠️ ONNX not found or failed, using PyTorch model instead.")
    model = DrowsinessCNN(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH_PTH, map_location="cpu"))
    model.eval()
    print("✅ Loaded PyTorch model.")

# -----------------------
# PREPROCESS FUNCTION
# -----------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)
    return img

# -----------------------
# INFERENCE FUNCTION
# -----------------------
def predict(frame):
    img_tensor = preprocess(frame)

    if use_onnx:
        ort_inputs = {"input": img_tensor.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        output = torch.tensor(ort_outs[0])
    else:
        with torch.no_grad():
            output = model(img_tensor)

    pred = torch.argmax(output, dim=1).item()
    return LABELS[pred]

# -----------------------
# START CAMERA
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

# Optional: use Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("🎥 Starting real-time drowsiness detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        state = predict(face)

        color = (0, 255, 0) if state == "Awake" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, state, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Stream closed.")
