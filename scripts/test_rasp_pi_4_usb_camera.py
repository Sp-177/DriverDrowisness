import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import time

# =======================================================
# CONFIG
# =======================================================
MODEL_PATH = "models/DrwoisnessCNN_Pro.onnx"
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
# LOAD MODEL (ONNX)
# =======================================================
print("[INFO] Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print(f"[OK] Model loaded: {MODEL_PATH}")

# =======================================================
# PREPROCESS FUNCTION
# =======================================================
def preprocess(image_pil):
    image_pil = image_pil.resize((IMG_SIZE, IMG_SIZE))
    img = np.asarray(image_pil).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # normalize [-1,1]
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)
    return img

# =======================================================
# CAMERA SETUP (USB CAM)
# =======================================================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("[ERROR] Cannot access USB camera. Check connection.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("[INFO] Camera ready. Press 'q' to quit.")

# =======================================================
# MAIN LOOP
# =======================================================
last_alert = 0
ALERT_COOLDOWN = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame capture failed!")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        img_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = preprocess(img_pil)

        # Run inference
        outputs = session.run(None, {input_name: img_tensor})[0]
        probs = np.exp(outputs) / np.sum(np.exp(outputs))
        pred_idx = np.argmax(probs)
        label = CLASS_NAMES[pred_idx]
        conf = probs[0][pred_idx]

        # Alert & color logic
        if label in ["SleepyDriving", "Yawn"]:
            color = (0, 0, 255)
            status_text = f"⚠️ DROWSY ({conf*100:.1f}%)"
            if time.time() - last_alert > ALERT_COOLDOWN:
                print("[ALERT] Drowsiness detected!")
                last_alert = time.time()
        elif label in ["Distracted", "Drinking"]:
            color = (0, 165, 255)
            status_text = f"⚠️ DISTRACTED ({conf*100:.1f}%)"
        else:
            color = (0, 255, 0)
            status_text = f"✅ SAFE ({conf*100:.1f}%)"

        # Draw rectangle + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Status bar
        cv2.rectangle(frame, (10, 20), (380, 50), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imshow("Drowsiness Detection (ONNX-RPi)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera closed.")
