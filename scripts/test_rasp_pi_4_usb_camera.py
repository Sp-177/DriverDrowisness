import cv2
import numpy as np
import onnxruntime as ort
import time

# --------------------------
# Load ONNX model
# --------------------------
MODEL_PATH = "models/driver_drowsiness.onnx"

# Initialize ONNX Runtime session
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --------------------------
# Haarcascade face detector
# --------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --------------------------
# Class labels (6 classes)
# --------------------------
CLASSES = [
    "Normal",
    "Yawning",
    "Eyes Closed",
    "Looking Away",
    "Phone Usage",
    "Drowsy"
]

# --------------------------
# Initialize camera
# --------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not detected. Check USB connection.")
    exit()

print("✅ Camera started. Press 'q' to quit.\n")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-8)
    prev_time = curr_time

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        face_norm = face_resized.astype(np.float32) / 255.0
        face_input = np.transpose(face_norm, (2, 0, 1))
        face_input = np.expand_dims(face_input, axis=0)

        try:
            preds = session.run([output_name], {input_name: face_input})[0]
            pred_idx = int(np.argmax(preds))
            label = CLASSES[pred_idx]
        except Exception as e:
            label = "Error"
            print("ONNX error:", e)

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 100), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display
    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
