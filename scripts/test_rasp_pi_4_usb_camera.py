# ==========================================================
# 🚗 DRIVER DROWSINESS DETECTION — Raspberry Pi 4 (Live)
# Model: driver_drowisness.onnx
# Author: Shubham Patel (NIT Raipur)
# ==========================================================

import cv2
import numpy as np
import onnxruntime as ort
import RPi.GPIO as GPIO
import time
from datetime import datetime

# ==========================================================
# 🔧 GPIO SETUP
# ==========================================================
LED_GREEN = 17   # Safe
LED_YELLOW = 27  # Distracted / Yawn
LED_RED = 22     # Sleepy / Dangerous
BUZZER = 23      # Buzzer

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in [LED_GREEN, LED_YELLOW, LED_RED, BUZZER]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# ==========================================================
# 🧩 ONNX MODEL LOADING
# ==========================================================
MODEL_PATH = "driver_drowisness.onnx"

print("[INFO] Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("[INFO] Model loaded successfully.")

CLASSES = [
    "DangerousDriving",
    "Distracted",
    "Drinking",
    "SafeDriving",
    "SleepyDriving",
    "Yawn"
]

# ==========================================================
# 📷 CAMERA INITIALIZATION
# ==========================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("❌ Could not access camera. Check USB connection.")

print("[INFO] Camera initialized successfully.")

# ==========================================================
# 📡 FIREBASE (COMMENTED OUT)
# ==========================================================
"""
import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("path/to/firebase-key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://your-project.firebaseio.com/"
})
ref = db.reference("driver_status")
"""

def send_to_firebase(state):
    """
    Uncomment firebase block above and ref.push() below
    to enable realtime alert sync.
    """
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "state": state,
        "caution": "Driver inactive or same state for long!"
    }
    # ref.push(data)
    print("[Firebase] Would send:", data)

# ==========================================================
# ⚡ GPIO CONTROL FUNCTIONS
# ==========================================================
def set_leds(green=False, yellow=False, red=False):
    GPIO.output(LED_GREEN, GPIO.HIGH if green else GPIO.LOW)
    GPIO.output(LED_YELLOW, GPIO.HIGH if yellow else GPIO.LOW)
    GPIO.output(LED_RED, GPIO.HIGH if red else GPIO.LOW)

def buzz(freq_hz=3, duration=0.5):
    """Generate buzzer beep of given frequency and duration."""
    if freq_hz <= 0:
        return
    period = 1.0 / freq_hz
    end_time = time.time() + duration
    while time.time() < end_time:
        GPIO.output(BUZZER, GPIO.HIGH)
        time.sleep(period / 2)
        GPIO.output(BUZZER, GPIO.LOW)
        time.sleep(period / 2)

# ==========================================================
# 🧠 INFERENCE LOOP
# ==========================================================
prev_state = None
last_alert_time = time.time()

print("[INFO] Starting live detection... (press 'q' to quit)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed, retrying...")
            continue

        # -------- Preprocessing --------
        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # -------- Inference --------
        preds = session.run(None, {input_name: img})[0]
        state = CLASSES[np.argmax(preds)]

        # -------- Display --------
        color_map = {
            "SafeDriving": (0, 255, 0),
            "Distracted": (0, 255, 255),
            "Drinking": (255, 255, 0),
            "Yawn": (0, 165, 255),
            "SleepyDriving": (0, 0, 255),
            "DangerousDriving": (0, 0, 128),
        }
        cv2.putText(frame, f"State: {state}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_map.get(state, (255, 255, 255)), 2)
        cv2.imshow("Driver Drowsiness Detection", frame)

        # -------- State Logic --------
        current_time = time.time()
        if state != prev_state:
            prev_state = state
            last_alert_time = current_time

            # SAFE
            if state == "SafeDriving":
                set_leds(green=True)
                buzz(1, 0.1)

            # MID ALERTS
            elif state in ["Distracted", "Drinking", "Yawn"]:
                set_leds(yellow=True)
                buzz(3, 0.4)

            # HIGH ALERT
            elif state in ["SleepyDriving", "DangerousDriving"]:
                set_leds(red=True)
                buzz(8, 1.0)

        # Send Firebase alert if same state > 15s
        elif current_time - last_alert_time > 15:
            send_to_firebase(state)
            last_alert_time = current_time

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Cleaning up...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("[INFO] GPIO released. Program terminated successfully.")
