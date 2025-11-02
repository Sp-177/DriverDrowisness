# ==========================================================
# 📘 DRIVER DROWSINESS DETECTION — REAL-TIME PI TEST SCRIPT
# Model: driver_drowisness.onnx
# Author: Shubham Patel (NIT Raipur)
# ==========================================================

import cv2
import numpy as np
import onnxruntime as ort
import RPi.GPIO as GPIO
import time
from datetime import datetime

# ----------------------------------------------------------
# ⚙️ OPTIONAL: Firebase setup (commented out)
# ----------------------------------------------------------
"""
import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("/home/pi/firebase-key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://<your-project>.firebaseio.com/"
})
ref = db.reference("DriverAlerts")

def send_firebase_alert(state):
    ref.push({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "state": state,
        "message": f"⚠️ Driver state: {state}"
    })
"""

# ----------------------------------------------------------
# 🧠 MODEL LOADING
# ----------------------------------------------------------
MODEL_PATH = "/home/raspberrypi/Desktop/Driver_Drowisness/driver_drowsiness.onnx"
INPUT_SIZE = 128  # must match your training size
CLASSES = ["Safe", "Sleepy", "Drinking", "Distracted", "Dangerous"]

print("[INFO] Loading ONNX model...")
try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ----------------------------------------------------------
# 🎛️ GPIO SETUP (FIXED PIN ASSIGNMENTS)
# ----------------------------------------------------------
LED_RED = 17     # Danger - GPIO17 (FLIPPED ORDER)
LED_YELLOW = 27  # Warning - GPIO27
LED_GREEN = 22   # Safe - GPIO22 (FLIPPED ORDER)
BUZZER = 23      # Buzzer - GPIO23

# Clean up any previous GPIO settings
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Setup all pins as output
for pin in [LED_GREEN, LED_YELLOW, LED_RED, BUZZER]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

print("[INFO] GPIO configured successfully.")
print(f"[INFO] Pin mapping: GREEN={LED_GREEN}, YELLOW={LED_YELLOW}, RED={LED_RED}, BUZZER={BUZZER}")

# ----------------------------------------------------------
# 🎥 CAMERA INIT (IMPROVED ERROR HANDLING)
# ----------------------------------------------------------
print("[INFO] Initializing camera...")
cap = cv2.VideoCapture(0)

# Wait a moment for camera to initialize
time.sleep(2)

if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    print("   Try running: sudo modprobe bcm2835-v4l2")
    GPIO.cleanup()
    exit(1)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[INFO] Camera initialized successfully.")

# ----------------------------------------------------------
# 🔁 STATE CONTROL LOGIC (IMPROVED)
# ----------------------------------------------------------
def control_gpio(state):
    """Control LEDs and buzzer based on driver state"""
    # Reset all outputs
    GPIO.output(LED_GREEN, GPIO.LOW)
    GPIO.output(LED_YELLOW, GPIO.LOW)
    GPIO.output(LED_RED, GPIO.LOW)
    GPIO.output(BUZZER, GPIO.LOW)

    if state == "Safe":
        GPIO.output(LED_GREEN, GPIO.HIGH)

    elif state == "Sleepy":
        GPIO.output(LED_YELLOW, GPIO.HIGH)
        # Short beep
        GPIO.output(BUZZER, GPIO.HIGH)
        time.sleep(0.15)
        GPIO.output(BUZZER, GPIO.LOW)

    elif state in ["Drinking", "Distracted"]:
        GPIO.output(LED_YELLOW, GPIO.HIGH)
        # Medium beep
        GPIO.output(BUZZER, GPIO.HIGH)
        time.sleep(0.25)
        GPIO.output(BUZZER, GPIO.LOW)

    elif state == "Dangerous":
        GPIO.output(LED_RED, GPIO.HIGH)
        # Triple beep pattern
        for _ in range(3):
            GPIO.output(BUZZER, GPIO.HIGH)
            time.sleep(0.15)
            GPIO.output(BUZZER, GPIO.LOW)
            time.sleep(0.1)

# ----------------------------------------------------------
# 🚀 REAL-TIME INFERENCE LOOP
# ----------------------------------------------------------
print("[INFO] Starting live detection... (press 'q' to quit)")
print("[INFO] Press Ctrl+C to stop the program safely")

last_state = None
last_alert_time = 0
frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Warning: Frame not captured, retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1

        # --- Preprocess frame for ONNX ---
        img_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)   # add batch dimension
        img = np.ascontiguousarray(img)

        # --- Run inference ---
        try:
            preds = session.run(None, {input_name: img})[0]
            pred_class = np.argmax(preds)
            confidence = preds[0][pred_class]
            state = CLASSES[pred_class]
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            continue

        # --- Display info on frame ---
        cv2.putText(frame, f"State: {state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Calculate and display FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- Show frame ---
        cv2.imshow("Driver Drowsiness Detection", frame)

        # --- Control GPIO ---
        control_gpio(state)

        # --- Firebase alert (commented out) ---
        """
        now = time.time()
        if state != last_state or (now - last_alert_time) > 10:
            send_firebase_alert(state)
            last_state = state
            last_alert_time = now
        """

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] 'q' pressed - exiting...")
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user (Ctrl+C)")

except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # --- Cleanup ---
    print("[INFO] Cleaning up resources...")

    # Turn off all GPIO outputs before cleanup
    try:
        GPIO.output(LED_GREEN, GPIO.LOW)
        GPIO.output(LED_YELLOW, GPIO.LOW)
        GPIO.output(LED_RED, GPIO.LOW)
        GPIO.output(BUZZER, GPIO.LOW)
    except:
        pass

    # Release camera
    if cap.isOpened():
        cap.release()

    # Close windows
    cv2.destroyAllWindows()

    # Cleanup GPIO
    GPIO.cleanup()

    print("[INFO] GPIO released. Camera closed.")
    print("[INFO] Program terminated successfully.")
    print(f"[INFO] Total frames processed: {frame_count}")
