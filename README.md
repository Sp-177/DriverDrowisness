# 🚗 Driver Drowsiness Detection System (AI + IoT)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A real-time **driver drowsiness detection system** powered by **deep learning (CNN)** and **computer vision**. This project monitors driver alertness and classifies states such as safe driving, yawning, eyes closed, looking down, or distracted behavior.

🎯 **Deployable on**: Local PC, Raspberry Pi 4, or expandable to web-controlled dashboards.

---

## 🌟 Features

- ✅ Real-time drowsiness detection using webcam/USB camera
- ✅ Deep learning model trained on Kaggle dataset (6 driver states)
- ✅ Visual alerts with bounding boxes (Green/Yellow/Red)
- ✅ ONNX model export for edge deployment
- ✅ Raspberry Pi 4 compatible with optimized inference
- ✅ Modular codebase with training, evaluation, and testing scripts

---

## 📁 Project Structure

```
DriverDrowsiness/
│
├── data/
│   └── dataset/              # Kaggle dataset (train/test images)
│
├── models/
│   ├── cnn_model.py          # CNN architecture
│   ├── driver_drowsiness_final.pth   # Trained PyTorch model
│   └── driver_drowsiness.onnx        # ONNX exported model
│
├── scripts/
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Model evaluation
│   ├── preprocess.py         # Dataset preprocessing utilities
│   ├── test_camera.py        # Real-time webcam detection (PC)
│   ├── test_camera_pi.py     # Real-time USB camera detection (Raspberry Pi 4)
│   └── convert_to_onnx.py    # Convert PyTorch → ONNX
│
├── requirements.txt          # Dependencies (PC)
├── requirements_rpi.txt      # Dependencies (Raspberry Pi)
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules
```

---

## 🧠 Dataset

**Dataset:** [Driver's Inattention Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/zaydmanndhour/driver-drowsiness-detection)

### Driver States Detected:
- 🟢 Safe driving
- 🥱 Yawning
- 😴 Eyes closed
- 🔽 Looking down
- 📱 Talking on phone
- 🤳 Distracted

### Preprocessing:
- Images resized to **128×128**
- Normalized to `[0,1]` pixel range
- Data augmentation: random flips, brightness/contrast adjustments, rotations

---

## ⚙️ Installation

### 🖥️ Local PC (Windows/Linux/macOS)

```bash
# Clone the repository
git clone https://github.com/<your-username>/DriverDrowsiness.git
cd DriverDrowsiness

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 🍓 Raspberry Pi 4

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install python3-pip libatlas-base-dev libopenblas-dev libjpeg-dev libpng-dev -y

# Clone repository
git clone https://github.com/<your-username>/DriverDrowsiness.git
cd DriverDrowsiness

# Install Python dependencies
pip3 install -r requirements_rpi.txt
```

---

## 🧩 Model Architecture

**DrowsinessCNN** - Custom Convolutional Neural Network

```python
Architecture:
├── Conv2D (32 filters) + BatchNorm + ReLU + MaxPool
├── Conv2D (64 filters) + BatchNorm + ReLU + MaxPool
├── Conv2D (128 filters) + BatchNorm + ReLU + MaxPool
├── Flatten
├── FC Layer (256 units) + ReLU + Dropout(0.5)
└── FC Layer (6 classes - output)
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Batch Size: 32
- Epochs: 25

---

## 🚀 Usage

### 1️⃣ Train the Model

```bash
python scripts/train.py
```

**Outputs:**
- `models/driver_drowsiness_final.pth` (best model checkpoint)
- Training logs with accuracy and loss metrics
- Confusion matrix and classification report

### 2️⃣ Evaluate the Model

```bash
python scripts/evaluate.py
```

Displays:
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix

### 3️⃣ Convert to ONNX (for deployment)

```bash
python scripts/convert_to_onnx.py
```

**Output:** `models/driver_drowsiness.onnx`

### 4️⃣ Real-Time Detection (PC)

```bash
python scripts/test_camera.py
```

- Opens webcam feed
- Detects driver state in real-time
- Draws bounding boxes with color-coded alerts:
  - 🟢 **Green**: Alert/Safe
  - 🟡 **Yellow**: Drowsy (yawning, distracted)
  - 🔴 **Red**: Asleep (eyes closed)
- Press `Q` to quit

### 5️⃣ Real-Time Detection (Raspberry Pi 4)

```bash
python3 scripts/test_camera_pi.py
```

- Uses USB camera input
- Lightweight ONNX inference
- Live video feed with driver state labels

---

## 📦 Dependencies

### `requirements.txt` (PC)
```
torch
torchvision
torchaudio
opencv-python
numpy
matplotlib
pandas
scikit-learn
onnx
onnxruntime
tqdm
```

### `requirements_rpi.txt` (Raspberry Pi)
```
torch
torchvision
opencv-python
numpy
onnxruntime
```

---

## 🔮 Future Enhancements

- 🌐 **Web Dashboard**: Stream camera feed and log driver behavior
- ☁️ **Cloud Integration**: Firebase/MongoDB for data storage
- 🚦 **Vehicle Control**: Trigger alerts (buzzer, speed limiter, hazard lights)
- 📱 **Mobile App**: Remote monitoring via smartphone
- 🤖 **Edge Optimization**: TensorFlow Lite for ultra-low power inference
- 🧭 **Sensor Fusion**: GPS and motion sensor integration

---

## 📝 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute with attribution.

---

## 👨‍💻 Author

**Shubham Patel**  
B.Tech IT, NIT Raipur  
📧 shub404.x@gmail.com

---

## 🙏 Acknowledgments

- **Dataset**: [Driver's Inattention Detection - Kaggle](https://www.kaggle.com/datasets/zaydmanndhour/driver-drowsiness-detection)
- **Frameworks**: PyTorch, OpenCV, ONNX Runtime
- **Hardware**: Raspberry Pi 4, USB Camera

---

## ⭐ Star This Repository

If you find this project helpful, please consider giving it a ⭐ on GitHub!

---

## 📞 Contact & Support

For questions, suggestions, or collaboration:
- 📧 Email: shub404.x@gmail.com
- 💼 GitHub: [@your-username](https://github.com/your-username)

---

**Made with ❤️ for safer roads**
