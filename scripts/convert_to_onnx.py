import os
import torch
from models.cnn_model import DrowsinessCNN
from colorama import Fore, Style

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "models/driver_drowsiness_final.pth"
ONNX_PATH = "models/driver_drowsiness.onnx"
IMG_SIZE = 128
NUM_CLASSES = 6

# ==========================================
# CONVERT TO ONNX
# ==========================================
def convert_model_to_onnx():
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(Fore.RED + f"❌ Model file not found: {MODEL_PATH}" + Style.RESET_ALL)
        return

    # Load trained model
    print(Fore.CYAN + "🔹 Loading PyTorch model..." + Style.RESET_ALL)
    model = DrowsinessCNN(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Create a dummy input (for model tracing)
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # Export to ONNX
    print(Fore.YELLOW + "⚙️  Converting model to ONNX format..." + Style.RESET_ALL)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11
    )

    print(Fore.GREEN + f"✅ Model successfully converted to ONNX → {ONNX_PATH}" + Style.RESET_ALL)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    convert_model_to_onnx()
