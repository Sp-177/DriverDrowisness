import os
import sys
import torch
from colorama import Fore, Style

# ==========================================================
# IMPORTS & PATH SETUP
# ==========================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.cnn_model import DrowsinessCNN  # Attention-enhanced model


# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_PATH = "models/DrwoisnessCNN_Pro.pth"
ONNX_PATH = "models/DrwoisnessCNN_Pro.onnx"
IMG_SIZE = 128
NUM_CLASSES = 6


# ==========================================================
# ONNX CONVERSION FUNCTION
# ==========================================================
def convert_to_onnx():
    # --- Check model existence ---
    if not os.path.exists(MODEL_PATH):
        print(Fore.RED + f"❌ Model not found at: {MODEL_PATH}" + Style.RESET_ALL)
        return

    # --- Load model ---
    print(Fore.CYAN + "🔹 Loading trained DrwoisnessCNN_Pro model..." + Style.RESET_ALL)
    model = DrowsinessCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # --- Create dummy input ---
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # --- Display info ---
    total_params = sum(p.numel() for p in model.parameters()) / 1_000_000
    print(Fore.BLUE + f"ℹ️  Model Parameters: {total_params:.2f}M" + Style.RESET_ALL)

    # --- Export to ONNX ---
    print(Fore.YELLOW + "⚙️  Converting model to ONNX format..." + Style.RESET_ALL)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input_image"],
        output_names=["output_logits"],
        dynamic_axes={
            "input_image": {0: "batch"},
            "output_logits": {0: "batch"}
        },
        opset_version=17,  # use latest stable opset
        do_constant_folding=True,
        verbose=False
    )

    print(Fore.GREEN + f"✅ Successfully converted to ONNX → {ONNX_PATH}" + Style.RESET_ALL)
    print(Fore.CYAN + "💡 You can now load it with ONNX Runtime or TensorRT for deployment." + Style.RESET_ALL)


# ==========================================================
# OPTIONAL: ONNX RUNTIME VALIDATION
# ==========================================================
def validate_onnx_runtime():
    try:
        import onnxruntime as ort
        import numpy as np

        print(Fore.YELLOW + "\n🔍 Validating exported ONNX model..." + Style.RESET_ALL)
        ort_session = ort.InferenceSession(ONNX_PATH)

        dummy_input = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        outputs = ort_session.run(None, {"input_image": dummy_input})

        print(Fore.GREEN + f"✅ ONNX model ran successfully — Output shape: {outputs[0].shape}" + Style.RESET_ALL)

    except ImportError:
        print(Fore.RED + "⚠️  onnxruntime not installed. Skipping validation. Run: pip install onnxruntime" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"❌ Validation failed: {str(e)}" + Style.RESET_ALL)


# ==========================================================
# MAIN ENTRY
# ==========================================================
if __name__ == "__main__":
    convert_to_onnx()
    validate_onnx_runtime()
