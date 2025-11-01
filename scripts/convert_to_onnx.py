import torch
from models.cnn_model import DrowsinessCNN

model = DrowsinessCNN(num_classes=6, img_size=128)
model.load_state_dict(torch.load("models/driver_drowsiness_final.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 128, 128)
torch.onnx.export(model, dummy_input, "models/driver_drowsiness.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                  opset_version=11)
print("✅ Model converted to ONNX successfully -> driver_drowsiness.onnx")
