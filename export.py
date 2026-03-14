from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("runs/segmentation/weights/best.pt")  # Path to your trained model

# Export the model to ONNX format with opset 21
model.export(format="onnx", imgsz=320, dynamic=True, simplify=True, optimize=True, half=False, device="cpu", name="evaluation", opset=21)

print("Model exported to ONNX format as 'evaluation.onnx'")