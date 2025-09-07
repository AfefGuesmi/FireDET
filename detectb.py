import torch
from ultralytics import YOLO

# Force PyTorch to load everything on CPU
torch_device = torch.device('cpu')

# Load the model safely on CPU
model = YOLO("pretrained/best.pt", task='detect')  # no device argument here

# Print the number of parameters
print(f"Number of parameters: {sum(p.numel() for p in model.model.parameters())}")

# Predict on CPU
results = model.predict(
    source=r"""C:\Users\starinfo\Desktop\images""",
    imgsz=640,
    conf=0.25,
    device='cpu'  # device set here for prediction
)

# Show and save results
for r in results:
    r.show()
    r.save("runs/detect/test")