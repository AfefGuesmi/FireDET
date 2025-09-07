from ultralytics import YOLO
import torch

# Force torch to always load on CPU
torch_load = torch.load
def load_on_cpu(*args, **kwargs):
    return torch_load(*args, map_location="cpu", **{k: v for k, v in kwargs.items() if k != "map_location"})
torch.load = load_on_cpu

# Now load model
model = YOLO("pretrained/best.pt")

# Run test
results = model.val(data="dataset/data.yaml", split="test", imgsz=640, batch=16, device="cpu")
print(results)
