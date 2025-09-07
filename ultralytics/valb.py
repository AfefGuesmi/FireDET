from ultralytics import YOLO

# Charger ton modèle
model = YOLO("best.pt")

# Lancer la validation
results = model.val(data="dataset/data.yaml", imgsz=640, batch=16)
