from ultralytics import YOLO
import torch

# Modeli yükle
model = YOLO("Models/yolov8n-face.pt")
# Cihazı kontrol et
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model {device} üzerinde çalışıyor.")