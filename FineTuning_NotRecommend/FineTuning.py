from ultralytics import YOLO


yaml_path = r"C:/Users/umutk/OneDrive/Belgeler/yeni veri setleri/data.yaml"
if __name__ == "__main__":
    # Modeli yükle
    model = YOLO("../Models/yolov8n-face.pt")

    # Fine-tuning başlat
    model.train(data=yaml_path,
                epochs=20,
                imgsz=640,
                batch=16,
                freeze=10)

