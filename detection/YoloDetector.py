from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame, conf_threshold=0.5):
        detections = []
        results = self.model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()

                if conf > conf_threshold:
                    detections.append([x1, y1, x2, y2, conf])

        return detections