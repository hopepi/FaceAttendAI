from ultralytics import YOLO
import cv2

# Modeli yükle
# model8Nano = YOLO("Models/yolov8n-face.pt")
modelFineTuning = YOLO("Models/yolov8n-face-fine-tuning.pt")
model11Small = YOLO("Models/yolov11s-face.pt")#Recommend
# model8Medium = YOLO("Models/yolov8m-face.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model11Small(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            if conf > 0.3:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Sonuçları ekranda göster
    cv2.imshow("YOLOv8 Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







