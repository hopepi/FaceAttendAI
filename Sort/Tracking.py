import cv2
import torch
import numpy as np
from sort import Sort  # SORT kütüphanesi
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO("../Models/yolov11s-face.pt")  # Yüz tespiti için uygun bir model kullan

# Video kaynağını aç
cap = cv2.VideoCapture("video.mp4")  # Video dosyası veya cap = cv2.VideoCapture(0) ile canlı kamera

# SORT tracker'ı başlat
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Pencere boyutlarını ayarla
WINDOW_WIDTH = 800  # Genişlik
WINDOW_HEIGHT = 600  # Yükseklik

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Video bittiğinde çık

    # Frame'i yeniden boyutlandır (ekrana sığması için)
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # YOLO ile yüzleri tespit et
    results = model(frame)

    detections = []  # SORT için kullanılacak tespit listesi

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()

            if conf > 0.5:  # Güven eşiği (Yanlış tespitleri önlemek için)
                detections.append([x1, y1, x2, y2, conf])  # [x1, y1, x2, y2, confidence]

    # SORT'a güncellenmiş yüz tespitlerini ver
    if len(detections) > 0:
        tracked_objects = tracker.update(np.array(detections))

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)  # Takip edilen objelerin bilgileri
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yüzü çiz
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # ID yazdır

    # Sonucu göster
    cv2.imshow("Face Tracking", frame)

    # Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
