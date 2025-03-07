"""
Bu projede nesne takibi için SORT  algoritması kullanılmıştır.
SORT, Kalman Filtresi ve IOU metriği ile çoklu nesne takibini gerçekleştiren gerçek zamanlı bir takip algoritmasıdır
Kaynak https://github.com/abewley/sort
Teşekkürler Alex Bewley
"""

import cv2
import torch
import numpy as np
from sort import Sort
from ultralytics import YOLO


model = YOLO("../Models/yolov11s-face.pt")

cap = cv2.VideoCapture("video.mp4")

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))


    results = model(frame)

    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()

            if conf > 0.5:
                detections.append([x1, y1, x2, y2, conf])


    if len(detections) > 0:
        tracked_objects = tracker.update(np.array(detections))

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # ID yazdır


    cv2.imshow("Face Tracking", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
