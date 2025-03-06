import os
from ultralytics import YOLO
import cv2


model11Small = YOLO("../Models/yolov11s-face.pt")


def detect_and_save_faces(image_path, output_folder="detected_faces", padding=10):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Görüntü yüklenemedi. Lütfen doğru bir dosya yolu girin.")
        return

    os.makedirs(output_folder, exist_ok=True)

    original_frame = frame.copy()

    results = model11Small(frame)
    face_count = 0
    height, width, _ = frame.shape

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            if conf > 0.3:

                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(width, x2 + padding)
                y2 = min(height, y2 + padding)

                face = original_frame[y1:y2, x1:x2]
                face_filename = os.path.join(output_folder, f"face_{face_count}.jpg")
                cv2.imwrite(face_filename, face)
                face_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Face Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"{face_count} yüz tespit edildi ve {output_folder} klasörüne kaydedildi.")


image_path = r"C:\Users\ingin\Downloads\Tokatta-gorev-yapan-dogustan-gorme-engelli-34-yasindaki-Yunus-Yilmaz-cocukluk-hayali-ogretmenligi-9-yildir-basariyla-surduruyor-730x470.jpg"  # Buraya resmin yolunu girin
detect_and_save_faces(image_path)