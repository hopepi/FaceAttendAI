import cv2
import numpy as np
from deepface import DeepFace
from sortAlgorithm.Tracking import FaceTracker
from detection.YoloDetector import YOLODetector
from virtualZoom.VirtualZoom import VirtualZoom

# Veritabanı klasörü
db_path = r"C:\Users\umutk\OneDrive\Belgeler\fasas\dataset"

# Takip edilen yüzleri ve kimlikleri saklayan sözlük
recognized_faces = {}

def main():
    cap = cv2.VideoCapture(0)

    detector = YOLODetector("../models/yolov11s-face.pt")  # YOLO modeli
    tracker = FaceTracker()  # SORT takip algoritması
    zoom = VirtualZoom()

    width, height = 800, 600
    id_map = {}

    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        original_frame = frame.copy()

        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)

            if track_id not in id_map:
                for old_id in list(id_map.keys()):
                    if np.linalg.norm(np.array(id_map[old_id]) - np.array([x1, y1])) < 30:
                        track_id = old_id
                        break
                id_map[track_id] = (x1, y1)

            # Yüz bölgesini kes
            face_roi = original_frame[y1:y2, x1:x2]

            # Daha önce tanınmamış yüzler için DeepFace çalıştır
            if track_id not in recognized_faces:
                try:
                    result = DeepFace.find(face_roi, db_path=db_path, model_name="Facenet", enforce_detection=False)
                    if len(result) > 0 and not result[0].empty:
                        identity = result[0]["identity"][0].split("\\")[-2]
                        recognized_faces[track_id] = identity
                    else:
                        recognized_faces[track_id] = "Bilinmiyor"
                except Exception as e:
                    recognized_faces[track_id] = "Hata"

            text = recognized_faces[track_id]

            # Zoom özelliğini kullan (yakınlaştırılmış yüz)
            zoomed_face = zoom.get_zoomed_face(original_frame, (x1, y1, x2, y2), track_id, current_frame)
            if zoomed_face is not None:
                cv2.imshow(f"Zoomed Face - ID {track_id}", zoomed_face)

            # Yüz etrafına dikdörtgen çiz ve kimliği ekrana yaz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            text_display = f"ID {track_id} - {text}"
            (text_width, text_height), baseline = cv2.getTextSize(text_display, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (0, 255, 0), thickness=-1)
            cv2.putText(frame, text_display, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)

        cv2.imshow("Face Tracking + Recognition", frame)

        current_frame += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    for track_id, count in zoom.ss_per_id.items():
        print(f"---ID {track_id} için toplam SS: {count}---")
    print(f"------Toplam SS Sayısı: {zoom.get_total_screenshots()}------")

if __name__ == "__main__":
    main()
