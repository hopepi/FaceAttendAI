import cv2
import numpy as np
from deepface import DeepFace
from sortAlgorithm.Tracking import FaceTracker
from detection.YoloDetector import YOLODetector
from virtualZoom.VirtualZoom import VirtualZoom

# Veritabanı klasörü
db_path = r"C:\Users\umutk\OneDrive\Belgeler\fasas\dataset"

# Tanınan yüzleri saklayan sözlük
recognized_faces = {}
face_center = {}
last_update_frame = {}
previous_track_id = {}

def main():
    cap = cv2.VideoCapture(0)

    detector = YOLODetector("../models/yolov11s-face.pt")
    tracker = FaceTracker()
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

            # Zoom ile yüzü al
            zoomed_face = zoom.get_zoomed_face(original_frame, (x1, y1, x2, y2), track_id, current_frame)

            # Eğer track_id daha önce tanınmamışsa, "Bilinmiyor" olarak ayarla
            if track_id not in recognized_faces:
                recognized_faces[track_id] = "Bilinmiyor"

            # Eğer zoomed_face boş değilse ve belirli koşullar sağlanıyorsa tahmin yap
            if zoomed_face is not None and (
                    track_id not in previous_track_id or
                    track_id != previous_track_id.get(track_id, -1) or
                    (current_frame - last_update_frame.get(track_id, 0)) > 50 or
                    np.linalg.norm(
                        np.array(face_center.get(track_id, (0, 0))) - np.array([(x1 + x2) // 2, (y1 + y2) // 2])) > 30):

                try:
                    result = DeepFace.find(zoomed_face, db_path=db_path, model_name="Facenet", enforce_detection=False)
                    if len(result) > 0 and not result[0].empty:
                        identity = result[0]["identity"][0].split("\\")[-2]
                        recognized_faces[track_id] = identity
                    else:
                        recognized_faces[track_id] = "Bilinmiyor"
                except Exception as e:
                    recognized_faces[track_id] = "Hata"

                # Güncellenen ID ve yüz merkezini kaydet
                last_update_frame[track_id] = current_frame
                face_center[track_id] = ((x1 + x2) // 2, (y1 + y2) // 2)
                previous_track_id[track_id] = track_id

            # Track ID'nin her durumda tanımlı olduğunu garantile
            text = recognized_faces.get(track_id, "Bilinmiyor")

            if zoomed_face is not None:
                cv2.imshow(f"Zoomed Face - ID {track_id}", zoomed_face)

            # Yüz etrafına dikdörtgen çiz ve kimliği ekrana yaz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            text_display = f"ID {track_id} - {text}"
            (text_width, text_height), baseline = cv2.getTextSize(text_display, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (0, 255, 0),
                          thickness=-1)
            cv2.putText(frame, text_display, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)

        cv2.imshow("Face Tracking + Recognition", frame)

        current_frame += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()