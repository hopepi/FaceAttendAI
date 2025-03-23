import cv2
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from deepface import DeepFace
from sortAlgorithm.Tracking import FaceTracker
from detection.YoloDetector import YOLODetector
from virtualZoom.VirtualZoom import VirtualZoom


recognized_faces = {}
face_center = {}
last_update_frame = {}
processing_faces = set()

lock = threading.Lock()

executor = ThreadPoolExecutor(max_workers=4)

THRESHOLD = 0.5


def recognize_face(zoomed_face, track_id, current_frame, x1, y1, x2, y2,db_path,model_name):
    global recognized_faces, last_update_frame, face_center, processing_faces
    result = DeepFace.find(zoomed_face, db_path=db_path, model_name=model_name, enforce_detection=False)
    identity = "Bilinmiyor"

    if len(result) > 0 and not result[0].empty:
        min_distance = result[0]["distance"][0]

        if min_distance < THRESHOLD:
            identity = result[0]["identity"][0].split("\\")[-2]
        else:
            identity = "Bilinmiyor"

    with lock:
        recognized_faces[track_id] = identity
        last_update_frame[track_id] = current_frame
        face_center[track_id] = ((x1 + x2) // 2, (y1 + y2) // 2)
        processing_faces.discard(track_id)
    with lock:
        recognized_faces[track_id] = "Hata"
        processing_faces.discard(track_id)


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
        frame_h, frame_w, _ = frame.shape

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

            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame_w, x2 + margin)
            y2 = min(frame_h, y2 + margin)

            zoomed_face = zoom.get_zoomed_face(original_frame, (x1, y1, x2, y2), track_id, current_frame)

            if track_id not in recognized_faces:
                recognized_faces[track_id] = "Bilinmiyor"

            if zoomed_face is not None and track_id not in processing_faces:
                if (
                        track_id not in last_update_frame or
                        (current_frame - last_update_frame[track_id]) > 50 or
                        np.linalg.norm(np.array(face_center.get(track_id, (0, 0))) - np.array(
                            [(x1 + x2) // 2, (y1 + y2) // 2])) > 30
                ):
                    processing_faces.add(track_id)
                    executor.submit(recognize_face, zoomed_face, track_id, current_frame, x1, y1, x2, y2,db_path=r"C:\Users\umutk\OneDrive\Belgeler\dataset1",model_name="Facenet")

            text = recognized_faces.get(track_id, "Bilinmiyor")

            if zoomed_face is not None:
                cv2.imshow(f"Zoomed Face - ID {track_id}", zoomed_face)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            text_display = f"ID {track_id} - {text}"
            (text_width, text_height), baseline = cv2.getTextSize(text_display, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (255, 0, 0),
                          thickness=-1)
            cv2.putText(frame, text_display, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)

        cv2.imshow("Face Tracking + Recognition", frame)

        current_frame += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()