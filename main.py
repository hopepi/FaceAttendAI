import cv2
import numpy as np
from sortAlgorithm.Tracking import FaceTracker
from detection.YoloDetector import YOLODetector
from virtualZoom.VirtualZoom import VirtualZoom

def main():
    cap = cv2.VideoCapture("video2.mp4")

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

            zoomed_face = zoom.get_zoomed_face(original_frame, (x1, y1, x2, y2), track_id, current_frame)

            if zoomed_face is not None:
                cv2.imshow(f"Zoomed Face - ID {track_id}", zoomed_face)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            text = f"ID {track_id}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (0, 255, 0), thickness=-1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)

        cv2.imshow("Face Tracking", frame)

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
