import cv2
from sortAlgorithm.Tracking import FaceTracker
from detection.YoloDetector import YOLODetector
from virtualZoom.VirtualZoom import VirtualZoom

def main():
    cap = cv2.VideoCapture("video.mp4")

    detector = YOLODetector("../models/yolov11s-face.pt")
    tracker = FaceTracker()
    zoom = VirtualZoom(output_folder="ZoomedFaces")

    width, height = 800, 600

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        # Orijinal frame'i koru (çizimsiz temiz görüntü)
        original_frame = frame.copy()

        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)

            # Orijinal frame üzerinden zoom yapıp kaydet
            zoom.save_zoom(original_frame, (x1, y1, x2, y2), track_id, interval=15)

            # Ana frame üzerine daha ince çerçeve çizimi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            text = f"ID {track_id}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5),
                          (x1 + text_width, y1), (0, 255, 0), thickness=-1)

            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)

        cv2.imshow("Face Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
