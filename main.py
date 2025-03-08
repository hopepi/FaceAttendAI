import cv2
from Sort.Tracking import FaceTracker
from Detection.YoloDetector import YOLODetector

def main():
    cap = cv2.VideoCapture("video.mp4")

    detector = YOLODetector("../Models/yolov11s-face.pt")
    tracker = FaceTracker()

    width, height = 800, 600

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        detections = detector.detect(frame)

        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 200, 250), 2)

            text = f"ID {track_id}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 10),
                          (x1 + text_width, y1), (100, 200, 250), thickness=-1)

            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)

        cv2.imshow("Face Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()