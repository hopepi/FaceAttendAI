import cv2
import numpy as np


class VirtualZoom:
    def __init__(self, zoom_size=(160, 160), fps=30):
        self.zoom_size = zoom_size
        self.frame_counts = {}
        self.blur_thresholds = {}
        self.intervals = {}
        self.last_saved_frame = {}
        self.ss_per_id = {}
        self.max_frame_gap = fps
        self.max_ss_per_id = 20

    def detect_blur(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance

    def get_zoomed_face(self, frame, bbox, track_id, current_frame):
        if track_id not in self.frame_counts:
            self.frame_counts[track_id] = 0
            self.blur_thresholds[track_id] = 15
            self.intervals[track_id] = 5
            self.last_saved_frame[track_id] = 0
            self.ss_per_id[track_id] = 0

        self.frame_counts[track_id] += 1

        if self.ss_per_id[track_id] >= self.max_ss_per_id:
            print(f"[Frame {current_frame}] [ID {track_id}] Maksimum SS limitine ulaşıldı yeni SS alınmıyor")
            return None

        print(f"[Frame {current_frame}] [ID {track_id}] Bbox: {bbox}")


        x1, y1, x2, y2 = map(int, bbox)
        face_area = (x2 - x1) * (y2 - y1)
        min_face_area = 625

        if face_area < min_face_area:
            return None

        force_save = (current_frame - self.last_saved_frame[track_id]) >= self.max_frame_gap

        if force_save:
            print(f"[Frame {current_frame}] [ID {track_id}]  Zorunlu SS alınıyor")

        if self.frame_counts[track_id] % self.intervals[track_id] == 0 or force_save:
            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0:
                return None

            zoom_face = cv2.resize(face_img, self.zoom_size)
            blur_value = self.detect_blur(zoom_face)

            print(f"[Frame {current_frame}] [ID {track_id}] Blur Değeri: {blur_value}")

            size_factor = max(1.0, min(3.0, face_area / 3000))
            dynamic_blur_threshold = int(18 * size_factor)

            if blur_value > 25:
                self.intervals[track_id] = min(10, self.intervals[track_id] + 1)
            else:
                self.intervals[track_id] = max(2, self.intervals[track_id] - 1)

            self.blur_thresholds[track_id] = max(18, min(30, dynamic_blur_threshold))

            if blur_value < self.blur_thresholds[track_id] and not force_save:
                print(f"[Frame {current_frame}] [ID {track_id}] Yüz bulanık embedding çıkarılmıyor")
                return None

            self.last_saved_frame[track_id] = current_frame
            self.ss_per_id[track_id] += 1

            return zoom_face

        return None

    def get_total_screenshots(self):
        return sum(self.ss_per_id.values())
