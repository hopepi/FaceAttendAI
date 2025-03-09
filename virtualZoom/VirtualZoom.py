import cv2
import os


class VirtualZoom:
    def __init__(self, output_folder="zoomed_faces", zoom_size=(300, 300)):
        self.zoom_size = zoom_size
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.frame_counts = {}

    def save_zoom(self, frame, bbox, track_id, interval=15):
        # Her id için ayrı sayaç oluştur
        if track_id not in self.frame_counts:
            self.frame_counts[track_id] = 0

        self.frame_counts[track_id] += 1

        # Her 'interval' karede bir yüzü kaydet
        if self.frame_counts[track_id] % interval == 0:
            x1, y1, x2, y2 = map(int, bbox)

            # Yüzü kırp
            face_img = frame[y1:y2, x1:x2]

            # Yeniden boyutlandır (zoom işlemi)
            zoom_face = cv2.resize(face_img, self.zoom_size)

            # Yüzleri id'ye özel klasöre kaydet
            track_folder = os.path.join(self.output_folder, f"ID_{track_id}")
            os.makedirs(track_folder, exist_ok=True)

            face_filename = os.path.join(track_folder, f"{self.frame_counts[track_id]}.jpg")
            cv2.imwrite(face_filename, zoom_face)
