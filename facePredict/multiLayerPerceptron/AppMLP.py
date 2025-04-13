import tkinter as tk
import cv2
import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from MLP import FaceMLP
from sortAlgorithm.Tracking import FaceTracker
from detection.YoloDetector import YOLODetector
from virtualZoom.VirtualZoom import VirtualZoom
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image, ImageTk

recognized_faces = {}
face_center = {}
last_update_frame = {}
processing_faces = set()
lastSim=0

lock = threading.Lock()

executor = ThreadPoolExecutor(max_workers=4)

THRESHOLD = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_class_names(db_path):
    class_names = [name for name in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, name))]
    return class_names

def predict_face_with_mlp(face, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.fromarray(face).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image.view(image.size(0), -1))
        probabilities = F.softmax(output, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)

    print(f"Predicted ID: {predicted.item()}, Max Probability: {max_prob.item()}")  # Debugging for output

    if max_prob.item() < THRESHOLD:
        return "Bilinmiyor", max_prob.item()

    predicted_id = predicted.item()
    return class_names[predicted_id], max_prob.item()

def recognize_face(zoomed_face, track_id, current_frame, x1, y1, x2, y2, db_path, model, class_names):
    global recognized_faces, last_update_frame, face_center, processing_faces
    identity, similarity = predict_face_with_mlp(zoomed_face, model, class_names)


    with lock:
        recognized_faces[track_id] = identity
        last_update_frame[track_id] = current_frame
        face_center[track_id] = ((x1 + x2) // 2, (y1 + y2) // 2)
        processing_faces.discard(track_id)

    return similarity

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Yüz Tanıma Uygulaması")
        self.root.geometry("800x600")

        self.video_source = cv2.VideoCapture(0)

        self.detector = YOLODetector("../models/yolov11s-face.pt")
        self.tracker = FaceTracker()
        self.zoom = VirtualZoom()

        self.width = 800
        self.height = 600
        self.id_map = {}

        self.current_frame = 0
        self.db_path = r"C:\Users\umutk\OneDrive\Belgeler\dataset2"
        self.class_names = get_class_names(self.db_path)

        self.model = FaceMLP(160 * 160 * 3, 128, 64, len(self.class_names)).to(device)
        self.model.load_state_dict(torch.load("../../GUI/best_model.pth"))
        self.model.eval()

        self.start_button = tk.Button(self.root, text="Kamerayı Başlat", command=self.start_camera)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(self.root, text="Kamerayı Durdur", command=self.stop_camera)
        self.stop_button.pack(pady=5)

        self.quit_button = tk.Button(self.root, text="Çıkış", command=self.quit_app)
        self.quit_button.pack(pady=5)

        # Video Label
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

    def start_camera(self):
        self.current_frame = 0
        self.update_frame()

    def stop_camera(self):
        self.video_source.release()
        self.video_label.config(image="")

    def quit_app(self):
        self.stop_camera()
        self.root.quit()

    def update_frame(self):
        ret, frame = self.video_source.read()
        if not ret:
            return

        frame = cv2.resize(frame, (self.width, self.height))
        original_frame = frame.copy()
        frame_h, frame_w, _ = frame.shape

        detections = self.detector.detect(frame)
        tracked_objects = self.tracker.update(detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)

            if track_id not in self.id_map:
                for old_id in list(self.id_map.keys()):
                    if np.linalg.norm(np.array(self.id_map[old_id]) - np.array([x1, y1])) < 30:
                        track_id = old_id
                        break
                self.id_map[track_id] = (x1, y1)

            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame_w, x2 + margin)
            y2 = min(frame_h, y2 + margin)

            zoomed_face = self.zoom.get_zoomed_face(original_frame, (x1, y1, x2, y2), track_id, self.current_frame)
            similarity = 0
            if zoomed_face is not None and track_id not in processing_faces:
                processing_faces.add(track_id)
                similarity = executor.submit(recognize_face, zoomed_face, track_id, self.current_frame, x1, y1, x2, y2, db_path=self.db_path, model=self.model, class_names=self.class_names).result()
            name = recognized_faces.get(track_id, "Bilinmiyor")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if similarity == 0:
                global lastSim
                text_display = f"ID {track_id} - {name} - {lastSim:.5f}%"
            else:
                lastSim = similarity * 100
                text_display = f"ID {track_id} - {name} - {lastSim:.5f}%"
            (text_width, text_height), baseline = cv2.getTextSize(text_display, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (255, 0, 0), thickness=-1)
            cv2.putText(frame, text_display, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

        self.current_frame += 1

        self.video_label.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
