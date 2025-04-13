import threading
from sklearn.metrics.pairwise import cosine_similarity
import torch
import cv2
from PIL import Image
from detection.YoloDetector import YOLODetector
from sortAlgorithm.Tracking import FaceTracker
from virtualZoom.VirtualZoom import VirtualZoom
from facePredict.CNN.FaceRecognition import (
    Model as MyModel,
    add_all_faces_to_db
)


my_model = None
last_update_frame = {}
face_center = {}
processing_faces = set()
summary_predictions = {}
lock = torch.mutex() if hasattr(torch, "mutex") else threading.Lock()
THRESHOLD = 0.7


def generate_embeddings(dataset_path):
    add_all_faces_to_db(dataset_path)


def initialize_system():
    global my_model

    try:
        my_model = MyModel()
        my_model.load_state_dict(torch.load("facenet_like_model.pth", map_location=torch.device("cpu")))
        my_model.eval()
        print("Kendi model başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return None, None, None, None

    cap = cv2.VideoCapture(0)
    detector = YOLODetector("../../models/yolov11s-face.pt")
    tracker = FaceTracker()
    zoom = VirtualZoom()

    return cap, detector, tracker, zoom

def recognize_face(zoomed_face, track_id, current_frame, x1, y1, x2, y2, db_path, recognized_faces_dict):
    from facePredict.CNN.FaceRecognition import transform, stored_embeddings, model as my_model

    try:
        device = next(my_model.parameters()).device

        image = cv2.cvtColor(zoomed_face, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = my_model(image_tensor).cpu().numpy().flatten()

        identity = "Bilinmiyor"
        best_score = -1
        best_name = "Bilinmiyor"

        for person, emb_list in stored_embeddings.items():
            for emb in emb_list:
                sim = cosine_similarity([embedding], [emb])[0][0]
                if sim > best_score:
                    best_score = sim
                    best_name = person

        identity = best_name

        with lock:
            recognized_faces_dict[track_id] = identity
            print(f"ID {track_id} → Tahmin: {identity} (skor: {best_score:.4f})")

    except Exception as e:
        print(f"Tanıma sırasında sorun oluştu: {e}")
        with lock:
            recognized_faces_dict[track_id] = "Hata"


