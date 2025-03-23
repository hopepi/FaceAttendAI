import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from facePredict.MyModel import initialize_system, recognize_face, generate_embeddings

cap = None
executor = ThreadPoolExecutor(max_workers=2)
frame_lock = threading.Lock()
database_path = ""

root = tk.Tk()
root.title("Yüz Tanıma Sistemi (Kendi Modelin)")
root.geometry("400x300")

video_window = None
video_label = None

id_map = {}
current_frame = 0
processing_faces = set()
recognized_faces = {}
lock = threading.Lock()


def select_database():
    global database_path
    database_path = filedialog.askdirectory(title="Veri Klasörünü Seç")
    if database_path:
        generate_embeddings(database_path)
        messagebox.showinfo("Başarılı", "Veritabanı oluşturuldu veya yüklendi.")

def start_camera():
    global cap, detector, tracker, zoom, video_window, video_label, current_frame
    cap, detector, tracker, zoom = initialize_system()
    current_frame = 0
    if cap:
        video_window = tk.Toplevel(root)
        video_window.title("Kamera Görüntüsü")
        video_window.geometry("800x600")

        video_label = tk.Label(video_window)
        video_label.pack(pady=10)

        update_frame()

def update_frame():
    global cap, video_label, current_frame
    if cap is None or video_label is None:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.resize(frame, (800, 600))
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

        if zoomed_face is not None and track_id not in processing_faces:
            processing_faces.add(track_id)
            executor.submit(wrapper_recognition, zoomed_face, track_id, current_frame, x1, y1, x2, y2)

        name = recognized_faces.get(track_id, "Bilinmiyor")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id} - {name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    current_frame += 1

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    video_label.after(10, update_frame)

def wrapper_recognition(zoomed_face, track_id, current_frame, x1, y1, x2, y2):
    global processing_faces
    try:
        recognize_face(zoomed_face, track_id, current_frame, x1, y1, x2, y2, database_path, recognized_faces)
    finally:
        with lock:
            processing_faces.discard(track_id)

def stop_camera():
    global cap, video_label, video_window
    if cap:
        cap.release()
        cap = None
    if video_label:
        video_label.config(image="")
    if video_window:
        video_window.destroy()
        video_window = None

def quit_app():
    stop_camera()
    root.destroy()

btn_select_db = tk.Button(root, text="Veri Klasörü Seç", command=select_database)
btn_select_db.pack(pady=5)

btn_start = tk.Button(root, text="Kamerayı Başlat", command=start_camera)
btn_start.pack(pady=5)

btn_stop = tk.Button(root, text="Kamerayı Durdur", command=stop_camera)
btn_stop.pack(pady=5)

btn_quit = tk.Button(root, text="Çıkış", command=quit_app)
btn_quit.pack(pady=5)

root.mainloop()