"""
İssue : Our own model and many of the deepface algorithms could not be integrated
"""
import os
import pickle
import tkinter as tk
from tkinter import ttk, filedialog
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from deepface import DeepFace
import cv2
from detection.YoloDetector import YOLODetector
from sortAlgorithm.Tracking import FaceTracker
from virtualZoom.VirtualZoom import VirtualZoom



root = tk.Tk()
root.title("Yüz Tespiti ile Yoklama Sistemi")
root.geometry("450x500")

selected_source = tk.StringVar()
selected_source.set("Canlı Kamera")

face_recognition_model = None
recognized_faces = {}
face_center = {}
last_update_frame = {}
processing_faces = set()
lock = threading.Lock()
THRESHOLD = 0.7
final_recognized = []
summary_predictions = {}
loading_embeddings = False
database = ""


def recognize_face(zoomed_face, track_id, current_frame, x1, y1, x2, y2, db_path):
    global recognized_faces, last_update_frame, face_center, processing_faces, summary_predictions, loading_embeddings

    if loading_embeddings:
        print(f"Yüz tanıma şu an yapılamaz, embedding hesaplanıyor.")
        return

    try:
        zoomed_face_rgb = cv2.cvtColor(zoomed_face, cv2.COLOR_BGR2RGB)

        result = DeepFace.find(zoomed_face_rgb, db_path=db_path, model_name="Facenet", enforce_detection=False)

        identity = "Bilinmiyor"

        if len(result) > 0 and not result[0].empty:
            min_distance = result[0]["distance"][0]

            if min_distance < THRESHOLD:
                identity = result[0]["identity"][0].split("\\")[-2]
            else:
                print(f"Tanıma başarısız!")



        with lock:
            recognized_faces[track_id] = identity
            last_update_frame[track_id] = current_frame
            face_center[track_id] = ((x1 + x2) // 2, (y1 + y2) // 2)

            if track_id not in summary_predictions:
                summary_predictions[track_id] = []

            if len(summary_predictions[track_id]) > 10:
                summary_predictions[track_id].pop(0)

            summary_predictions[track_id].append(identity)

            processing_faces.discard(track_id)

    except Exception as e:
        print(f"Hata oluştu! Track ID: {track_id}, Hata: {str(e)}")
        with lock:
            recognized_faces[track_id] = "Hata"
            processing_faces.discard(track_id)



def generate_embeddings(dataset_path):
    global loading_embeddings
    loading_embeddings = True
    embedding_file = os.path.join(dataset_path, "embeddings.pkl")

    if os.path.exists(embedding_file):
        print(f"Embedding dosyası zaten var: {embedding_file}")
        loading_embeddings = False
        return

    print(f"Embedding oluşturuluyor")

    embeddings = {}

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        embeddings[person_name] = []

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)

            try:
                embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)
                if embedding:
                    embeddings[person_name].append(embedding[0]["embedding"])
            except Exception as e:
                print(f"Hata oluştu: {img_path} → {e}")

    with open(embedding_file, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Embeddingler başarıyla kaydedildi: {embedding_file}")
    loading_embeddings = False



def initialize_system():
    try:
        global face_recognition_model
        face_recognition_model = DeepFace.build_model("Facenet")
        print(f"Başarıyla yüklendi.")
    except Exception as e:
        print(f"Modeli yüklenirken hata oluştu: {e}")
        return None, None, None, None

    cap = cv2.VideoCapture(0)
    detector = YOLODetector("../models/yolov11s-face.pt")
    tracker = FaceTracker()
    zoom = VirtualZoom()

    return cap, detector, tracker, zoom

def process_frame(frame, detector, tracker, zoom, id_map, current_frame, executor_thread):
    width, height = 800, 600
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

        handle_recognition(zoomed_face, track_id, current_frame, x1, y1, x2, y2, executor_thread,db_path=database)

        draw_annotations(frame, x1, y1, x2, y2, track_id)

    return frame

def draw_annotations(frame, x1, y1, x2, y2, track_id):
    text = recognized_faces.get(track_id, "Bilinmiyor")

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    text_display = f"ID {track_id} - {text}"
    (text_width, text_height), baseline = cv2.getTextSize(text_display, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (255, 0, 0), thickness=-1)
    cv2.putText(frame, text_display, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)

def handle_recognition(
        zoomed_face,
        track_id,
        current_frame, x1, y1, x2, y2,
        executor_thread,
        db_path):
    global recognized_faces, processing_faces, last_update_frame, face_center

    if loading_embeddings:
        print("Tanıma işlemi şu an yapılamaz embedding hesaplanıyor.")
        return

    if track_id not in recognized_faces:
        recognized_faces[track_id] = "Bilinmiyor"

    if zoomed_face is None:
        return

    if track_id not in processing_faces:
        processing_faces.add(track_id)
        executor_thread.submit(recognize_face, zoomed_face, track_id, current_frame, x1, y1, x2, y2, db_path)


def main_loop(mode="summary"):
    global loading_embeddings

    dataset_path = database_label.selected_path

    embedding_file = os.path.join(dataset_path,
                                  f"ds_model_facenet_detector_opencv_aligned_normalization_base_expand_0.pkl")

    if not os.path.exists(embedding_file):
        print(f"Embedding bulunamadı. Önce hesaplanıyor...")
        loading_embeddings = True
        generate_embeddings(dataset_path)
        loading_embeddings = False

    if loading_embeddings:
        print("Embedding hesaplanıyor, lütfen bekleyin...")
        return

    cap, detector, tracker, zoom = initialize_system()
    if cap is None:
        print("Sistem başlatılamadı, model yüklenemedi.")
        return

    id_map = {}
    current_frame = 0
    executor_thread = ThreadPoolExecutor(max_workers=2)

    if mode == "summary":
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Kamera bağlantısı koptu, yeniden başlatılıyor")
                cap.release()
                cap = cv2.VideoCapture(0)
                continue

            frame = process_frame(frame, detector, tracker, zoom, id_map, current_frame, executor_thread)

            if frame is not None:
                cv2.imshow(f"Face Tracking + Recognition (Summary)", frame)

            current_frame += 1

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                final_summary = {}
                for track_id, predictions in summary_predictions.items():
                    most_common_prediction = max(set(predictions), key=predictions.count)
                    final_summary[track_id] = most_common_prediction

                print("\n--- Özet Çıkarılan Kişiler ---")
                for track_id, identity in final_summary.items():
                    print(f"ID {track_id}: {identity}")
                print("-----------------------------------")

                cap.release()
                cv2.destroyAllWindows()
                return final_summary

        cap.release()
        cv2.destroyAllWindows()

    elif mode == "individual":
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = process_frame(frame, detector, tracker, zoom, id_map, current_frame, executor_thread)

            if frame is not None:
                cv2.imshow(f"Face Tracking + Recognition (Individual)", frame)

            current_frame += 1

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n--- Tanımlanan Kişiler (Q Basıldı) ---")
                for track_id, identity in recognized_faces.items():
                    identity = recognized_faces.get(track_id, "Bilinmiyor")

                    if identity != "Bilinmiyor":
                        existing_entry = next((entry for entry in final_recognized if entry[0] == track_id), None)

                        if existing_entry:
                            final_recognized.remove(existing_entry)
                            final_recognized.append((track_id, identity))
                            print(f"ID {track_id}: {identity} Güncellendi")
                        else:
                            final_recognized.append((track_id, identity))
                            print(f"ID {track_id}: {identity} Kaydedildi")
                    else:
                        print(f"ID {track_id}: {identity} Tanımlanamadı kaydedilmedi")

                print("-----------------------------------")

            elif key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


def open_database():
    global database
    database = filedialog.askdirectory(title="Veri Klasörü Seçiniz")
    if database:
        database_label.config(text=f"Seçilen Klasör:\n{database}")
        database_label.selected_path = database
        live_button.config(state="normal")
        individual_button.config(state="normal")

def select_excel_file(label):
    file_path = filedialog.askopenfilename(
        title="Kişiler Excel Dosyasını Seç",
        filetypes=[("Excel Dosyası", "*.xlsx"), ("CSV Dosyası", "*.csv"), ("Tüm Dosyalar", "*.*")]
    )
    if file_path:
        label.config(text=f"Seçilen Kişiler Dosyası:\n{file_path}")

def open_live_summary():
    root.withdraw()
    new_window = tk.Toplevel()
    new_window.title("Canlı Tekte Özet Çıkarma")
    new_window.geometry("400x500")

    tk.Label(new_window, text="Canlı Tekte Özet Çıkarma Modu Açıldı!", font=("Helvetica", 12)).pack(pady=10)

    tk.Label(new_window, text="Kaynak Seçiniz:", font=("Helvetica", 10)).pack()
    source_options = ["Canlı Kamera", "Video Dosyası", "Resim Dosyası"]
    source_selection = ttk.Combobox(new_window, textvariable=selected_source, values=source_options, state="readonly")
    source_selection.pack(pady=5)

    tk.Label(new_window, text="Ders İsmi:", font=("Helvetica", 10)).pack()
    course_name_entry = tk.Entry(new_window, width=30)
    course_name_entry.pack(pady=5)

    tk.Label(new_window, text="Kişiler Excel Dosyanızı Seçin:", font=("Helvetica", 10)).pack()
    excel_label = tk.Label(new_window, text="Henüz seçilmedi", font=("Helvetica", 10), fg="red")
    excel_label.pack(pady=5)

    tk.Button(new_window, text="Kişiler Excel Dosyası Seç", command=lambda: select_excel_file(excel_label)).pack(pady=5)

    start_button = tk.Button(new_window, text="İşlemleri Başlat", font=("Helvetica", 10, "bold"), fg="white", bg="green",
                             command=lambda: main_loop("summary"))
    start_button.pack(pady=10)

    new_window.protocol("WM_DELETE_WINDOW", lambda: reopen_main(new_window))

def open_individual_recognition():
    root.withdraw()
    new_window = tk.Toplevel()
    new_window.title("Tek Tek Tanımlama")
    new_window.geometry("400x500")

    tk.Label(new_window, text="Tek Tek Tanımlama Modu Açıldı!", font=("Helvetica", 12)).pack(pady=10)

    tk.Label(new_window, text="Kaynak Seçiniz:", font=("Helvetica", 10)).pack()
    source_options = ["Canlı Kamera", "Video Dosyası", "Resim Dosyası"]
    source_selection = ttk.Combobox(new_window, textvariable=selected_source, values=source_options, state="readonly")
    source_selection.pack(pady=5)

    tk.Label(new_window, text="Ders İsmi:", font=("Helvetica", 10)).pack()
    course_name_entry = tk.Entry(new_window, width=30)
    course_name_entry.pack(pady=5)

    tk.Label(new_window, text="Kişiler Excel Dosyanızı Seçin:", font=("Helvetica", 10)).pack()
    excel_label = tk.Label(new_window, text="Henüz seçilmedi", font=("Helvetica", 10), fg="red")
    excel_label.pack(pady=5)

    tk.Button(new_window, text="Kişiler Excel Dosyası Seç", command=lambda: select_excel_file(excel_label)).pack(pady=5)

    start_button = tk.Button(new_window, text="İşlemleri Başlat", font=("Helvetica", 10, "bold"), fg="white", bg="green",
                             command=lambda: main_loop("individual"))
    start_button.pack(pady=10)

    new_window.protocol("WM_DELETE_WINDOW", lambda: reopen_main(new_window))

def reopen_main(window):
    window.destroy()
    root.deiconify()

label = tk.Label(root, text="Hoşgeldiniz", font=("Helvetica", 14, "bold"))
label.pack(pady=10)

database_button = tk.Button(root, text="Veri Klasörü Seç", width=20, command=open_database)
database_button.pack(pady=10)

database_label = tk.Label(root, text="Henüz klasör seçilmedi.", font=("Helvetica", 10), fg="red")
database_label.pack(pady=5)
database_label.selected_path = ""


live_button = tk.Button(root, text="Canlı Özet Çıkarma", width=20, command=open_live_summary, state="disabled")
live_button.pack(pady=10)

individual_button = tk.Button(root, text="Tek Tek Tanımlama", width=20, command=open_individual_recognition, state="disabled")
individual_button.pack(pady=10)

github_label_1 = tk.Label(root, text="https://github.com/hopepi", fg="blue", cursor="hand2")
github_label_1.pack(pady=5)

github_label_2 = tk.Label(root, text="https://github.com/EfeCyber", fg="blue", cursor="hand2")
github_label_2.pack(pady=5)

root.mainloop()
