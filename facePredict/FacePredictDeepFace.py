import cv2
from deepface import DeepFace
import numpy as np

# Veritabanının olduğu dizini belirt
db_path = r"C:\Users\umutk\OneDrive\Belgeler\fasas\dataset"

# OpenCV ile kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Kameradan görüntü al
    if not ret:
        break

    # OpenCV ile yüz tespiti için yüz tanıma modeli yükle
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Yüz bölgesini al

        # DeepFace ile yüzü veritabanında ara
        try:
            result = DeepFace.find(face, db_path=db_path, model_name="Facenet", enforce_detection=False)
            if len(result) > 0 and not result[0].empty:
                identity = result[0]["identity"][0].split("\\")[-2]  # Klasör ismi kişiyi temsil eder
                text = f"{identity}"
            else:
                text = "Bilinmiyor"
        except Exception as e:
            text = "Hata"

        # Yüz etrafına dikdörtgen çiz ve ismi ekrana yaz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Kameradan alınan görüntüyü göster
    cv2.imshow("Yüz Tanıma", frame)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
