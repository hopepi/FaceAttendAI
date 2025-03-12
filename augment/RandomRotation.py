import cv2
import numpy as np
import random

# Fotoğrafı yükle
image_path = "../efefoto.jpeg"  # Fotoğrafın dosya yolunu belirtin
image = cv2.imread(image_path)

# Rastgele bir döndürme açısı belirle (örneğin -15 ile +15 derece arasında)
angle = random.uniform(-15, 15)  # -15 ile +15 derece arasında rastgele bir açı

# Fotoğrafın boyutlarını al
(h, w) = image.shape[:2]

# Döndürme işlemi için merkez noktasını belirle
center = (w // 2, h // 2)

# Döndürme matrisini oluştur
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

# Döndürülmüş fotoğrafı oluştur
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

# Döndürülmüş fotoğrafı kaydet
output_path = "../RandomRotation.jpg"
cv2.imwrite(output_path, rotated_image)

print(f"Fotoğraf {angle:.2f} derece döndürüldü ve kaydedildi: {output_path}")

# (Opsiyonel) Döndürülmüş fotoğrafı ekranda göster
cv2.imshow("Rotated Image", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()