import cv2
import numpy as np
import random

# Fotoğrafı yükle
image_path = "../efefoto.jpeg"  # Fotoğrafın dosya yolunu belirtin
image = cv2.imread(image_path)

# Fotoğrafın orijinal boyutlarını al
(h, w) = image.shape[:2]

# Rastgele kırpma için boyutları belirle
crop_ratio = random.uniform(0.7, 0.9)  # Orijinal boyutun %70-%90'ı arasında kırpma
new_width = int(w * crop_ratio)
new_height = int(h * crop_ratio)

# Rastgele başlangıç noktasını belirle
start_x = random.randint(0, w - new_width)
start_y = random.randint(0, h - new_height)

# Kırpma işlemini uygula
cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]

# Yeni boyutları belirle (örneğin, orijinal boyuta ölçeklendirme)
resized_width = w  # Orijinal genişlik
resized_height = h  # Orijinal yükseklik

# Ölçeklendirme işlemini uygula
resized_image = cv2.resize(cropped_image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

# İşlenmiş fotoğrafı kaydet
output_path = "../Random_Resized_Crop.jpg"
cv2.imwrite(output_path, resized_image)

print(f"Fotoğraf kırpıldı ve ölçeklendirildi. Kaydedildi: {output_path}")

# (Opsiyonel) İşlenmiş fotoğrafı ekranda göster
cv2.imshow("Resized Cropped Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()