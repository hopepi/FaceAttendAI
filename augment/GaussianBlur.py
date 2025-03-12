import cv2
import numpy as np

# Fotoğrafı yükle
image_path = "../efefoto.jpeg"  # Fotoğrafın dosya yolunu belirtin
image = cv2.imread(image_path)

# Gaussian Blur uygula
kernel_size = (15, 15)  # Bulanıklaştırma çekirdeği boyutu (tek sayılar olmalı)
blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX=0)

# Bulanıklaştırılmış fotoğrafı kaydet
output_path = "../GaussianBlur.jpg"
cv2.imwrite(output_path, blurred_image)

print(f"Fotoğrafa Gaussian Blur uygulandı ve kaydedildi: {output_path}")

# (Opsiyonel) Bulanıklaştırılmış fotoğrafı ekranda göster
cv2.imshow("Blurred Image", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()