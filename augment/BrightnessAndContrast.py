import cv2
import numpy as np

# Fotoğrafı yükle
image_path = "../efefoto.jpeg"  # Fotoğrafın dosya yolunu belirtin
image = cv2.imread(image_path)

# Parlaklık ve kontrast değerlerini belirle
alpha = 1.5  # Kontrast kontrolü (1.0 orijinal, 1.5 artırılmış kontrast)
beta = 30    # Parlaklık kontrolü (0 orijinal, pozitif değerler parlaklık artırır)

# Parlaklık ve kontrast ayarını uygula
adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Değiştirilmiş fotoğrafı kaydet
output_path = "fotoğraf_parlaklik_kontrast_opencv.jpg"
cv2.imwrite(output_path, adjusted_image)

print(f"Fotoğraf kaydedildi: {output_path}")

# (Opsiyonel) Fotoğrafı ekranda göster
cv2.imshow("Adjusted Image", adjusted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()