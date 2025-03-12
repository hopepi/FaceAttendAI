import cv2
import numpy as np
import random

# Fotoğrafı yükle
image_path = "../efefoto.jpeg"  # Fotoğrafın dosya yolunu belirtin
image = cv2.imread(image_path)

# Fotoğrafın boyutlarını al
(h, w) = image.shape[:2]

# Rastgele perspektif değişikliği için 4 köşe noktası belirle
# Orijinal köşe noktaları
original_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

# Rastgele küçük ofsetler oluştur (perspektif etkisi için)
max_offset = 0.1  # Maksimum ofset oranı (örneğin, genişliğin veya yüksekliğin %10'u)
offset_x = int(w * max_offset)
offset_y = int(h * max_offset)

# Yeni köşe noktalarını rastgele oluştur
new_points = np.float32([
    [random.randint(-offset_x, offset_x), random.randint(-offset_y, offset_y)],  # Sol üst
    [w + random.randint(-offset_x, offset_x), random.randint(-offset_y, offset_y)],  # Sağ üst
    [random.randint(-offset_x, offset_x), h + random.randint(-offset_y, offset_y)],  # Sol alt
    [w + random.randint(-offset_x, offset_x), h + random.randint(-offset_y, offset_y)]  # Sağ alt
])

# Perspektif dönüşüm matrisini hesapla
perspective_matrix = cv2.getPerspectiveTransform(original_points, new_points)

# Perspektif dönüşümünü uygula
perspective_image = cv2.warpPerspective(image, perspective_matrix, (w, h))

# Perspektif değiştirilmiş fotoğrafı kaydet
output_path = "../RandomPerspective.jpg"
cv2.imwrite(output_path, perspective_image)

print(f"Fotoğrafa perspektif değişikliği uygulandı ve kaydedildi: {output_path}")

# (Opsiyonel) Perspektif değiştirilmiş fotoğrafı ekranda göster
cv2.imshow("Perspective Transformed Image", perspective_image)
cv2.waitKey(0)
cv2.destroyAllWindows()