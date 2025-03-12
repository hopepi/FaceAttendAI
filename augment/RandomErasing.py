import cv2
import numpy as np
import random

# Fotoğrafı yükle
image_path = "../efefoto.jpeg"  # Fotoğrafın dosya yolunu belirtin
image = cv2.imread(image_path)

# Random Erasing fonksiyonu
def random_erasing(image, erase_ratio=0.02):
    h, w, _ = image.shape

    # Silinecek bölgenin boyutlarını hesapla
    erase_area = int(h * w * erase_ratio)  # Silinecek toplam piksel sayısı
    erase_width = int(np.sqrt(erase_area))  # Kare şeklinde bir bölge silmek için genişlik
    erase_height = erase_width

    # Rastgele bir başlangıç noktası belirle
    start_x = random.randint(0, w - erase_width)
    start_y = random.randint(0, h - erase_height)

    # Bölgeyi siyah renkle doldur (veya başka bir renk seçebilirsiniz)
    image[start_y:start_y + erase_height, start_x:start_x + erase_width] = 0

    return image

# Random Erasing uygula (hafif silme için küçük bir erase_ratio kullan)
erase_ratio = 0.005  # Görüntünün %2'sini sil
erased_image = random_erasing(image, erase_ratio)

# İşlenmiş fotoğrafı kaydet
output_path = "RandomErasing2.jpg"
cv2.imwrite(output_path, erased_image)

print(f"Fotoğrafa Random Erasing uygulandı ve kaydedildi: {output_path}")

# (Opsiyonel) İşlenmiş fotoğrafı ekranda göster
cv2.imshow("Random Erased Image", erased_image)
cv2.waitKey(0)
cv2.destroyAllWindows()