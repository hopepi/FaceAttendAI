import cv2
import numpy as np

# Fotoğrafı yükle
image_path = "../efefoto.jpeg"  # Fotoğrafın dosya yolunu belirtin
image = cv2.imread(image_path)

# Gaussian Noise ekleme fonksiyonu
def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))  # Gaussian dağılımına göre gürültü oluştur
    noisy_image = image + gauss  # Gürültüyü fotoğrafa ekle
    noisy_image = np.clip(noisy_image, 0, 255)  # Piksel değerlerini 0-255 aralığına sınırla
    noisy_image = noisy_image.astype(np.uint8)  # Veri tipini uint8'e dönüştür
    return noisy_image

# Gaussian Noise ekle
noisy_image = add_gaussian_noise(image, mean=0, sigma=25)

# Gürültülü fotoğrafı kaydet
output_path = "../GaussianNoise.jpg"
cv2.imwrite(output_path, noisy_image)

print(f"Fotoğrafa Gaussian Noise eklendi ve kaydedildi: {output_path}")

# (Opsiyonel) Gürültülü fotoğrafı ekranda göster
cv2.imshow("Noisy Image", noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()