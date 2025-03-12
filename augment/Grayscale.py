import cv2

# Fotoğrafı yükle
image_path = "../efefoto.jpeg"  # Fotoğrafın dosya yolunu belirtin
image = cv2.imread(image_path)

# Fotoğrafı Grayscale (Siyah-Beyaz) formatına dönüştür
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Grayscale fotoğrafı kaydet
output_path = "../Grayscale.jpg"
cv2.imwrite(output_path, gray_image)

print(f"Fotoğraf Siyah-Beyaz'a dönüştürüldü ve kaydedildi: {output_path}")

# (Opsiyonel) Grayscale fotoğrafı ekranda göster
cv2.imshow("Grayscale Image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()