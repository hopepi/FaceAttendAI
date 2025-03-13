import cv2

class GrayscaleConverter:
    def __init__(self):
        pass

    def convert_to_grayscale(self, image_path, output_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} yüklenemedi, dosya yolunu kontrol edin!!!!!!!!")
            return

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_path, gray_image)

        print(f"Fotoğraf Siyah-Beyaza dönüştürüldü ve kaydedildi: {output_path}")
