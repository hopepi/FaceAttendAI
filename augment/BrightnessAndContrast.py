import cv2

class ImageAdjuster:
    def __init__(self, alpha=1.5, beta=30):
        self.alpha = alpha
        self.beta = beta

    def adjust_image(self, image_path, output_path):

        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} yüklenemedi!")
            return

        adjusted_image = cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)
        cv2.imwrite(output_path, adjusted_image)

        print(f"Parlaklık & Kontrast ayarlandı ve kaydedildi: {output_path}")
