import cv2
import numpy as np

class GaussianNoiseProcessor:
    def __init__(self, mean=0, sigma=25):
        self.mean = mean
        self.sigma = sigma

    def add_gaussian_noise(self, image_path, output_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} yüklenemedi, dosya yolunu kontrol edin.")
            return

        row, col, ch = image.shape
        gauss = np.random.normal(self.mean, self.sigma, (row, col, ch))
        noisy_image = image.astype(np.float32) + gauss
        noisy_image = np.clip(noisy_image, 0, 255)
        noisy_image = noisy_image.astype(np.uint8)

        cv2.imwrite(output_path, noisy_image)
        print(f"Gaussian Noise uygulandı ve kaydedildi: {output_path}")
