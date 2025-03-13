import cv2
import numpy as np
import random

class RandomErasing:
    def __init__(self, erase_ratio=0.02, erase_color=0, num_patches=1):

        if not (0 < erase_ratio < 1):
            raise ValueError("erase_ratio değeri 0 ile 1 arasında olmalıdır!!!!!!!!")

        self.erase_ratio = erase_ratio
        self.erase_color = erase_color
        self.num_patches = num_patches

    def apply_erasing(self, image_path, output_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} yüklenemedi, dosya yolunu kontrol edin.")
            return

        h, w, _ = image.shape

        for _ in range(self.num_patches):
            erase_area = int(h * w * self.erase_ratio)
            erase_width = int(np.sqrt(erase_area))
            erase_height = erase_width

            start_x = random.randint(0, w - erase_width)
            start_y = random.randint(0, h - erase_height)

            image[start_y:start_y + erase_height, start_x:start_x + erase_width] = self.erase_color

        cv2.imwrite(output_path, image)
        print(f"Random Erasing uygulandı ve kaydedildi: {output_path}")
