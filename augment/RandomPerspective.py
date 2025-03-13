import cv2
import numpy as np
import random

class RandomPerspectiveTransform:
    def __init__(self, max_offset=0.1):
        if not (0 < max_offset < 1):
            raise ValueError("max_offset değeri 0 ile 1 arasında olmalıdır.")

        self.max_offset = max_offset

    def apply_perspective_transform(self, image_path, output_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} yüklenemedi, dosya yolunu kontrol edin.")
            return

        (h, w) = image.shape[:2]

        original_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

        offset_x = int(w * self.max_offset)
        offset_y = int(h * self.max_offset)

        new_points = np.float32([
            [random.randint(0, offset_x), random.randint(0, offset_y)],
            [w - random.randint(0, offset_x), random.randint(0, offset_y)],
            [random.randint(0, offset_x), h - random.randint(0, offset_y)],
            [w - random.randint(0, offset_x), h - random.randint(0, offset_y)]
        ])

        perspective_matrix = cv2.getPerspectiveTransform(original_points, new_points)
        perspective_image = cv2.warpPerspective(image, perspective_matrix, (w, h))

        cv2.imwrite(output_path, perspective_image)
        print(f"Fotoğrafa rastgele perspektif değişikliği uygulandı ve kaydedildi: {output_path}")
