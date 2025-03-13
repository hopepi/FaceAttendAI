import cv2
import numpy as np
import random

class RandomRotation:
    def __init__(self, angle_range=(-15, 15), border_mode=cv2.BORDER_REFLECT):
        if not (isinstance(angle_range, tuple) and len(angle_range) == 2):
            raise ValueError("angle_range bir çift min, max değerinden oluşmalıdır.")

        self.angle_range = angle_range
        self.border_mode = border_mode

    def apply_rotation(self, image_path, output_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} yüklenemedi, dosya yolunu kontrol edin.")
            return

        (h, w) = image.shape[:2]

        # Rastgele bir açı seç
        angle = random.uniform(*self.angle_range)
        center = (w // 2, h // 2)

        # Döndürme matrisini oluştur
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Görüntüyü döndür
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=self.border_mode)

        cv2.imwrite(output_path, rotated_image)
        print(f"Fotoğraf {angle:.2f} derece döndürüldü ve kaydedildi: {output_path}")
