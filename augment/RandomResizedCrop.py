import cv2
import numpy as np
import random

class RandomResizedCrop:
    def __init__(self, crop_ratio_range=(0.7, 0.9)):
        if not (0 < crop_ratio_range[0] < 1 and 0 < crop_ratio_range[1] < 1):
            raise ValueError("crop_ratio_range değerleri 0 ile 1 arasında olmalıdır.")

        self.crop_ratio_range = crop_ratio_range

    def apply_crop(self, image_path, output_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} yüklenemedi, dosya yolunu kontrol edin.")
            return

        (h, w) = image.shape[:2]

        crop_ratio = random.uniform(*self.crop_ratio_range)
        new_width = int(w * crop_ratio)
        new_height = int(h * crop_ratio)

        start_x = random.randint(0, w - new_width)
        start_y = random.randint(0, h - new_height)

        cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]

        resized_image = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(output_path, resized_image)
        print(f"Fotoğraf rastgele kırpıldı ve ölçeklendirildi. Kaydedildi: {output_path}")
