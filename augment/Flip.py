import cv2
from enum import Enum

class FlipMode(Enum):
    HORIZONTAL = 1
    VERTICAL = 0
    BOTH = -1
    ALL = "all"

class ImageFlipper:
    def __init__(self, mode=FlipMode.HORIZONTAL):
        """
        FlipMode.HORIZONTAL Yatay çevirme (1)
        FlipMode.VERTICAL Dikey çevirme (0)
        FlipMode.BOTH Hem yatay hem dikey çevirme (-1)
        FlipMode.ALL Tüm flip türlerini uygula
        """
        if not isinstance(mode, FlipMode):
            raise ValueError(f"Geçersiz mod Desteklenen modlar: {list(FlipMode)}")

        self.mode = mode

    def flip_image(self, image_path, output_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} yüklenemedi, dosya yolunu kontrol edin.")
            return

        if self.mode == FlipMode.HORIZONTAL:
            flipped_image = cv2.flip(image, FlipMode.HORIZONTAL.value)
            cv2.imwrite(output_path, flipped_image)

        elif self.mode == FlipMode.VERTICAL:
            flipped_image = cv2.flip(image, FlipMode.VERTICAL.value)
            cv2.imwrite(output_path, flipped_image)

        elif self.mode == FlipMode.BOTH:
            flipped_image = cv2.flip(image, FlipMode.BOTH.value)
            cv2.imwrite(output_path, flipped_image)

        elif self.mode == FlipMode.ALL:
            cv2.imwrite(output_path.replace(".png", "_Horizontal.png"), cv2.flip(image, FlipMode.HORIZONTAL.value))
            cv2.imwrite(output_path.replace(".png", "_Vertical.png"), cv2.flip(image, FlipMode.VERTICAL.value))
            cv2.imwrite(output_path.replace(".png", "_Both.png"), cv2.flip(image, FlipMode.BOTH.value))

        print(f"Resim {self.mode.name} modunda çevrildi ve {output_path} yoluna kaydedildi.")
