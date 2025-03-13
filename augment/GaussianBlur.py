import cv2

class GaussianBlurProcessor:
    def __init__(self, kernel_size=(15, 15)):

        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError("Kernel boyutu tek sayı olmalıdır!!!!!!!!!!!")

        self.kernel_size = kernel_size

    def apply_blur(self, image_path, output_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} yüklenemedi!!!!!!!!")
            return

        blurred_image = cv2.GaussianBlur(image, self.kernel_size, sigmaX=0)
        cv2.imwrite(output_path, blurred_image)

        print(f"Gaussian Blur uygulandı ve kaydedildi: {output_path}")
