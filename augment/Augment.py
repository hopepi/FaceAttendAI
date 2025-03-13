import os
from augment.BrightnessAndContrast import ImageAdjuster
from augment.DataSetRefactorAndResize import DatasetOrganizer
from augment.GaussianBlur import GaussianBlurProcessor
from augment.GaussianNoise import GaussianNoiseProcessor
from augment.Grayscale import GrayscaleConverter
from augment.Flip import ImageFlipper, FlipMode
from augment.RandomErasing import RandomErasing
from augment.RandomPerspective import RandomPerspectiveTransform
from augment.RandomResizedCrop import RandomResizedCrop
from augment.RandomRotation import RandomRotation

class Augment:
    def __init__(
        self,
        alpha=1.5, beta=30,
        kernel_size=(15, 15),
        mean=0, sigma=25,
        flip_mode=FlipMode.ALL,
        erase_ratio=0.02, erase_color=0, num_patches=1,
        max_offset=0.1,
        crop_ratio_range=(0.7, 0.9),
        angle_range=(-15, 15),
    ):
        self.alpha = alpha
        self.beta = beta
        self.kernel_size = kernel_size
        self.mean = mean
        self.sigma = sigma
        self.flip_mode = flip_mode
        self.erase_ratio = erase_ratio
        self.erase_color = erase_color
        self.num_patches = num_patches
        self.max_offset = max_offset
        self.crop_ratio_range = crop_ratio_range
        self.angle_range = angle_range

        self.image_adjuster = ImageAdjuster(alpha=self.alpha, beta=self.beta)
        self.gaussian_blur = GaussianBlurProcessor(kernel_size=self.kernel_size)
        self.gaussian_noise = GaussianNoiseProcessor(mean=self.mean, sigma=self.sigma)
        self.grayscale_converter = GrayscaleConverter()
        self.image_flipper = ImageFlipper(mode=self.flip_mode)
        self.random_erasing = RandomErasing(erase_ratio=self.erase_ratio, erase_color=self.erase_color, num_patches=self.num_patches)
        self.perspective_transform = RandomPerspectiveTransform(max_offset=self.max_offset)
        self.random_resized_crop = RandomResizedCrop(crop_ratio_range=self.crop_ratio_range)
        self.random_rotation = RandomRotation(angle_range=self.angle_range)

    def organize_dataset(self, source_dir, target_dir, image_size=(160, 160)):
        dataset_organizer = DatasetOrganizer(source_dir, target_dir, image_size)
        dataset_organizer.organize_dataset()

        print("Dataset başarıyla yolları oluşturuldu")


    def get_image_paths(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise ValueError("Verilen dataset klasörü bulunamadı!")

        image_paths = []
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)

            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(folder_path, file)
                        file_name, _ = os.path.splitext(file)
                        image_paths.append((file_path, folder_path, file_name))

        return image_paths

    def apply_augmentations(
        self,
        dataset_path,
        brightness_contrast=False,
        gaussian_blur=False,
        gaussian_noise=False,
        grayscale=False,
        flip=False,
        random_erasing=False,
        perspective_transform=False,
        random_resized_crop=False,
        random_rotation=False
    ):
        image_paths = self.get_image_paths(dataset_path)

        for image_path, folder_path, file_name in image_paths:
            if brightness_contrast:
                output_file = os.path.join(folder_path, f"{file_name}_BrightnessContrast.png")
                self.image_adjuster.adjust_image(image_path, output_file)

            if gaussian_blur:
                output_file = os.path.join(folder_path, f"{file_name}_GaussianBlur.png")
                self.gaussian_blur.apply_blur(image_path, output_file)

            if gaussian_noise:
                output_file = os.path.join(folder_path, f"{file_name}_GaussianNoise.png")
                self.gaussian_noise.add_gaussian_noise(image_path, output_file)

            if grayscale:
                output_file = os.path.join(folder_path, f"{file_name}_Grayscale.png")
                self.grayscale_converter.convert_to_grayscale(image_path, output_file)

            if flip:
                output_file = os.path.join(folder_path, f"{file_name}_Flipped.png")
                self.image_flipper.flip_image(image_path, output_file)

            if random_erasing:
                output_file = os.path.join(folder_path, f"{file_name}_RandomErasing.png")
                self.random_erasing.apply_erasing(image_path, output_file)

            if perspective_transform:
                output_file = os.path.join(folder_path, f"{file_name}_PerspectiveTransform.png")
                self.perspective_transform.apply_perspective_transform(image_path, output_file)

            if random_resized_crop:
                output_file = os.path.join(folder_path, f"{file_name}_RandomResizedCrop.png")
                self.random_resized_crop.apply_crop(image_path, output_file)

            if random_rotation:
                output_file = os.path.join(folder_path, f"{file_name}_RandomRotation.png")
                self.random_rotation.apply_rotation(image_path, output_file)

            print(f"{image_path} dosyasına seçilen augmentasyonlar uygulandı.")


