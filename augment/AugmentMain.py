from augment.Augment import Augment

if __name__ == "__main__":
    augmentor = Augment()
    """
    Gerekirse Özelleştirme
    augmentor = Augment(
        alpha=2.0, beta=40,
        kernel_size=(5, 5),
        mean=5, sigma=30,
        flip_mode=FlipMode.ALL,
        erase_ratio=0.05, erase_color=255, num_patches=3,
        max_offset=0.2,
        crop_ratio_range=(0.6, 0.9),
        angle_range=(-30, 30)
    )
    """

    dataset_way = r"C:\Users\umutk\OneDrive\Belgeler\Deneme\asd"
    augmentor.organize_dataset(
        source_dir=r"C:\Users\umutk\OneDrive\Masaüstü\Data1\2",
        target_dir=r"C:\Users\umutk\OneDrive\Belgeler\Deneme\asd",
        image_size=(160,160)
    )
    augmentor.get_image_paths(dataset_path=dataset_way)
    augmentor.apply_augmentations(
        dataset_way,
        brightness_contrast=True,
        gaussian_blur=True,
        gaussian_noise=True,
        grayscale=True,
        flip=True,
        random_erasing=True,
        perspective_transform=True,
        random_resized_crop=True,
        random_rotation=True
    )