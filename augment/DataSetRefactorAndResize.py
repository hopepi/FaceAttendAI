import os
from PIL import Image

class DatasetOrganizer:
    def __init__(self, source_dir=None, target_dir=None, image_size=(160, 160)):

        if source_dir is None or target_dir is None:
            raise ValueError("Hem kaynak yolu hem de çıkış yolu parametreleri girilmelidir.")

        self.source_dir = source_dir
        self.target_dir = target_dir
        self.image_size = image_size
        self.supported_formats = ['.jpg', '.jpeg', '.png']

        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

    def organize_dataset(self):
        files = os.listdir(self.source_dir)

        for file in files:
            file_path = os.path.join(self.source_dir, file)
            file_extension = os.path.splitext(file)[1].lower()

            if os.path.isfile(file_path) and file_extension in self.supported_formats:
                folder_name = os.path.splitext(file)[0]
                folder_path = os.path.join(self.target_dir, folder_name)

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                new_file_path = os.path.join(folder_path, file)

                with Image.open(file_path) as img:
                    img = img.resize(self.image_size)
                    img.save(new_file_path)

                print(f"{file} resized to {self.image_size} and saved in {folder_path}")