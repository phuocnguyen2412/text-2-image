import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class FlowerDataset(Dataset):
    def __init__(self, image_dir, captions, transform=None):
        self.image_dir = image_dir
        self.captions = captions
        self.transform = transform
        self.image_names = list(captions.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_dir, image_name + ".jpg")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Lấy wrong image khác ảnh hiện tại
        wrong_index = index
        while wrong_index == index:
            wrong_index = torch.randint(0, len(self.image_names), (1,)).item()

        wrong_image_name = self.image_names[wrong_index]
        wrong_image_path = os.path.join(self.image_dir, wrong_image_name + ".jpg")
        wrong_image = Image.open(wrong_image_path).convert("RGB")
        if self.transform:
            wrong_image = self.transform(wrong_image)

        return {
            "wrong_image": wrong_image,
            "image": image,
            "text": self.captions[image_name]["text"],
            "embed": torch.Tensor(self.captions[image_name]["embed"])
        }
