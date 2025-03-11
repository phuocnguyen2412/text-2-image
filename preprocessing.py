import os

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
        return {
            "image": image,
            "text": self.captions[image_name]["text"],
            "embed": self.captions[image_name]["embed"]
        }