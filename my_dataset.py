import cv2

from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """custom the dataset"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # img = Image.open(self.images_path[item])
        
        img_path = self.images_path[item]
        image = cv2.imread(img_path)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img = img.convert('RGB')

        # L is the gray image
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item].convert('RGB')))
        label = self.images_class[item]

        if self.transform is not None:
            # img = self.transform(img)
            img = self.transform(image=img)['image']
        return img, label

    @staticmethod
    def collate_fn(batch):
        # official default_collate
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels