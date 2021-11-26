from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import glob


class ShadowSet(Dataset):
    def __init__(self, data_path, transform=None):
        self.image_dir = os.path.join(data_path, 'image')
        self.mask_dir = os.path.join(data_path, 'mask')
        self.real_dir = os.path.join(data_path, 'real')
        self.image_list = glob.glob(os.path.join(self.image_dir, '*.png'))
        self.mask_list = glob.glob(os.path.join(self.mask_dir, '*.png'))
        self.real_list = glob.glob(os.path.join(self.real_dir, '*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        name = image.filename.split('/')[-1]
        print(name)
        mask = Image.open(self.mask_list[idx])
        real = Image.open(self.real_list[idx])
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
            real = self.transform(real)
        return image, mask, real, name
