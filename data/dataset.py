import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'tif'])


class DataLoaderTrain(Dataset):
    def __init__(self, img_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(img_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(img_dir, 'target')))
        mask_files = sorted(os.listdir(os.path.join(img_dir, 'mask')))

        self.inp_filenames = [os.path.join(
            img_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(
            img_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(
            img_dir, 'mask', x) for x in mask_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        mask_path = self.mask_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')
        mask_img = Image.open(mask_path)

        inp_img = TF.to_tensor(inp_img)
        inp_img = TF.resize(inp_img, [256, 256])
        tar_img = TF.to_tensor(tar_img)
        tar_img = TF.resize(tar_img, [256, 256])
        mask_img = TF.to_tensor(mask_img)
        mask_img = TF.resize(mask_img, [256, 256])

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]
        mask_img = mask_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
            mask_img = mask_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
            mask_img = mask_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
            mask_img = torch.rot90(mask_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
            mask_img = torch.rot90(mask_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
            mask_img = torch.rot90(mask_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
            mask_img = torch.rot90(mask_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))
            mask_img = torch.rot90(mask_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, mask_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, img_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(img_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(img_dir, 'target')))
        mask_files = sorted(os.listdir(os.path.join(img_dir, 'mask')))

        self.inp_filenames = [os.path.join(
            img_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(
            img_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(
            img_dir, 'mask', x) for x in mask_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        mask_path = self.mask_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')
        mask_img = Image.open(mask_path)

        inp_img = TF.to_tensor(inp_img)
        inp_img = TF.resize(inp_img, [256, 256])
        tar_img = TF.to_tensor(tar_img)
        tar_img = TF.resize(tar_img, [256, 256])
        mask_img = TF.to_tensor(mask_img)
        mask_img = TF.resize(mask_img, [256, 256])

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, mask_img, filename


class DataLoaderTest(Dataset):
    def __init__(self, img_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(img_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(img_dir, 'target')))
        mask_files = sorted(os.listdir(os.path.join(img_dir, 'mask')))

        self.inp_filenames = [os.path.join(
            img_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(
            img_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(
            img_dir, 'mask', x) for x in mask_files if is_image_file(x)]

        self.img_options = img_options
        self.inp_size = len(self.tar_filenames)

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        inp_path = self.inp_filenames[index]
        tar_path = self.tar_filenames[index]
        mask_path = self.mask_filenames[index]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')
        mask_img = Image.open(mask_path)

        inp_img = TF.to_tensor(inp_img)
        inp_img = TF.resize(inp_img, [256, 256])
        tar_img = TF.to_tensor(tar_img)
        tar_img = TF.resize(tar_img, [256, 256])
        mask_img = TF.to_tensor(mask_img)
        mask_img = TF.resize(mask_img, [256, 256])

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, mask_img, filename
