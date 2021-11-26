import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from PIL import Image
import pdb
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models.vgg as vgg
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from dataset import *
from loss import *
from util import *

import torch.nn as nn
import torch.nn.functional as F
import torch

epochs = 1500
lr = 1e-4
batch_size = 1
b1 = 0.5
b2 = 0.999
decay_epoch = 40
num_workers = 8
img_size = 256
channels = 3
data_name = "ISTD"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if data_name == 'ISTD':
    train_set = ShadowSet('./datasets/ISTD/train', transform)
    test_set = ShadowSet('./datasets/ISTD/test', transform)
    x, y, z, name = train_set[0]
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    print("load data finished!")

wavelet_dec = WaveletTransform(scale=2, dec=True)
wavelet_rec = WaveletTransform(scale=2, dec=False)
# generator = Net()
# optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(b1, b2))

step = 0
for epoch in range(0, epochs):
    for i, batch in enumerate(train_loader):
        step += 1
        iter = i
        image = batch[0].cuda()
        mask = batch[1].cuda()
        real = batch[2].cuda()
        name = batch[3]
        current_lr = 0.0002 * 0.5 ** (step/100000)

        x_r = (image[:, 0, :, :]*255-105.648186)/255.+0.5
        x_g = (image[:, 1, :, :]*255-95.4836)/255.+0.5
        x_b = (image[:, 2, :, :]*255-86.4105)/255.+0.5
        image = torch.cat(
            [x_r.unsqueeze(1), x_g.unsqueeze(1), x_b.unsqueeze(1)], 1)

        y_r = ((image[:, 0, :, :]-0.5)*255+121.2556)/255.
        y_g = ((image[:, 1, :, :]-0.5)*255+114.89969)/255.
        y_b = ((image[:, 2, :, :]-0.5)*255+102.02478)/255.
        image = torch.cat(
            [y_r.unsqueeze(1), y_g.unsqueeze(1), y_b.unsqueeze(1)], 1)
