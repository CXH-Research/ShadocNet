import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models.vgg as vgg
import pdb
import math
import numpy as np
import pickle
import os


class WaveletTransform(nn.Module):
    def __init__(self, scale=1, dec=True, params_path='./cache/wavelet_weights_c2.pkl', transpose=True):
        super(WaveletTransform, self).__init__()

        self.scale = scale
        self.dec = dec
        self.transpose = transpose

        ks = int(math.pow(2, self.scale))
        nc = 3 * ks * ks

        if dec:
            self.conv = nn.Conv2d(in_channels=3, out_channels=nc,
                                  kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = open(params_path, 'rb')
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                dct = u.load()
                f.close()
                m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                m.weight.requires_grad = False

    def forward(self, x):
        if self.dec:
            # pdb.set_trace()
            output = self.conv(x)
            if self.transpose:
                osz = output.size()
                output = output.view(
                    osz[0], 3, -1, osz[2], osz[3]).transpose(1, 2).contiguous().view(osz)
        else:
            if self.transpose:
                xx = x
                xsz = xx.size()
                xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]
                             ).transpose(1, 2).contiguous().view(xsz)
            output = self.conv(xx)
        return output
