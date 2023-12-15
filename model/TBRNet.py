import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        # https://pytorch.org/hub/pytorch_vision_vgg/
        mean = np.array(
            [0.485, 0.456, 0.406], dtype=np.float32)
        mean = mean.reshape((1, 3, 1, 1))
        self.mean = torch.from_numpy(mean).cuda()
        std = np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)
        std = std.reshape((1, 3, 1, 1))
        self.std = torch.from_numpy(std).cuda()
        self.initial_model()

    def forward(self, x):
        relu1_1 = self.relu1_1((x-self.mean)/self.std)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


    def initial_model(self):
            vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            features = vgg19.features
            self.relu1_1 = torch.nn.Sequential()
            self.relu1_2 = torch.nn.Sequential()

            self.relu2_1 = torch.nn.Sequential()
            self.relu2_2 = torch.nn.Sequential()

            self.relu3_1 = torch.nn.Sequential()
            self.relu3_2 = torch.nn.Sequential()
            self.relu3_3 = torch.nn.Sequential()
            self.relu3_4 = torch.nn.Sequential()

            self.relu4_1 = torch.nn.Sequential()
            self.relu4_2 = torch.nn.Sequential()
            self.relu4_3 = torch.nn.Sequential()
            self.relu4_4 = torch.nn.Sequential()

            self.relu5_1 = torch.nn.Sequential()
            self.relu5_2 = torch.nn.Sequential()
            self.relu5_3 = torch.nn.Sequential()
            self.relu5_4 = torch.nn.Sequential()

            for x in range(2):
                self.relu1_1.add_module(str(x), features[x])

            for x in range(2, 4):
                self.relu1_2.add_module(str(x), features[x])

            for x in range(4, 7):
                self.relu2_1.add_module(str(x), features[x])

            for x in range(7, 9):
                self.relu2_2.add_module(str(x), features[x])

            for x in range(9, 12):
                self.relu3_1.add_module(str(x), features[x])

            for x in range(12, 14):
                self.relu3_2.add_module(str(x), features[x])

            for x in range(14, 16):
                self.relu3_3.add_module(str(x), features[x])

            for x in range(16, 18):
                self.relu3_4.add_module(str(x), features[x])

            for x in range(18, 21):
                self.relu4_1.add_module(str(x), features[x])

            for x in range(21, 23):
                self.relu4_2.add_module(str(x), features[x])

            for x in range(23, 25):
                self.relu4_3.add_module(str(x), features[x])

            for x in range(25, 27):
                self.relu4_4.add_module(str(x), features[x])

            for x in range(27, 30):
                self.relu5_1.add_module(str(x), features[x])

            for x in range(30, 32):
                self.relu5_2.add_module(str(x), features[x])

            for x in range(32, 34):
                self.relu5_3.add_module(str(x), features[x])

            for x in range(34, 36):
                self.relu5_4.add_module(str(x), features[x])


class BatchNorm_(nn.Module):
    def __init__(self, channels):
        super(BatchNorm_, self).__init__()
        self.w0 = torch.nn.Parameter(
            torch.FloatTensor([1.0]), requires_grad=True)
        self.w1 = torch.nn.Parameter(
            torch.FloatTensor([0.0]), requires_grad=True)
        self.BatchNorm2d = nn.BatchNorm2d(
            channels, affine=True, track_running_stats=False)

    def forward(self, x):
        outputs = self.w0*x+self.w1*self.BatchNorm2d(x)
        return outputs
    

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


def conv2d_layer(in_channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, norm="batch", activation_fn="LeakyReLU", conv_mode="none", pad_mode="ReflectionPad2d"):
    """
    norm: batch, spectral, instance, spectral_instance, none

    activation_fn: Sigmoid, ReLU, LeakyReLU, none

    conv_mode: transpose, upsample, none

    pad_mode: ReflectionPad2d, ReplicationPad2d, ZeroPad2d
    """
    layer = []
    # padding
    if conv_mode == "transpose":
        pass
    else:
        if pad_mode == "ReflectionPad2d":
            layer.append(nn.ReflectionPad2d(padding))
        elif pad_mode == "ReplicationPad2d":
            layer.append(nn.ReflectionPad2d(padding))
        else:
            layer.append(nn.ZeroPad2d(padding))
        padding = 0

    # conv layer
    if norm == "spectral" or norm == "spectral_instance":
        bias = False
        # conv
        if conv_mode == "transpose":
            conv_ = nn.ConvTranspose2d
        elif conv_mode == "upsample":
            layer.append(nn.Upsample(mode='bilinear', scale_factor=stride))
            conv_ = nn.Conv2d
        else:
            conv_ = nn.Conv2d
    else:
        bias = True
        # conv
        if conv_mode == "transpose":
            layer.append(nn.ConvTranspose2d(in_channels, channels, kernel_size,
                                            bias=bias, stride=stride, padding=padding, dilation=dilation))
        elif conv_mode == "upsample":
            layer.append(nn.Upsample(mode='bilinear', scale_factor=stride))
            layer.append(nn.Conv2d(in_channels, channels, kernel_size,
                                   bias=bias, stride=stride, padding=padding, dilation=dilation))
        else:
            layer.append(nn.Conv2d(in_channels, channels, kernel_size,
                                   bias=bias, stride=stride, padding=padding, dilation=dilation))

    # norm
    if norm == "spectral":
        layer.append(spectral_norm(conv_(in_channels, channels, kernel_size,
                                         stride=stride, bias=bias, padding=padding, dilation=dilation), True))
    elif norm == "instance":
        layer.append(nn.InstanceNorm2d(
            channels, affine=True, track_running_stats=False))
    elif norm == "batch":
        layer.append(nn.BatchNorm2d(
            channels, affine=True, track_running_stats=False))
    elif norm == "spectral_instance":
        layer.append(spectral_norm(conv_(in_channels, channels, kernel_size,
                                         stride=stride, bias=bias, padding=padding, dilation=dilation), True))
        layer.append(nn.InstanceNorm2d(
            channels, affine=True, track_running_stats=False))
    elif norm == "batch_":
        layer.append(BatchNorm_(channels))
    else:
        pass

    # activation_fn
    if activation_fn == "Sigmoid":
        layer.append(nn.Sigmoid())
    elif activation_fn == "ReLU":
        layer.append(nn.ReLU(True))
    elif activation_fn == "none":
        pass
    else:
        layer.append(nn.LeakyReLU(0.2,inplace=True))

    return nn.Sequential(*layer)


def avgcov2d_layer(pool_kernel_size, pool_stride, in_channels, channels, conv_kernel_size=3, conv_stride=1, padding=1, dilation=1, norm="batch", activation_fn="LeakyReLU"):
    layer = []
    layer.append(nn.AvgPool2d(pool_kernel_size, pool_stride))
    layer.append(conv2d_layer(in_channels, channels, kernel_size=conv_kernel_size, stride=conv_stride,
                              padding=padding, dilation=dilation, norm=norm, activation_fn=activation_fn))
    return nn.Sequential(*layer)




class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='identity', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(
                        m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "identity":
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, gain)
                    else:
                        identity_initializer(m.weight.data)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


def identity_initializer(data):
    shape = data.shape
    array = np.zeros(shape, dtype=float)
    cx, cy = shape[2]//2, shape[3]//2
    for i in range(np.minimum(shape[0], shape[1])):
        array[i, i, cx, cy] = 1
    return torch.tensor(array, dtype=torch.float32)


class TBRNet(BaseNetwork):
    """TBRNet"""

    def __init__(self, in_channels=3):
        super(TBRNet, self).__init__()

        # gan
        self.network = TBRNetSOURCE(
            in_channels, 64, norm='batch', stage_num=[6,2])

        self.init_weights()

    def forward(self, x):
        outputs = self.network(x)
        return outputs


class TBRNetSOURCE(nn.Module):
    def __init__(self, in_channels=3, channels=64, norm="batch", stage_num=[6,2]):
        super(TBRNetSOURCE, self).__init__()
        self.stage_num = stage_num

        # Pre-trained VGG19
        self.add_module('vgg19', VGG19())

        # SE
        cat_channels = in_channels+64+128+256+512+512
        self.se = SELayer(cat_channels)
        self.conv_up = conv2d_layer(
            cat_channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)
        self.conv_mid = conv2d_layer(
            cat_channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)
        self.conv_down = conv2d_layer(
            cat_channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)

        # coarse
        self.down_sample = conv2d_layer(
            channels, 2*channels, kernel_size=4, stride=2, padding=1, dilation=1, norm=norm)
        coarse_list = []
        for i in range(self.stage_num[0]):
            coarse_list.append(TBR(2*channels, norm, mid_dilation=2**(i % 6)))
        self.coarse_list = nn.Sequential(*coarse_list)

        self.up_conv = conv2d_layer(
            2*channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, norm=norm)

        # refine
        fine_list = []
        for i in range(self.stage_num[1]):
            fine_list.append(TBR(channels, norm, mid_dilation=2**(i % 6)))
        self.fine_list = nn.Sequential(*fine_list)

        self.se_coarse = nn.Sequential(SELayer(2*channels),
                                       conv2d_layer(2*channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm))

        self.spp1 = SPP(channels, norm=norm)
        self.spp2 = SPP(channels, norm=norm)
        self.spp3 = SPP(channels, norm=norm)

        self.toimg1 = conv2d_layer(
            channels, 3, kernel_size=1,  padding=0, dilation=1, norm="none", activation_fn="Sigmoid")
        self.toimg2 = conv2d_layer(
            channels, 3, kernel_size=1,  padding=0, dilation=1, norm="none", activation_fn="Sigmoid")
        self.toimg3 = conv2d_layer(
            channels, 3, kernel_size=1,  padding=0, dilation=1, norm="none", activation_fn="Sigmoid")
    def forward(self, x):
        size = (x.shape[2], x.shape[3])

        # vgg
        x_vgg = self.vgg19(x)

        # hyper-column features
        x_cat = torch.cat((
            x,
            F.interpolate(x_vgg['relu1_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu2_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu3_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu4_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu5_2'], size, mode="bilinear", align_corners=True)), dim=1)

        # SE
        x_se = self.se(x_cat)
        x_up = self.conv_up(x_se)
        x_mid = self.conv_mid(x_se)
        x_down = self.conv_down(x_se)
        x_up_ = x_up
        x_mid_ = x_mid
        x_down_ = x_down
 
        # process
        x_up = self.down_sample(x_up)
        x_mid = self.down_sample(x_mid)
        x_down = self.down_sample(x_down)
        for i in range(self.stage_num[0]):
            x_up, x_mid, x_down = self.coarse_list[i](x_up, x_mid, x_down)

        x_up = F.interpolate(x_up, size, mode="bilinear", align_corners=True)
        x_up = self.up_conv(x_up)
        x_mid = F.interpolate(x_mid, size, mode="bilinear", align_corners=True)
        x_mid = self.up_conv(x_mid)
        x_down = F.interpolate(x_down, size, mode="bilinear", align_corners=True)
        x_down = self.up_conv(x_down)

        x_up = self.se_coarse(torch.cat((x_up_, x_up), dim=1))
        x_mid = self.se_coarse(torch.cat((x_mid_, x_mid), dim=1))
        x_down = self.se_coarse(torch.cat((x_down_, x_down), dim=1))
        for i in range(self.stage_num[1]):
            x_up, x_mid, x_down = self.fine_list[i](x_up, x_mid, x_down)

        # spp
        img = self.spp1(x_up)
        matte_out = self.spp2(x_mid)
        img_free = self.spp3(x_down)

        # output
        img = self.toimg1(img)
        matte_out = self.toimg2(matte_out)
        img_free = self.toimg3(img_free)

        return [img, matte_out, img_free]


class TBR(nn.Module):
    def __init__(self, channels=64, norm="batch", mid_dilation=1):
        super(TBR, self).__init__()
        # up
        self.conv_up = conv2d_layer(
            channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)
        avg_pool_down_mid = []
        avg_pool_down_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv2d_layer(channels, channels, kernel_size=1,  padding=0, dilation=1, norm="none")))
        avg_pool_down_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            conv2d_layer(channels, channels, kernel_size=3,  padding=0, dilation=1, norm="none")))
        avg_pool_down_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(5),
            conv2d_layer(channels, channels, kernel_size=5,  padding=0, dilation=1, norm="none")))
        avg_pool_down_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            conv2d_layer(channels, channels, kernel_size=7,  padding=0, dilation=1, norm="none")))
        self.avg_pool_down_mid = nn.Sequential(*avg_pool_down_mid)

        # mid
        self.conv_mid = conv2d_layer(
            channels, channels, kernel_size=3,  padding=mid_dilation, dilation=mid_dilation, norm=norm)

        # down
        self.conv_down = conv2d_layer(
            channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)
        avg_pool_up_mid = []
        avg_pool_up_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv2d_layer(channels, channels, kernel_size=1,  padding=0, dilation=1, norm="none")))
        avg_pool_up_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            conv2d_layer(channels, channels, kernel_size=3,  padding=0, dilation=1, norm="none")))
        avg_pool_up_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(5),
            conv2d_layer(channels, channels, kernel_size=5,  padding=0, dilation=1, norm="none")))
        avg_pool_up_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            conv2d_layer(channels, channels, kernel_size=7,  padding=0, dilation=1, norm="none")))
        self.avg_pool_up_mid = nn.Sequential(*avg_pool_up_mid)

        # conv
        self.conv_up_mid = conv2d_layer(
            channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)
        self.conv_up_mid_1x1 = conv2d_layer(
            2*channels, channels, kernel_size=1,  padding=0, dilation=1, norm=norm)
        self.conv_down_mid = conv2d_layer(
            channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)
        self.conv_down_mid_1x1 = conv2d_layer(
            2*channels, channels, kernel_size=1,  padding=0, dilation=1, norm=norm)

    def forward(self, up_in, mid_in, down_in):
        # up
        x_up = self.conv_up(up_in)
        x_down_mid = self.conv_down_mid(down_in-mid_in)
        pooling_up = 0
        for i in range(len(self.avg_pool_down_mid)):
            pooling_up = pooling_up + self.avg_pool_down_mid[i](x_down_mid)
        x_up = pooling_up*x_up + x_up + x_down_mid*x_up + up_in

        # mid
        x_mid = self.conv_mid(mid_in)
        x_mid = x_mid + mid_in

        # down
        x_down = self.conv_down(down_in)
        x_up_mid = self.conv_up_mid(up_in+mid_in)
        pooling_down = 0
        for i in range(len(self.avg_pool_up_mid)):
            pooling_down = pooling_down + self.avg_pool_up_mid[i](x_up_mid)
        x_down = pooling_down*x_down + x_down + x_up_mid*x_down + down_in

        return x_up, x_mid, x_down


class SPP(nn.Module):
    # SPP SOURCE - tensorflow http://github.com/vinthony/ghost-free-shadow-removal/
    def __init__(self, channels=64, norm="batch"):
        super(SPP, self).__init__()
        self.net2 = avgcov2d_layer(
            4, 4, channels, channels, 1, padding=0, norm=norm)
        self.net8 = avgcov2d_layer(
            8, 8, channels, channels, 1, padding=0, norm=norm)
        self.net16 = avgcov2d_layer(
            16, 16, channels, channels, 1, padding=0, norm=norm)
        self.net32 = avgcov2d_layer(
            32, 32, channels, channels, 1, padding=0, norm=norm)
        self.output = conv2d_layer(channels*5, channels, 3, norm=norm)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = torch.cat((
            F.interpolate(self.net2(x), size, mode="bilinear",
                          align_corners=True),
            F.interpolate(self.net8(x), size, mode="bilinear",
                          align_corners=True),
            F.interpolate(self.net16(x), size,
                          mode="bilinear", align_corners=True),
            F.interpolate(self.net32(x), size,
                          mode="bilinear", align_corners=True),
            x), dim=1)
        x = self.output(x)
        return x


class SELayer(nn.Module):
    # https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py -> https://github.com/vinthony/ghost-free-shadow-removal/blob/master/networks.py
    # reduction=16 -> reduction=8
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = TBRNet().cuda()
    res = model(t)
    print(res[0].shape)