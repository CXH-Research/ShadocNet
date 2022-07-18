import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision.transforms.functional import rgb_to_grayscale

from .rasc import *
from .model_init import *

import functools

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


#

class MinimalUnetV2(nn.Module):
    """docstring for MinimalUnet"""

    def __init__(self, down=None, up=None, submodule=None, attention=None, withoutskip=False, **kwags):
        super(MinimalUnetV2, self).__init__()

        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.sub = submodule
        self.attention = attention
        self.withoutskip = withoutskip
        self.is_attention = not self.attention == None
        self.is_sub = not submodule == None

    def forward(self, x, mask=None):
        if self.is_sub:
            x_up, _ = self.sub(self.down(x), mask)
        else:
            x_up = self.down(x)

        if self.withoutskip:  # outer or inner.
            x_out = self.up(x_up)
        else:
            if self.is_attention:
                x_out = (self.attention(torch.cat([x, self.up(x_up)], 1), mask), mask)
            else:
                x_out = (torch.cat([x, self.up(x_up)], 1), mask)

        return x_out


class MinimalUnet(nn.Module):
    """docstring for MinimalUnet"""

    def __init__(self, down=None, up=None, submodule=None, attention=None, withoutskip=False, **kwags):
        super(MinimalUnet, self).__init__()

        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.sub = submodule
        self.attention = attention
        self.withoutskip = withoutskip
        self.is_attention = not self.attention == None
        self.is_sub = not submodule == None

    def forward(self, x, mask=None):
        if self.is_sub:
            x_up, _ = self.sub(self.down(x), mask)
        else:
            x_up = self.down(x)

        if self.is_attention:
            x = self.attention(x, mask)

        if self.withoutskip:  # outer or inner.
            x_out = self.up(x_up)
        else:
            x_out = (torch.cat([x, self.up(x_up)], 1), mask)

        return x_out


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 is_attention_layer=False,
                 attention_model=RASC, basicblock=MinimalUnet, outermostattention=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = basicblock(down, up, submodule, withoutskip=outermost)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = basicblock(down, up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if is_attention_layer:
                if MinimalUnetV2.__qualname__ in basicblock.__qualname__:
                    attention_model = attention_model(input_nc * 2)
                else:
                    attention_model = attention_model(input_nc)
            else:
                attention_model = None

            if use_dropout:
                model = basicblock(down, up.append(nn.Dropout(0.5)), submodule, attention_model,
                                   outermostattention=outermostattention)
            else:
                model = basicblock(down, up, submodule, attention_model, outermostattention=outermostattention)

        self.model = model

    def forward(self, x, mask=None):
        # build the mask for attention use
        return self.model(x, mask)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 is_attention_layer=False, attention_model=RASC, use_inner_attention=False, basicblock=MinimalUnet):
        super(UnetGenerator, self).__init__()

        # 8 for 256x256
        # 9 for 512x512
        # construct unet structure
        self.need_mask = not input_nc == output_nc

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, basicblock=basicblock)  # 1
        for i in range(num_downs - 5):  # 3 times
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout,
                                                 is_attention_layer=use_inner_attention,
                                                 attention_model=attention_model, basicblock=basicblock)  # 8,4,2
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, is_attention_layer=is_attention_layer,
                                             attention_model=attention_model, basicblock=basicblock)  # 16
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, is_attention_layer=is_attention_layer,
                                             attention_model=attention_model, basicblock=basicblock)  # 32
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                             is_attention_layer=is_attention_layer, attention_model=attention_model,
                                             basicblock=basicblock, outermostattention=True)  # 64
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             basicblock=basicblock, norm_layer=norm_layer)  # 128

        self.model = unet_block

        self.optimizers = []

    def forward(self, input):
        if self.need_mask:
            return self.model(input, input[:, 3:4, :, :]), input
        else:
            return self.model(input[:, 0:3, :, :], input[:, 3:4, :, :]), input

    def set_optimizers(self, lr):
        self.optimizer_encoder = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizers.append(self.optimizer_encoder)

    def zero_grad_all(self):
        self.model.zero_grad()

    def step_all(self):
        self.optimizer_encoder.step()


class XBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU,
                 batch_norm=False, ins_norm=False):
        super(XBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None

        if batch_norm:
            self.bn = nn.BatchNorm2d(outc)
        elif ins_norm:
            self.bn = nn.InstanceNorm2d(outc)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False, ins_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),  #
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),  #
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),  #
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def squeezenet1_0(pretrained=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']))
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    return model


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            XBlock(3, 32, kernel_size=3, stride=2, padding=1),  # 128
            XBlock(32, 64, kernel_size=3, stride=2, padding=1),  # 64
            XBlock(64, 128, kernel_size=3, stride=2, padding=1),  # 32
            XBlock(128, 256, kernel_size=3, stride=2, padding=1),  # 16
            XBlock(256, 512, kernel_size=3, stride=2, padding=1, activation=None)  # 8x8
        )
        self.fc_f = nn.Linear(512, 192)  # bs x 3 x 64 #
        self.fc_b = nn.Linear(512, 192)
        self.optimizers = []

    def forward(self, x):
        x, mask = x[:, 0:3], x[:, 3:]
        # 256 dim features
        param_f = self.fc_f(F.adaptive_avg_pool2d(self.features(x * mask), 1).view(x.size(0), -1))
        param_b = self.fc_b(F.adaptive_avg_pool2d(self.features(x * (1 - mask)), 1).view(x.size(0), -1))

        param = param_b + param_f

        return CF(x, param.view(x.size(0), x.size(1), 1, 1, 64), 64)  # bsx64


def CF(img, param, pieces):
    # bs x 3 x 1 x 1 x 64
    color_curve_sum = torch.sum(param, 4) + 1e-30
    total_image = img * 0

    for i in range(pieces):
        total_image += torch.clamp(img - 1.0 * i / pieces, 0, 1.0 / pieces) * param[:, :, :, :, i]
    total_image *= pieces / color_curve_sum
    return total_image


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class BasicNet(nn.Module):
    def __init__(self, model, plane=64):
        super(BasicNet, self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.plane = plane
        # 512 -> 192
        self.fc_f = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        # self.fc_b = nn.Linear(512, 3*self.plane) # bs x 3 x 64 #

        self.norm = nn.Identity()
        self.optimizers = []

    def forward(self, x):
        x, m = x[:, 0:3], x[:, 3:]

        f_feature = self.squeezenet1_1(x)

        param = self.fc_f(F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1))

        return CF(x, param.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane)  # bsx64


class ForeNet(nn.Module):
    def __init__(self, model, plane=64):
        super(ForeNet, self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.plane = plane
        # 512 -> 192
        self.fc_f = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        self.fc_b = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        # self.fc = nn.Linear(2*3*self.plane,3*self.plane) # bs x 3 x 64 #

        # self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        self.norm = nn.Identity()
        self.optimizers = []

        # for param in self.squeezenet1_1.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x, m = x[:, 0:3], x[:, 3:]

        f_feature = self.squeezenet1_1(x * m)
        # b_feature = self.squeezenet1_1(x*(1-m))

        param_f = self.fc_f(F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1))
        # param_b = self.fc_b(F.adaptive_avg_pool2d(b_feature,1).view(x.size(0), -1))

        param = param_f  # + param_b
        # param = torch.cat([param_f,param_b],dim=1)
        # param = self.fc(F.relu(param))

        return CF(x, param.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane)  # bsx64


class CreateNet(nn.Module):
    def __init__(self, model, plane=64):
        super(CreateNet, self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.plane = plane
        # 512 -> 192
        self.fc_f = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        self.fc_b = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #

        # self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        self.norm = nn.Identity()
        self.optimizers = []

    def forward(self, x, fore):
        x, m = x[:, 0:3], x[:, 3:]
        fx, fm = fore[:, 0:3], fore[:, 3:]

        f = torch.cat([fx * fm, x * (1 - m)], dim=0)
        feature = self.squeezenet1_1(f)
        f_feature, b_feature = torch.split(feature, feature.size(0) // 2, dim=0)

        self.f_feature = F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1)
        self.b_feature = F.adaptive_avg_pool2d(b_feature, 1).view(x.size(0), -1)

        self.param_f = self.fc_f(self.f_feature)
        self.param_b = self.fc_b(self.b_feature)

        param = self.param_f + self.param_b

        return param  # bsx64


class CreateSimpleNet(nn.Module):
    def __init__(self, plane=64):
        super(CreateSimpleNet, self).__init__()
        self.net = SimpleNet()

        self.plane = plane
        # 512 -> 192
        self.fc_f = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        self.fc_b = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        # self.fc = nn.Linear(2*3*self.plane,3*self.plane) # bs x 3 x 64 #

        # self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        self.norm = nn.Identity()
        self.optimizers = []

        # for param in self.squeezenet1_1.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x, m = x[:, 0:3], x[:, 3:]

        f_feature = self.squeezenet1_1(x * m)
        b_feature = self.squeezenet1_1(x * (1 - m))

        param_f = self.fc_f(F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1))
        param_b = self.fc_b(F.adaptive_avg_pool2d(b_feature, 1).view(x.size(0), -1))

        param = param_f + param_b
        # param = torch.cat([param_f,param_b],dim=1)
        # param = self.fc(F.relu(param))

        return CF(x, param.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane)  # bsx64


class CreateNet3stage(nn.Module):
    def __init__(self):
        super(CreateNet3stage, self).__init__()
        self.s1 = CreateNet(squeezenet1_1(pretrained=True))
        self.s2 = CreateNet(squeezenet1_1(pretrained=True))
        self.s3 = CreateNet(squeezenet1_1(pretrained=True))

    def forward(self, x):
        m = x[:, 3:]
        x1 = self.s1(x)
        x2 = self.s2(torch.cat([x1, m], dim=1))
        x3 = self.s3(torch.cat([x2, m], dim=1))
        return x3, x2, x1


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class CreateNetADAIN(nn.Module):
    def __init__(self, model):
        super(CreateNetADAIN, self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.plane = 64
        # # 512 -> 192
        # self.fc_f = nn.Linear(512, 3*self.plane) # bs x 3 x 64 #

        # self.fc_b = nn.Linear(512, 3*self.plane) # bs x 3 x 64 #
        self.fc = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #

        # self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        self.norm = nn.Identity()
        self.optimizers = []
        self.alpha = 0.5

    def forward(self, x):
        x, m = x[:, 0:3], x[:, 3:]

        f_feature = self.squeezenet1_1(x)
        # b_feature = self.squeezenet1_1(x)

        # t = adaptive_instance_normalization(f_feature, b_feature)
        # t = self.alpha * t + (1 - self.alpha) * f_feature

        # param_f = self.fc_f(F.adaptive_avg_pool2d(f_feature,1).view(x.size(0), -1))
        # param_b = self.fc_b(F.adaptive_avg_pool2d(b_feature,1).view(x.size(0), -1))

        # param = param_f + param_b
        # param = torch.cat([param_f,param_b],dim=1)
        param = self.fc(F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1))

        return CF(x, param.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane)  # bsx64


class CreateNetFusion(nn.Module):
    def __init__(self, model, plane=128):
        super(CreateNetFusion, self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.plane = plane
        # 512 -> 192
        self.fc_f_1 = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        self.fc_f_2 = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #

        self.fc_b_1 = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        self.fc_b_2 = nn.Linear(512, 3 * self.plane)  # bs x 3 x 64 #
        # self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        # adaptive weights
        self.w1 = nn.Parameter(torch.zeros([1, 3, 1, 1]).fill_(0.5))
        self.w2 = nn.Parameter(torch.zeros([1, 3, 1, 1]).fill_(0.5))
        self.norm = nn.Identity()
        self.optimizers = []

    def forward(self, x):
        x, m = x[:, 0:3], x[:, 3:]

        f = torch.cat([x * m, x * (1 - m)], dim=0)

        feature = self.squeezenet1_1(f)

        f_feature, b_feature = torch.split(feature, feature.size(0) // 2, dim=0)

        f_feature, b_feature = F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1), F.adaptive_avg_pool2d(b_feature,
                                                                                                              1).view(
            x.size(0), -1)

        param_f_1 = self.fc_f_1(f_feature)
        param_f_2 = self.fc_f_2(f_feature)
        param_b_1 = self.fc_b_1(b_feature)
        param_b_2 = self.fc_b_2(b_feature)

        param_1 = param_b_1 + param_f_1
        param_2 = param_b_2 + param_f_2

        img_1 = CF(x, param_1.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane)
        img_2 = CF(x, param_2.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane)
        fusion_img = self.w1 * img_1 + self.w2 * img_2

        return fusion_img


class Projection(nn.Module):
    def __init__(self, ic, plane, final_relu=False):
        super(Projection, self).__init__()
        self.fc1 = nn.Linear(ic, plane)  # bs x 3 x 64 #
        self.fc2 = nn.Linear(plane, plane)  # bs x 3 x 64 #
        self.final_relu = final_relu

    def forward(self, f):

        x = self.fc2(F.relu(self.fc1(f)))

        if self.final_relu:
            return F.relu(x)
        else:
            return x


class MatrixRender(nn.Module):
    def __init__(self, plane, final_relu=False, p=Projection):
        super(MatrixRender, self).__init__()
        self.plane = plane
        self.fc_f_1 = p(512, 12, final_relu)  # bs x 3 x 64 #
        self.fc_b_1 = p(512, 12, final_relu)  # bs x 3 x 64 #

    def forward(self, x, f_feature, b_feature):
        bs, c, h, w = x.size()

        param = (self.fc_b_1(b_feature) + self.fc_f_1(f_feature)).view(bs, c, -1)

        weight, bias = param[:, :, :3], param[:, :, 3:]  # bs x 3 x 3, bs x c x 1 [bsx3x3][bsx3x1]

        # print(weight.size(),bias.size())

        return torch.bmm(weight, x.view(bs, c, -1)).view(bs, c, h, w) + bias.view(bs, c, 1, 1)  # bs c 3


class CurveRender(nn.Module):
    def __init__(self, plane=64, final_relu=False, p=Projection):
        super(CurveRender, self).__init__()
        self.plane = plane
        self.fc_f_1 = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #
        self.fc_b_1 = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #

    def forward(self, x, f_feature, b_feature, param_return=True):
        param_b = self.fc_b_1(b_feature)
        param_f = self.fc_f_1(f_feature)

        param = param_f + param_b

        if param_return:
            return CF(x, param.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane), param
        else:
            return CF(x, param.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane)


class ColorCurveRender(nn.Module):
    def __init__(self, plane=64, final_relu=False, p=Projection):
        super(ColorCurveRender, self).__init__()
        self.plane = plane
        self.fc_f_1 = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #
        self.fc_b_1 = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #
        self.fc_f_s = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #
        self.fc_b_s = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #
        self.fc = p(9 * self.plane, 3 * self.plane, final_relu)  # bs x 3 x 64 #

    def forward(self, x, f_feature, b_feature, param_return=False):
        gray = rgb_to_grayscale(x, num_output_channels=3)

        param_b = self.fc_b_1(b_feature)
        param_f = self.fc_f_1(f_feature)

        param_s = self.fc_b_s(b_feature) + self.fc_f_s(f_feature)
        param = self.fc(torch.cat([param_f, param_b, param_s], dim=1))

        return CF(x, param.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane), \
               CF(gray, param_f.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane), \
               CF(gray, param_b.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane),


class ACurveRender(nn.Module):
    def __init__(self, plane=64, final_relu=False, p=Projection):
        super(ACurveRender, self).__init__()
        self.plane = plane
        self.fc_f = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #
        self.fc_a = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #
        self.fc_b = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #

    def forward(self, x, f_feature, b_feature, a_feature):
        param = self.fc_b(b_feature) + self.fc_f(f_feature) + self.fc_a(a_feature)

        return CF(x, param.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane)


class SimpleFusion(nn.Module):
    def __init__(self, plane=64, stack=False, p=Projection, final_relu=False, cr1=CurveRender, cr2=CurveRender):
        super(SimpleFusion, self).__init__()
        self.plane = plane
        self.stack = stack

        self.cr1 = cr1(self.plane, final_relu, p=p)

    def forward(self, x, f_feature, b_feature):
        return self.cr1(x, f_feature, b_feature)


class Fusion(nn.Module):
    def __init__(self, plane=64, stack=False, p=Projection, final_relu=False, cr1=CurveRender, cr2=CurveRender):
        super(Fusion, self).__init__()
        self.plane = plane
        self.stack = stack

        self.cr1 = cr1(self.plane, final_relu, p=p)
        self.cr2 = cr2(self.plane, final_relu, p=p)

    def forward(self, x, f_feature, b_feature):

        img_1, param1 = self.cr1(x, f_feature, b_feature)

        if self.stack:
            img_2, param2 = self.cr2(img_1, f_feature, b_feature)
            fusion_img = (img_2, img_1)
        else:
            img_2 = self.cr2(x, f_feature, b_feature)
            fusion_img = img_1 + img_2

        return fusion_img


class AFusion(nn.Module):
    def __init__(self, plane=64, stack=False, p=Projection, final_relu=False, cr1=ACurveRender, cr2=ACurveRender):
        super(AFusion, self).__init__()
        self.plane = plane
        self.stack = stack

        self.cr1 = cr1(self.plane, final_relu, p=p)
        self.cr2 = cr2(self.plane, final_relu, p=p)

    def forward(self, x, f_feature, b_feature, a_feature):

        img_1 = self.cr1(x, f_feature, b_feature, a_feature)

        if self.stack:
            img_2 = self.cr2(img_1, f_feature, b_feature, a_feature)
            fusion_img = (img_2, img_1)
        else:
            img_2 = self.cr2(x, f_feature, b_feature, a_feature)
            fusion_img = img_1 + img_2

        return fusion_img


class CreateNetFusionV2(nn.Module):
    def __init__(self, model, plane=128):
        super(CreateNetFusionV2, self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.plane = plane
        self.norm = nn.Identity()
        self.optimizers = []
        self.fusion = Fusion()

    def forward(self, x):
        x, m = x[:, 0:3], x[:, 3:]

        f = torch.cat([x * m, x * (1 - m)], dim=0)
        feature = self.squeezenet1_1(f)
        f_feature, b_feature = torch.split(feature, feature.size(0) // 2, dim=0)
        f_feature = F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1)
        b_feature = F.adaptive_avg_pool2d(b_feature, 1).view(x.size(0), -1)

        fusion_img = self.fusion(x, f_feature, b_feature)

        return fusion_img


class CreateNetFusionV3(nn.Module):
    def __init__(self, model, plane=64, fusion=Fusion, final_relu=False, stack=False, cr1=CurveRender, cr2=CurveRender):
        super(CreateNetFusionV3, self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.fusion = fusion(plane=plane, final_relu=final_relu, stack=stack, cr1=cr1, cr2=cr2)

    def forward(self, x, fore):
        x, m = x[:, 0:3], x[:, 3:]
        fx, fm = fore[:, 0:3], fore[:, 3:]

        f = torch.cat([fx * fm, x * (1 - m)], dim=0)
        feature = self.squeezenet1_1(f)
        f_feature, b_feature = torch.split(feature, feature.size(0) // 2, dim=0)
        f_feature = F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1)
        b_feature = F.adaptive_avg_pool2d(b_feature, 1).view(x.size(0), -1)

        fusion_img = self.fusion(x, f_feature, b_feature)

        return fusion_img


class CreateNetFusionV4(nn.Module):
    def __init__(self, model, plane=64, fusion=AFusion, final_relu=False, stack=False, cr1=ACurveRender,
                 cr2=ACurveRender):
        super(CreateNetFusionV4, self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.fusion = fusion(plane=plane, final_relu=final_relu, stack=stack, cr1=cr1, cr2=cr2)

    def forward(self, x, fore):
        x, m = x[:, 0:3], x[:, 3:]
        fx, fm = fore[:, 0:3], fore[:, 3:]

        f = torch.cat([x, fx * fm, x * (1 - m)], dim=0)
        feature = self.squeezenet1_1(f)
        a_feature, f_feature, b_feature = torch.split(feature, feature.size(0) // 3, dim=0)

        a_feature = F.adaptive_avg_pool2d(a_feature, 1).view(x.size(0), -1)
        f_feature = F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1)
        b_feature = F.adaptive_avg_pool2d(b_feature, 1).view(x.size(0), -1)

        fusion_img = self.fusion(x, f_feature, b_feature, a_feature)

        return fusion_img


class CreateNetFusionV5(nn.Module):
    def __init__(self, model, plane=64, fusion=SimpleFusion, final_relu=False, stack=False, cr1=ColorCurveRender,
                 cr2=ColorCurveRender):
        super(CreateNetFusionV5, self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.fusion = fusion(plane=plane, final_relu=final_relu, stack=stack, cr1=cr1, cr2=cr2)

    def forward(self, x, fore):
        x, m = x[:, 0:3], x[:, 3:]
        fx, fm = fore[:, 0:3], fore[:, 3:]

        f = torch.cat([fx * fm, x * (1 - m)], dim=0)
        feature = self.squeezenet1_1(f)
        f_feature, b_feature = torch.split(feature, feature.size(0) // 2, dim=0)
        f_feature = F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1)
        b_feature = F.adaptive_avg_pool2d(b_feature, 1).view(x.size(0), -1)

        return self.fusion(x, f_feature, b_feature)


# class ADAMLP(nn.Module):

#     def __init__(self, in_dim, out_dim, hidden_list, act=nn.ReLU):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         lastv = in_dim
#         for hidden in hidden_list:
#             self.layers.append(nn.Linear(lastv, hidden))
#             lastv = hidden
#         self.layers.append(nn.Linear(lastv, out_dim))
#         # self.layers = nn.Sequential(*layers)


#     def forward(self, x, param=None):
#         if param is not None:
#             x = torch.cat([x,param],dim=-1)
#         shape = x.shape[:-1]
#         # bsxhw, c

#         for layer in self.layers:


#         x = self.layers(x.view(-1, x.shape[-1]))


#         return x.view(*shape, -1)

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, act=nn.ReLU):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(act())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, param=None):
        if param is not None:
            x = torch.cat([x, param], dim=-1)
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class ResMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, act=nn.ReLU):
        super().__init__()
        layers = []
        lastv = hidden_list[0]

        self.input = nn.Linear(in_dim, hidden_list[0])

        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(act())
            lastv = hidden

        self.resblock = nn.Sequential(*layers)
        self.last = nn.Linear(hidden_list[-1], out_dim)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.input(x.view(-1, x.shape[-1]))
        # print(x.size(),self.resblock(x).size())
        x = self.last(self.resblock(x) + x)

        return x.view(*shape, -1)


class CreateNetNeuralPointRender(nn.Module):
    def __init__(self, backbone='squeezenet', plane=64, ic=3, stage=2, resmlp=True, act=nn.ReLU, res=False,
                 use_fcb=True, use_norm=False):
        super(CreateNetNeuralPointRender, self).__init__()

        self.backbone = backbone

        # if self.backbone == 'squeezenet':
        #     model = squeezenet1_1(pretrained=False)
        #     # model.load_state_dict(
        #     #     torch.load('./pretrained_models/squeezenet1_1-b8a52dc0.pth'))
        #     self.backbone = nn.Sequential(*list(model.children())[0][:12])
        #     self.feature_dim = 512
        #
        # elif self.backbone == 'mobilenet':
        #     model = models.mobilenet_v3_small(pretrained=False)
        #     model.load_state_dict(
        #         torch.load('/apdcephfs/share_1290939/shadowcun/pretrained/mobilenet_v3_small-047dcff4.pth'))
        #     self.backbone = nn.Sequential(*list(model.features))
        #     # import pdb; pdb.set_trace()
        #     self.feature_dim = 576
        # elif self.backbone == 'eb0':
        #     model = models.efficientnet_b0(pretrained=False)
        #     model.load_state_dict(
        #         torch.load('/apdcephfs/share_1290939/shadowcun/pretrained/efficientnet_b0_rwightman-3dd342df.pth'))
        #     self.backbone = nn.Sequential(*list(model.features))
        #     # import pdb; pdb.set_trace()
        #     self.feature_dim = 1280
        # else:
        #     raise 'error'

        self.plane = plane
        # 512 -> 192
        self.fc_f = nn.Linear(self.feature_dim, plane)  # bs x 3 x 64 #
        weights_init_xavier(self.fc_f)

        self.use_fcb = use_fcb

        self.use_norm = use_norm

        self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()

        if self.use_fcb:
            self.fc_b = nn.Linear(self.feature_dim, plane)  # bs x 3 x 64 #
            weights_init_xavier(self.fc_b)

        self.norm = nn.Identity()
        self.optimizers = []
        self.res = res

        # mlp mapping
        self.mlp = ResMLP(plane + ic, ic, [plane] * stage, act) if resmlp else MLP(plane + ic, ic, [plane] * stage, act)

    def forward(self, x, f_feature, b_feature):
        # x, m = x[:, 0:3], x[:, 3:]
        # fx, fm = fore[:, 0:3], fore[:, 3:]
        bs, c, h, w = x.size()
        #
        # if self.use_norm:
        #     x = self.norm(x)
        #     fx = self.norm(fx)
        #
        # f = torch.cat([fx * fm, x * (1 - m)], dim=0)
        # feature = self.backbone(f)
        # f_feature, b_feature = torch.split(feature, feature.size(0) // 2, dim=0)
        #
        # self.f_feature = F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1)
        # self.b_feature = F.adaptive_avg_pool2d(b_feature, 1).view(x.size(0), -1)

        self.f_feature = f_feature
        self.b_feature = b_feature

        self.param_f = self.fc_f(self.f_feature)

        if self.use_fcb:
            self.param_b = self.fc_b(self.b_feature)
        else:
            self.param_b = self.fc_f(self.b_feature)

        param = self.param_f + self.param_b

        xp = x.permute(0, 2, 3, 1).reshape(bs, -1, c)
        param = param.view(bs, 1, -1).expand(-1, h * w, -1)

        if self.res:
            xx = self.mlp(xp, param) + xp
        else:
            xx = self.mlp(xp, param)

        return xx.view(bs, h, w, c).permute(0, 3, 1, 2).contiguous()  # bsx64


# class TransNeuralPointRender(nn.Module):
#     def __init__(self, model, plane=64,  ic=3,  stage=2, resmlp=True, act=nn.ReLU, res=False, use_fcb=True, use_norm=False):
#         super(TransNeuralPointRender,self).__init__()
#         self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

#         self.plane = plane
#         # 512 -> 192
#         self.fc_f = nn.Linear(512, plane) # bs x 3 x 64 #
#         weights_init_xavier(self.fc_f)

#         self.use_fcb = use_fcb

#         if self.use_fcb:
#             self.fc_b = nn.Linear(512, plane) # bs x 3 x 64 #
#             weights_init_xavier(self.fc_b)

#         self.fc_v = nn.Linear(512, plane)

#         self.use_norm = use_norm

#         self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()

#         self.optimizers = []
#         self.res = res

#         # mlp mapping
#         self.mlp = ResMLP(plane+ic,ic,[plane]*stage,act) if resmlp else MLP(plane+ic,ic,[plane]*stage,act)


#     def forward(self, x, fore):
#         x, m = x[:, 0:3], x[:, 3:]
#         fx, fm = fore[:, 0:3], fore[:, 3:]
#         bs,c,h,w = x.size()

#         if self.use_norm:
#             x = self.norm(x)
#             fx = self.norm(fx)

#         f = torch.cat([fx*fm,x*(1-m)],dim=0)

#         feature = self.squeezenet1_1(f)
#         f_feature, b_feature = torch.split(feature,feature.size(0)//2,dim=0)

#         self.f_feature = F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), 1, -1)
#         self.b_feature = F.adaptive_avg_pool2d(b_feature, 1).view(x.size(0), 1, -1)

#         self.param_f = self.fc_f(self.f_feature)

#         if self.use_fcb:
#             self.param_b = self.fc_b(self.b_feature)
#         else:
#             self.param_b = self.fc_f(self.b_feature)

#         ## tokens
#         self.param_v = self.fc_v(self.f_feature)
#         # import pdb; pdb.set_trace()
#         #(bs x 256 x 1) (bs x 1 x 256)
#         attn = torch.softmax((self.param_b.transpose(-2, -1) @ self.param_f),dim=-1)
#         # attn = attn)
#         import pdb; pdb.set_trace()
#         param = (attn @ self.param_v).transpose(1,2).reshape(bs,1,-1)

#         xp = x.permute(0,2,3,1).reshape(bs,-1,c)
#         param = param.view(bs,1,-1).expand(-1,h*w,-1)

#         if self.res:
#             xx = self.mlp(xp,param) + xp
#         else:
#             xx = self.mlp(xp,param)

#         return  xx.view(bs,h,w,c).permute(0,3,1,2).contiguous() # bsx64

