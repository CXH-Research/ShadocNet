import torch
import torch.nn as nn
from .WT import DWT, IWT
from .conv import *
from .basic_module import SAB, CAB, PAB, SAM, conv_down


# ---------- Basic Layers ----------
def bili_resize(factor):
    return nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)


# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# ---------- Basic Blocks ----------
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.body = [HWB(n_feat=in_size, o_feat=in_size, kernel_size=3, reduction=16, bias=False,
                         act=nn.PReLU())]  # for _ in range(wab)]
        self.body = nn.Sequential(*self.body)

        if downsample:
            self.downsample = PS_down(out_size, out_size, downscale=2)

        self.tail = nn.Conv2d(in_size, out_size, kernel_size=1)

    def forward(self, x):
        out = self.body(x)
        out = self.tail(out)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.up = PS_up(in_size, out_size, upscale=2)
        self.conv_block = UNetConvBlock(in_size, out_size, downsample=False)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], dim=1)
        out = self.conv_block(out)
        return out


# ---------- Resizing Modules (Pixel(Un)Shuffle) ----------
class PS_down(nn.Module):
    def __init__(self, in_size, out_size, downscale):
        super(PS_down, self).__init__()
        self.UnPS = nn.PixelUnshuffle(downscale)
        self.conv1 = nn.Conv2d((downscale ** 2) * in_size, out_size, 1, 1, 0)

    def forward(self, x):
        x = self.UnPS(x)  # h/2, w/2, 4*c
        x = self.conv1(x)
        return x


class PS_up(nn.Module):
    def __init__(self, in_size, out_size, upscale):
        super(PS_up, self).__init__()

        self.PS = nn.PixelShuffle(upscale)
        self.conv1 = nn.Conv2d(in_size // (upscale ** 2), out_size, 1, 1, 0)

    def forward(self, x):
        x = self.PS(x)  # h/2, w/2, 4*c
        x = self.conv1(x)
        return x


# ---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size, n_feats, H, W = inp_feats[1].shape

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)

        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


# Half Wavelet Dual Attention Block (HWB)
class HWB(nn.Module):
    def __init__(self, n_feat, o_feat, kernel_size, reduction, bias, act):
        super(HWB, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        modules_body = \
            [
                conv(n_feat * 2, n_feat, kernel_size, bias=bias),
                act,
                conv(n_feat, n_feat * 2, kernel_size, bias=bias)
            ]
        self.body = nn.Sequential(*modules_body)

        self.WSA = SALayer()
        self.WCA = CALayer(n_feat * 2, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1, bias=bias)
        self.conv3x3 = nn.Conv2d(n_feat, o_feat, kernel_size=3, padding=1, bias=bias)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        residual = x

        # Split 2 part
        wavelet_path_in, identity_path = torch.chunk(x, 2, dim=1)

        # Wavelet domain (Dual attention)
        x_dwt = self.dwt(wavelet_path_in)
        res = self.body(x_dwt)
        branch_sa = self.WSA(res)
        branch_ca = self.WCA(res)
        res = torch.cat([branch_sa, branch_ca], dim=1)
        res = self.conv1x1(res) + x_dwt
        wavelet_path = self.iwt(res)

        out = torch.cat([wavelet_path, identity_path], dim=1)
        out = self.activate(self.conv3x3(out))
        out += self.conv1x1_final(residual)

        return out


# U-Net
bn = 2


class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, block):
        super(Encoder, self).__init__()
        if block == 'CAB':
            self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                                   range(bn)]
            self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _
                                   in range(bn)]
        elif block == 'PAB':
            self.encoder_level1 = [PAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level2 = [PAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                                   range(bn)]
            self.encoder_level3 = [PAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _
                                   in range(bn)]
        elif block == 'SAB':
            self.encoder_level1 = [SAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level2 = [SAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                                   range(bn)]
            self.encoder_level3 = [SAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _
                                   in range(bn)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, x):
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, block):
        super(Decoder, self).__init__()
        if block == 'CAB':
            self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                                   range(bn)]
            self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _
                                   in range(bn)]
        elif block == 'PAB':
            self.decoder_level1 = [PAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level2 = [PAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                                   range(bn)]
            self.decoder_level3 = [PAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _
                                   in range(bn)]
        elif block == 'SAB':
            self.decoder_level1 = [SAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level2 = [SAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                                   range(bn)]
            self.decoder_level3 = [SAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _
                                   in range(bn)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)
        if block == 'CAB':
            self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
            self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        if block == 'PAB':
            self.skip_attn1 = PAB(n_feat, kernel_size, reduction, bias=bias, act=act)
            self.skip_attn2 = PAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        if block == 'SAB':
            self.skip_attn1 = SAB(n_feat, kernel_size, reduction, bias=bias, act=act)
            self.skip_attn2 = SAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return [dec1, dec2, dec3]


# ---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


# Mixed Residual Module
class Mix(nn.Module):
    def __init__(self, m=1):
        super(Mix, self).__init__()
        w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2, feat3):
        factor = self.mix_block(self.w)
        other = (1 - factor) / 2
        output = fea1 * other.expand_as(fea1) + fea2 * factor.expand_as(fea2) + feat3 * other.expand_as(feat3)
        return output, factor
