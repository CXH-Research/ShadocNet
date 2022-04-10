from .layers import *
from .unet import *


# ---------- HWMNet-LOL ----------
class HWMNet(nn.Module):
    def __init__(self, in_chn=3, wf=64, depth=4):
        super(HWMNet, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        self.bili_down = bili_resize(0.5)
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        # encoder of UNet-64
        prev_channels = 0
        for i in range(depth):  # 0,1,2,3
            downsample = True if (i + 1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels + wf, (2 ** i) * wf, downsample))
            prev_channels = (2 ** i) * wf

        # decoder of UNet-64
        self.up_path = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.bottom_conv = nn.Conv2d(prev_channels, wf, 3, 1, 1)
        self.bottom_up = bili_resize(2 ** (depth - 1))

        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2 ** i) * wf))
            self.skip_conv.append(nn.Conv2d((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))
            self.conv_up.append(nn.Sequential(*[bili_resize(2 ** i), nn.Conv2d((2 ** i) * wf, wf, 3, 1, 1)]))
            prev_channels = (2 ** i) * wf

        self.final_ff = SKFF(in_channels=wf, height=depth)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x):
        img = x
        scale_img = img

        ##### shallow conv #####
        x1 = self.conv_01(img)
        encs = []
        ######## UNet-64 ########
        # Down-path (Encoder)
        for i, down in enumerate(self.down_path):
            if i == 0:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            elif (i + 1) < self.depth:
                scale_img = self.bili_down(scale_img)
                left_bar = self.conv_01(scale_img)
                x1 = torch.cat([x1, left_bar], dim=1)
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                scale_img = self.bili_down(scale_img)
                left_bar = self.conv_01(scale_img)
                x1 = torch.cat([x1, left_bar], dim=1)
                x1 = down(x1)

        # Up-path (Decoder)
        ms_result = [self.bottom_up(self.bottom_conv(x1))]
        for i, up in enumerate(self.up_path):
            x1 = up(x1, self.skip_conv[i](encs[-i - 1]))
            ms_result.append(self.conv_up[i](x1))

        # Multi-scale selective feature fusion
        msff_result = self.final_ff(ms_result)

        # Reconstruct
        out_1 = self.last(msff_result) + img

        return out_1


class CMFNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=96, scale_unetfeats=48, kernel_size=3, reduction=4, bias=False):
        super(CMFNet, self).__init__()

        p_act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat // 2, kernel_size, bias=bias), p_act,
                                           conv(n_feat // 2, n_feat, kernel_size, bias=bias))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat // 2, kernel_size, bias=bias), p_act,
                                           conv(n_feat // 2, n_feat, kernel_size, bias=bias))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat // 2, kernel_size, bias=bias), p_act,
                                           conv(n_feat // 2, n_feat, kernel_size, bias=bias))

        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'CAB')
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'CAB')

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'PAB')
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'PAB')

        self.stage3_encoder = Encoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'SAB')
        self.stage3_decoder = Decoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'SAB')

        self.sam1o = SAM(n_feat, kernel_size=3, bias=bias)
        self.sam2o = SAM(n_feat, kernel_size=3, bias=bias)
        self.sam3o = SAM(n_feat, kernel_size=3, bias=bias)

        self.mix = Mix(1)
        self.add123 = conv(out_c, out_c, kernel_size, bias=bias)
        self.concat123 = conv(n_feat * 3, n_feat, kernel_size, bias=bias)
        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)

    def forward(self, x):
        ## Compute Shallow Features
        shallow1 = self.shallow_feat1(x)
        shallow2 = self.shallow_feat2(x)
        shallow3 = self.shallow_feat3(x)

        ## Enter the UNet-CAB
        x1 = self.stage1_encoder(shallow1)
        x1_D = self.stage1_decoder(x1)
        ## Apply SAM
        x1_out, x1_img = self.sam1o(x1_D[0], x)

        # Enter the UNet-PAB
        x2 = self.stage2_encoder(shallow2)
        x2_D = self.stage2_decoder(x2)
        # Apply SAM
        x2_out, x2_img = self.sam2o(x2_D[0], x)

        # Enter the UNet-SAB
        x3 = self.stage3_encoder(shallow3)
        x3_D = self.stage3_decoder(x3)
        # Apply SAM
        x3_out, x3_img = self.sam3o(x3_D[0], x)

        # Aggregate SAM features of Stage 1, Stage 2 and Stage 3
        mix_r = self.mix(x1_img, x2_img, x3_img)
        mixed_img = self.add123(mix_r[0])

        # Concat SAM features of Stage 1, Stage 2 and Stage 3
        concat_feat = self.concat123(torch.cat([x1_out, x2_out, x3_out], 1))
        x_final = self.tail(concat_feat)

        return [x_final + mixed_img, mixed_img, mix_r[1], x1_img, x2_img, x3_img, x_final]


class SSCurveNet(nn.Module):
    def __init__(self, model, plane=64, fusion=SimpleFusion, final_relu=False, stack=False, cr1=ColorCurveRender,
                 cr2=ColorCurveRender):
        super(SSCurveNet, self).__init__()
        sq = squeezenet1_1(pretrained=True)
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])

        self.fusion = fusion(plane=plane, final_relu=final_relu, stack=stack, cr1=cr1, cr2=cr2)

    def forward(self, x, y):  # two image for mixing
        x, m = x[:, 0:3], x[:, 3:]
        fx, fm = y[:, 0:3], y[:, 3:]

        f = torch.cat([fx * fm, x * (1 - m)], dim=0)
        feature = self.squeezenet1_1(f)
        f_feature, b_feature = torch.split(feature, feature.size(0) // 2, dim=0)
        f_feature = F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1)
        b_feature = F.adaptive_avg_pool2d(b_feature, 1).view(x.size(0), -1)

        return self.fusion(x, f_feature, b_feature)
