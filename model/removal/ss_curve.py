from . import mae
from .unet import *
from .mae import *


class SSCurveNet(nn.Module):
    def __init__(self, model=squeezenet1_1(pretrained=False), plane=64, fusion=SimpleFusion, final_relu=False,
                 stack=False, cr1=ColorCurveRender,
                 cr2=ColorCurveRender):
        super(SSCurveNet, self).__init__()
        # self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])
        self.fusion = CreateNetNeuralPointRender()
        self.mae = getattr(mae, 'mae_vit_large_patch16')()
        # self.mae = MAE(image_size=512,
        #                image_channel=3,
        #                patch_size=16,
        #                enc_dim=512,
        #                dec_dim=256,
        #                encoder=dict(
        #                    num_layers=12,
        #                    norm=None,
        #                    nhead=8,
        #                    dim_feedforward=2048,
        #                    dropout=0,
        #                    activation='relu'
        #                ),
        #                decoder=dict(
        #                    num_layers=12,
        #                    norm=None,
        #                    # layer_kwargs=dict(
        #                    nhead=4,
        #                    dim_feedforward=1024,
        #                    dropout=0,
        #                    activation='relu'
        #                    # )
        #                ),
        #                mask_ratio=0.75)

    def fuse(self, feed, fore):
        return self.fusion(feed, fore)

    def forward(self, inp, mas, foremas):  # two image for mixing

        foreground = inp * mas
        background = inp * foremas

        loss, y, mask = self.mae(foreground.float(), mask_ratio=0.75)
        y = self.mae.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.mae.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
        mask = self.mae.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        exit()

        foreground_mae = self.mae(foreground)
        background_mae = self.mae(background)

        feed = torch.cat([foreground_mae, mas], dim=1).cuda()
        fore = torch.cat([background_mae, foremas], dim=1).cuda()

        mas_part = self.fuse(feed, fore) * mas

        foremas_part = self.fuse(feed, fore) * foremas

        res = mas_part + foremas_part

        return res
