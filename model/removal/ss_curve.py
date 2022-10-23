import torch

from .mae import *
from .maeutil import *
from .refine import *
from .unet import *
import losses


class SSCurveNet(nn.Module):
    def __init__(self, model=squeezenet1_1(pretrained=False), plane=64, fusion=SimpleFusion, final_relu=False,
                 stack=False, cr1=ColorCurveRender,
                 cr2=ColorCurveRender):
        super(SSCurveNet, self).__init__()
        self.criterion_l1_loss = losses.l1_relative
        self.criterion_perc = losses.Perceptual()
        self.criterion_mse = nn.MSELoss()
        self.criterion_mask = losses.MaskLoss()
        # self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])
        self.fusion = CreateNetNeuralPointRender()
        domain_conf = {
            'rgb': {
                'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1),
                'output_adapter': partial(SpatialOutputAdapter, num_channels=3, stride_level=1),
            }
        }
        domains = ['rgb']

        input_adapters = {
            domain: dinfo['input_adapter'](
                patch_size_full=16,
            )
            for domain, dinfo in domain_conf.items()
        }
        output_adapters = {
            domain: dinfo['output_adapter'](
                patch_size_full=16,
                dim_tokens=256,
                use_task_queries=True,
                depth=2,
                context_tasks=domains,
                task=domain
            )
            for domain, dinfo in domain_conf.items()
        }

        self.multimae = pretrain_multimae_base(
            input_adapters=input_adapters,
            output_adapters=output_adapters,
        )

        # ckpt_url = 'https://github.com/EPFL-VILAB/MultiMAE/releases/download/pretrained-weights/multimae-b_98_rgb' \
        #            '+-depth-semseg_1600e_multivit-afff3f8c.pth '
        # ckpt = torch.hub.load_state_dict_from_url(ckpt_url, map_location='cpu')
        # self.multimae.load_state_dict(ckpt['model'], strict=False)

        self.refine = ShadowRemoval()
        weights_init_normal(self.refine)

    def fuse_foward(self, inp, f_feature, b_feature):
        # return self.fusion(inp, fore)
        return self.fusion(inp, f_feature, b_feature)

    def refine_forward(self, res):
        return self.refine(res)

    def encode_forward(self, inp, mas, foremas):
        input_dict = {}
        mask = {}

        fg = []
        bg = []
        for bs in range(inp.shape[0]):
            inp_batch = inp[bs].unsqueeze(0)
            mas_batch = mas[bs].unsqueeze(0)
            foremas_batch = foremas[bs].unsqueeze(0)

            mas_mae = F.interpolate(mas_batch, (16, 16))
            mas_mae = mas_mae.cpu().detach().numpy()
            mas_mae = torch.LongTensor(mas_mae).flatten()[None].cuda()

            foremas_mae = F.interpolate(foremas_batch, (16, 16))
            foremas_mae = foremas_mae.cpu().detach().numpy()
            foremas_mae = torch.LongTensor(foremas_mae).flatten()[None].cuda()

            input_dict['rgb'] = inp_batch
            mask['rgb'] = mas_mae
            fg_encode = self.multimae.forward(
                input_dict,
                task_masks=mask
            )

            mask['rgb'] = foremas_mae
            bg_encode = self.multimae.forward(
                input_dict,
                task_masks=mask
            )

            fg.append(fg_encode)
            bg.append(bg_encode)

        return fg, bg

    def forward(self, inp, gt_mas, mas, foremas, tar):  # two image for mixing

        f_features, b_features = self.encode_forward(inp, mas, foremas)

        # 1, 1019, 768
        # print(f_features[0].shape)
        # 1, 773, 768
        # print(b_features[0].shape)
        # 1, 1024, 768
        # print(f_features[1].shape)
        # 1, 792, 768
        # print(b_features[1].shape)

        f_f, b_f = [], []
        for x, y in zip(f_features, b_features):
            if x.shape[1] == 0:
                x = torch.zeros(1, 1024, 768).cuda()
            temp_x = F.adaptive_avg_pool2d(x, (1, 512))
            if y.shape[1] == 0:
                y = torch.zeros(1, 1024, 768).cuda()
            temp_y = F.adaptive_avg_pool2d(y, (1, 512))
            f_f.append(temp_x.squeeze(0))
            b_f.append(temp_y.squeeze(0))
        f_f = torch.cat(f_f, 0)
        b_f = torch.cat(b_f, 0)

        # inp_fore = torch.cat((inp, foremas), dim=1)
        # inp_back = torch.cat((inp, mas), dim=1)
        #
        # res = self.fuse_foward(inp_back, inp_fore)

        res = self.fuse_foward(inp, f_f, b_f)

        # loss_rl1_1 = self.criterion_l1_loss(res, tar, mas)
        # loss_rl1_2 = self.criterion_l1_loss(res, tar, foremas)
        loss_mse = self.criterion_mse(res, tar)
        loss_perc = self.criterion_perc(res, tar)

        # loss = loss_rl1_1 + loss_rl1_2 + 0.04 * loss_perc
        loss = loss_mse + 0.04 * loss_perc

        res = self.refine_forward(res)
        # finalrgb, side0_rgb, side1_rgb, side2_rgb, side3_rgb, finalmask, side0_mask, side1_mask, side2_mask, side3_mask = self.refine_forward(res)
        #
        # loss_l1_1 = self.criterion_l1(finalrgb, tar)
        # loss_l1_2 = self.criterion_l1(side0_rgb, tar)
        # loss_l1_3 = self.criterion_l1(side1_rgb, tar)
        # loss_l1_4 = self.criterion_l1(side2_rgb, tar)
        # loss_l1_5 = self.criterion_l1(side3_rgb, tar)
        #
        # loss_l1 = loss_l1_1 + loss_l1_2 + loss_l1_3 + loss_l1_4 + loss_l1_5
        #
        # loss_l1_mask_1 = self.criterion_mask(torch.sigmoid(finalmask), gt_mas)
        # loss_l1_mask_2 = self.criterion_mask(torch.sigmoid(side0_mask), gt_mas)
        # loss_l1_mask_3 = self.criterion_mask(torch.sigmoid(side1_mask), gt_mas)
        # loss_l1_mask_4 = self.criterion_mask(torch.sigmoid(side2_mask), gt_mas)
        # loss_l1_mask_5 = self.criterion_mask(torch.sigmoid(side3_mask), gt_mas)
        #
        # loss_mask = loss_l1_mask_1 + loss_l1_mask_2 + loss_l1_mask_3 + loss_l1_mask_4 + loss_l1_mask_5
        # loss_rl1_1 = self.criterion_l1_loss(res, tar, mas)
        # loss_rl1_2 = self.criterion_l1_loss(res, tar, foremas)
        loss_mse = self.criterion_mse(res, tar)
        loss_perc = self.criterion_perc(res, tar)

        # loss += loss_l1 + loss_mask + 0.04 * loss_perc
        # loss += loss_rl1_1 + loss_rl1_2 + 0.04 * loss_perc
        loss = loss_mse + 0.04 * loss_perc
        return res, loss
