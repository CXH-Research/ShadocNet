import torch

from . import mae
from .unet import *
from .mae import *
from .refine import *
from .maeutil import *
import torchvision.transforms as T


class SSCurveNet(nn.Module):
    def __init__(self, model=squeezenet1_1(pretrained=False), plane=64, fusion=SimpleFusion, final_relu=False,
                 stack=False, cr1=ColorCurveRender,
                 cr2=ColorCurveRender):
        super(SSCurveNet, self).__init__()
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

        self.refine = MPRNet()

    # def fuse_foward(self, inp, f_features, b_features):
    #     for i in range(len(f_features)):
    #         res = self.fusion(inp, f_features[i], b_features[i])
    #         print(res.shape)
    #         exit()
    #     return self.fusion(feed, fore)

    def fuse_foward(self, feed, fore):
        self.fusion(feed, fore)
        exit()
        return self.fusion(feed, fore)

    def refine_forward(self, res):
        return self.refine(res)

    def encode_forward(self, inp, mas, foremas):
        transform_1 = T.Resize(256)
        transform_2 = T.Resize(128)
        transform_3 = T.Resize(64)
        transform_4 = T.Resize(32)

        input_dict = {}
        mask = {}

        fg = []
        bg = []
        for bs in range(inp.shape[0]):
            inp_batch = inp[bs].unsqueeze(0)
            mas_batch = mas[bs].unsqueeze(0)
            foremas_batch = foremas[bs].unsqueeze(0)

            mas_mae = transform_4(transform_3(transform_2(transform_1(mas_batch))))
            mas_mae = mas_mae.cpu().detach().numpy()
            mas_mae = torch.LongTensor(mas_mae).flatten()[None].cuda()

            foremas_mae = transform_4(transform_3(transform_2(transform_1(foremas_batch))))
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

    def forward(self, inp, mas, foremas):  # two image for mixing

        f_features, b_features = self.encode_forward(inp, mas, foremas)

        feed = torch.cat([inp, mas], dim=1).cuda()
        fore = torch.cat([inp, foremas], dim=1).cuda()

        res = self.fuse_foward(feed, fore)


        # res = self.fuse_foward(inp, f_features, b_features)

        res = self.refine_forward(res)

        return res
