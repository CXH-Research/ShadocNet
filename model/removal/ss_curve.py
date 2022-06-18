from . import mae
from .unet import *
from .mae import *
from .maeutil import *
import numpy as np

class SSCurveNet(nn.Module):
    def __init__(self, model=squeezenet1_1(pretrained=False), plane=64, fusion=SimpleFusion, final_relu=False,
                 stack=False, cr1=ColorCurveRender,
                 cr2=ColorCurveRender):
        super(SSCurveNet, self).__init__()
        # self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])
        self.fusion = CreateNetNeuralPointRender()
        DOMAIN_CONF = {
            'rgb': {
                'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1),
                'output_adapter': partial(SpatialOutputAdapter, num_channels=3, stride_level=1),
            }
        }
        DOMAINS = ['rgb']

        input_adapters = {
            domain: dinfo['input_adapter'](
                patch_size_full=16,
            )
            for domain, dinfo in DOMAIN_CONF.items()
        }
        output_adapters = {
            domain: dinfo['output_adapter'](
                patch_size_full=16,
                dim_tokens=256,
                use_task_queries=True,
                depth=2,
                context_tasks=DOMAINS,
                task=domain
            )
            for domain, dinfo in DOMAIN_CONF.items()
        }

        self.multimae = pretrain_multimae_base(
            input_adapters=input_adapters,
            output_adapters=output_adapters,
        )
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

        input_dict = {}
        input_dict['rgb'] = inp

        mask = {}

        mask['rgb'] = np.array([
            [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1]
        ])

        task_masks = {k: torch.LongTensor(v).flatten()[None].cuda() for k, v in mask.items()}
        preds, masks = self.multimae.forward(
            input_dict,
            mask_inputs=True,
            task_masks=task_masks
        )

        preds = {domain: pred.detach().cpu() for domain, pred in preds.items()}
        masks = {domain: mask.detach().cpu() for domain, mask in masks.items()}

        exit()

        feed = torch.cat([foreground_mae, mas], dim=1).cuda()
        fore = torch.cat([background_mae, foremas], dim=1).cuda()

        mas_part = self.fuse(feed, fore) * mas

        foremas_part = self.fuse(feed, fore) * foremas

        res = mas_part + foremas_part

        return res
