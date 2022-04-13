from .unet import *


class SSCurveNet(nn.Module):
    def __init__(self, model, plane=64, fusion=SimpleFusion, final_relu=False, stack=False, cr1=ColorCurveRender,
                 cr2=ColorCurveRender):
        super(SSCurveNet, self).__init__()
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
