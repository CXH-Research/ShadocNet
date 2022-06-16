from .unet import *


class SSCurveNet(nn.Module):
    def __init__(self, model=squeezenet1_1(pretrained=False), plane=64, fusion=SimpleFusion, final_relu=False,
                 stack=False, cr1=ColorCurveRender,
                 cr2=ColorCurveRender):
        super(SSCurveNet, self).__init__()
        # self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])
        self.fusion = CreateNetNeuralPointRender()

    def fuse(self, feed, fore):
        return self.fusion(feed, fore)

    def forward(self, feed, fore, mas, foremas):  # two image for mixing

        mas_part = self.fuse(feed, fore) * mas

        foremas_part = self.fuse(feed, fore) * foremas

        res = mas_part + foremas_part

        return res
