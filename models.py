import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.utils.model_zoo as model_zoo

import torch.nn.functional as F
import math
import random
import torchvision.models as models



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

# class MLP(nn.Module):

#     def __init__(self, in_dim, out_dim, hidden_list, act=nn.ReLU):
#         super().__init__()
#         layers = []
#         lastv = in_dim
#         for hidden in hidden_list:
#             layers.append(nn.Linear(lastv, hidden))
#             layers.append(act())
#             lastv = hidden
#         layers.append(nn.Linear(lastv, out_dim))
#         self.layers = nn.Sequential(*layers)

#     def forward(self, x, param=None):
#         if param is not None: 
#             x = torch.cat([x,param],dim=-1)
#         shape = x.shape[:-1]
#         x = self.layers(x.view(-1, x.shape[-1]))
#         return x.view(*shape, -1)


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list,  act=nn.ReLU):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, param=None):
        if param is not None:
            x = torch.cat([x,param],dim=-1)
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

def CFP(img, param, pieces):
    color_curve_sum = torch.sum(param, -1) + 1e-30
    total_image = img * 0
    # param = [bs,c,h,w,i]
    for i in range(pieces):
        total_image += torch.clamp(img - 1.0 * i / pieces, 0, 1.0 / pieces) * param[:, :, :, i]
    total_image *= pieces / color_curve_sum   
    return total_image

def CF(img, param, pieces):
    color_curve_sum = torch.sum(param, 4) + 1e-30
    total_image = img * 0
    # param = [bs,c,h,w,i]
    for i in range(pieces):
        total_image += torch.clamp(img - 1.0 * i / pieces, 0, 1.0 / pieces) * param[:, :, :, :, i]
    total_image *= pieces / color_curve_sum

    return total_image


class Projection(nn.Module):
    def __init__(self, ic, plane, final_relu=False):
        super(Projection, self).__init__()
        self.fc1 = nn.Linear(ic, plane)  # bs x 3 x 64 #
        self.final_relu = final_relu

    def forward(self, f):

        x = self.fc1(f)
        if self.final_relu:
            return F.relu(x)
        else:
            return x


class SCRM(nn.Module):
    def __init__(self, plane=128, final_relu=False, p=Projection):
        super(SCRM, self).__init__()
        self.plane = plane
        self.fc_f_1 = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #
        self.fc_b_1 = p(512, 3 * self.plane, final_relu)  # bs x 3 x 64 #

        self.fc_label_1 = nn.Linear(5, 32)
        self.fc_label_2 = nn.Linear(32, 64)
        self.fc_f_label = nn.Linear(512 + 64, 3 * self.plane, bias=True)


    def forward(self, x, f_feature, b_feature, label=None):

        if label is not None:
            label_feature = self.fc_label_2(F.relu(self.fc_label_1(label)))

            f_feature = torch.cat((f_feature, label_feature), dim=1)
            param_f = self.fc_f_label(f_feature)
        else:
            param_f = self.fc_f_1(f_feature)
        param = self.fc_b_1(b_feature) + param_f

        return CF(x, param.view(x.size(0), x.size(1), 1, 1, self.plane), self.plane), param



class Fusion(nn.Module):
    def __init__(self, plane=64, stack=False, p=Projection, final_relu=False, cr1=SCRM,
                 cr2=SCRM):
        super(Fusion, self).__init__()
        self.plane = plane
        self.stack = stack
        print('stack:', self.stack)
        self.cr1 = cr1(self.plane, final_relu, p=p)
        self.cr2 = cr2(self.plane, final_relu, p=p)

    def forward(self, ori_img, x, f_feature, b_feature, label, withLabel):
      
        img_1, param1 = self.cr1(ori_img, f_feature, b_feature, label)
        if self.stack:
            img_2, param2 = self.cr2(img_1, f_feature, b_feature, label)
            fusion_img = (img_2, img_1)

        else:
            _, param2 = img_1, param1
            fusion_img = (img_1, img_1)
       
        return fusion_img, param1, param2

class S2CRNet(nn.Module):
    def __init__(self, model, plane=64, crm=Fusion, final_relu=False, stack=False, cr1=SCRM, cr2=SCRM):
        super(S2CRNet, self).__init__()

        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])
        self.fusion = crm(plane=plane, final_relu=final_relu, stack=stack, cr1=SCRM,
                             cr2=SCRM)
        self.stack = stack

    def forward(self, ori_img, x, fore, label, withLabel):
        x, m = x[:, 0:3], x[:, 3:]
        fx, fm = fore[:, 0:3], fore[:, 3:]


        #### backbone stage
        f = torch.cat([fx * fm, x * (1 - m)], dim=0)
        feature = self.squeezenet1_1(f)
        f_feature, b_feature = torch.split(feature, feature.size(0) // 2, dim=0)

        f_feature = F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1)
        b_feature = F.adaptive_avg_pool2d(b_feature, 1).view(x.size(0), -1)


        ##### curve rendering
        fusion_img, param1, param2 = self.fusion(ori_img, x, f_feature, b_feature, label, withLabel)

        return fusion_img, param1, param2


class CreateNetNeuralPointRender(nn.Module):
    def __init__(self, backbone='squeezenet', plane=64,  ic=3,  stage=2, resmlp=True, act=nn.ReLU, res=False, use_fcb=True, use_norm=False):
        super(CreateNetNeuralPointRender,self).__init__()

        self.backbone = backbone

        if self.backbone == 'squeezenet':
            model = models.squeezenet1_1(pretrained=False)
            # model.load_state_dict(torch.load('/apdcephfs/share_1290939/shadowcun/pretrained/squeezenet1_1-b8a52dc0.pth'))
            self.backbone = nn.Sequential(*list(model.children())[0][:12])
            self.feature_dim = 512

        elif self.backbone == 'mobilenet':
            model = models.mobilenet_v3_small(pretrained=False)
            # model.load_state_dict(torch.load('/apdcephfs/share_1290939/shadowcun/pretrained/mobilenet_v3_small-047dcff4.pth'))
            self.backbone = nn.Sequential(*list(model.features))
            # import pdb; pdb.set_trace()
            self.feature_dim = 576
        elif self.backbone == 'eb0':
            model = models.efficientnet_b0(pretrained=False)
            # model.load_state_dict(torch.load('/apdcephfs/share_1290939/shadowcun/pretrained/efficientnet_b0_rwightman-3dd342df.pth'))
            self.backbone = nn.Sequential(*list(model.features))
            # import pdb; pdb.set_trace()
            self.feature_dim = 1280
        else:
            raise('error')

        self.plane = plane
        # 512 -> 192
        self.fc_f = nn.Linear(self.feature_dim, plane) # bs x 3 x 64 # 
        # weights_init_xavier(self.fc_f)

        self.use_fcb = use_fcb


        self.use_norm = use_norm
        
        self.norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True)


        if self.use_fcb:
            self.fc_b = nn.Linear(self.feature_dim, plane) # bs x 3 x 64 # 
            # weights_init_xavier(self.fc_b)

        self.norm = nn.Identity()
        self.optimizers = []
        self.res = res

        # mlp mapping
        self.mlp = ResMLP(plane+ic,ic,[plane]*stage,act) if resmlp else MLP(plane+ic,ic,[plane]*stage,act)


    def forward(self, img, x, fore):
        x, m = x[:, 0:3], x[:, 3:]
        fx, fm = fore[:, 0:3], fore[:, 3:]
        bs,c,h,w = x.size()

        if self.use_norm:
            x = self.norm(x)
            fx = self.norm(fx)

        f = torch.cat([fx*fm,x*(1-m)],dim=0)
        feature = self.backbone(f)
        f_feature, b_feature = torch.split(feature,feature.size(0)//2,dim=0)
        
        self.f_feature = F.adaptive_avg_pool2d(f_feature, 1).view(x.size(0), -1)
        self.b_feature = F.adaptive_avg_pool2d(b_feature, 1).view(x.size(0), -1)

        self.param_f = self.fc_f(self.f_feature)
        
        if self.use_fcb:
            self.param_b = self.fc_b(self.b_feature)
        else:
            self.param_b = self.fc_f(self.b_feature)

        param = self.param_f + self.param_b

        xp = img.permute(0,2,3,1).reshape(bs,-1,c)
        param = param.view(bs,1,-1).expand(-1,img.size(2)*img.size(3),-1)

        if self.res:
            xx = self.mlp(xp,param) + xp
        else:
            xx = self.mlp(xp,param)

        return  xx.view(bs,img.size(2),img.size(3),c).permute(0,3,1,2).contiguous() # bsx64