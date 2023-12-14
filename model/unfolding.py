import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion=6, stride=1):
        super(ConvBlock, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes,bias=False)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.shortcutConv2d = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False)
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        weight_map = self.conv3(out)
        x_re = self.shortcutConv2d(x)
        out = weight_map * x_re + x_re
        return out
    
class NBNetUnet_initA(nn.Module):
    def __init__(self, expansion=6):
        super(NBNetUnet_initA, self).__init__()
        self.ConvBlock1 = ConvBlock(4, 32, expansion=expansion,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.ConvBlock2 = ConvBlock(32, 64, expansion=expansion,stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.ConvBlock3 = ConvBlock(64, 128,expansion=expansion, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.ConvBlock4 = ConvBlock(128, 256, expansion=expansion,stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.ConvBlock5 = ConvBlock(256, 512, expansion=expansion,stride=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ConvBlock6 = ConvBlock(512, 256,expansion=expansion, stride=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ConvBlock7 = ConvBlock(256, 128, expansion=expansion,stride=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ConvBlock8 = ConvBlock(128, 64, expansion=expansion,stride=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.ConvBlock9 = ConvBlock(64, 32, expansion=expansion, stride=1)

        self.conv10 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, mask):
        input = torch.cat([x, mask], dim=1)
        conv1 = self.ConvBlock1(input)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        skip4 = conv4
        up6 = torch.cat([up6, skip4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        skip3 = conv3#self.skip3(conv3)
        up7 = torch.cat([up7, skip3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        skip2 = conv2#self.skip2(conv2)
        up8 = torch.cat([up8, skip2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        skip1 = conv1#self.skip1(conv1)
        up9 = torch.cat([up9, skip1], 1)
        conv9 = self.ConvBlock9(up9)
        weight_map = self.conv10(conv9)
        out = x + weight_map * x
        return weight_map, out

class ConvBlock1(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion=4, strides=1):
        super(ConvBlock1, self).__init__()
        self.strides = strides
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=strides, padding=1, groups=planes,bias=False)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.shortcutConv2d = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False)
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = self.conv3(out)
        out = out + self.shortcutConv2d(x)
        return out
    
class NBNetUnet(nn.Module):
    def __init__(self, expansion=4):
        super(NBNetUnet, self).__init__()
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.ConvBlock1 = ConvBlock1(4, 32,expansion=expansion, strides=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.ConvBlock2 = ConvBlock1(32, 64, expansion=expansion,strides=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.ConvBlock3 = ConvBlock1(64, 128,  expansion=expansion,strides=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.ConvBlock4 = ConvBlock1(128, 256,  expansion=expansion,strides=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.ConvBlock5 = ConvBlock1(256, 512, expansion=expansion, strides=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ConvBlock6 = ConvBlock1(512, 256, expansion=expansion, strides=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ConvBlock7 = ConvBlock1(256, 128,  expansion=expansion,strides=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ConvBlock8 = ConvBlock1(128, 64,  expansion=expansion,strides=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.ConvBlock9 = ConvBlock1(64, 32, expansion=expansion, strides=1)

        self.conv10 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x,mask):
        x = torch.cat([x, mask], dim=1)
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        skip4 = conv4

        up6 = torch.cat([up6, skip4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        skip3 = conv3

        up7 = torch.cat([up7, skip3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        skip2 = conv2

        up8 = torch.cat([up8, skip2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        skip1 = conv1

        up9 = torch.cat([up9, skip1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        return conv10
    
class A_net2(nn.Module):
    def __init__(self,expansion=4):
        super(A_net2, self).__init__()
        self.m_unet = NBNetUnet(expansion=expansion)
        self.lambd = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)
        self.eta = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)
    def forward(self, I,J,A,mask):
        delta_f = (J/(1+A) -I) * (- J/torch.pow((1+A),2)) #data fidelity term
        delta_g = (1-mask)*(1-mask)*A  #regular term
        A_prior = self.m_unet(A,mask)
        A = A - self.eta * ( torch.mean(delta_f, 1, keepdim=True) + self.beta * torch.mean(delta_g, 1, keepdim=True) + self.lambd * A_prior)
        return A
    

class Deshadow_netS4(nn.Module):
    def __init__(self,ex1=6,ex2=4):
        super(Deshadow_netS4, self).__init__()
        self.init_net = NBNetUnet_initA(expansion=ex1)
        self.iters_A_net = A_net2(expansion=ex2)
    def forward(self, inputs, mask):
        listA =[]
        listJ =[]
        #init step
        A0, J0 = self.init_net(inputs,mask)
        J0 = (1 + A0) * inputs
        listA.append(A0)
        listJ.append(J0)
        #iter1
        A1 = self.iters_A_net(inputs,J0,A0,mask)
        J1 = (1 + A1) * inputs
        listA.append(A1)
        listJ.append(J1)
        # iter2
        A2 = self.iters_A_net(inputs, J1, A1, mask)
        J2 = (1 + A2) * inputs
        listA.append(A2)
        listJ.append(J2)
        # iter3
        A3 = self.iters_A_net(inputs, J2, A2, mask)
        J3 = (1 + A3) * inputs
        listA.append(A3)
        listJ.append(J3)
        # iter4
        A4 = self.iters_A_net(inputs, J3, A3, mask)
        J4 = (1 + A4) * inputs
        listA.append(A4)
        listJ.append(J4)
        return listA, listJ
        # return listA, listA[-1], listJ, listJ[-1]

if __name__ == "__main__":
    model = Deshadow_netS4().cuda()
    input = torch.randn(1, 3, 256, 256).cuda()
    mask = torch.randn(1, 1, 256, 256).cuda()
    listA, listJ = model(input, mask)
    # _, weight_map, _, output = model(input,mask)
    # sum([criterion_mse(listJ[j], target) for j in range(len(listJ))])
    # sum([criterion_mse(listA[j], mask * listA[j]) for j in range(len(listA))])
    # res = listJ[-1]
    print(len(listA), len(listJ))
    print(listA[0].shape, listJ[0].shape)