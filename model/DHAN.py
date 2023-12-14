import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation = 1, bias = True, groups = 1, norm = 'in', nonlinear = 'relu'):
        super(ConvLayer, self).__init__()
        reflection_padding = (kernel_size + (dilation - 1)*(kernel_size - 1))//2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups = groups, bias = bias, dilation = dilation)
        self.norm = norm
        self.nonlinear = nonlinear
        
        if norm == 'bn':
          self.normalization = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
          self.normalization = nn.InstanceNorm2d(out_channels, affine = False)
        else:
          self.normalization = None
          
        if nonlinear == 'relu':
          self.activation = nn.ReLU(inplace = True)
        elif nonlinear == 'leakyrelu':
          self.activation = nn.LeakyReLU(0.2)
        elif nonlinear == 'PReLU':
          self.activation = nn.PReLU()
        else:
          self.activation = None
          
    def forward(self, x):
        
        out = self.conv2d(self.reflection_pad(x))
        if self.normalization is not None:
          out = self.normalization(out)
        
        if self.activation is not None:
          out = self.activation(out)
        
        return out
        
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels = 64, out_channels = 64, dilation = 1, stride = 1, attention = False, nonlinear = 'PReLU', norm = 'in'):
        super(ResidualBlock, self).__init__()
        
        self.Attention = attention
        
        self.conv1 = ConvLayer(in_channels, in_channels, kernel_size=3, stride=stride, dilation = dilation, norm = norm)
        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, dilation = 1, nonlinear = None, norm = norm)
          
        self.activation = nn.LeakyReLU(0.2) 
        self.downsample = None
        
        if in_channels != out_channels or stride !=1:
          self.downsample = ConvLayer(in_channels, out_channels, kernel_size=1, stride=stride, dilation = 1, nonlinear = None, norm = norm)
        else:
          self.downsample = None
          
        if nonlinear == 'relu':
          self.activation = nn.ReLU(inplace = True)
        elif nonlinear == 'leakyrelu':
          self.activation = nn.LeakyReLU(0.2)
        elif nonlinear == 'PReLU':
          self.activation = nn.PReLU()
        else:
          raise ValueError
          
        if attention:
          self.attention = Self_Attention(out_channels, k = 8, nonlinear = 'leakyrelu')
        else:
          self.attention = None
        
    def forward(self, x):
        
        residual = x
        if self.downsample is not None:
          residual = self.downsample(residual)
        out = self.conv1(x)
        out = self.conv2(out)
        if self.attention:
          out = self.attention(out)
        
        out = self.activation(torch.add(out, residual))
        
        return out

class Self_Attention(nn.Module):
    def __init__(self, channels, k, nonlinear = 'relu'):
      super(Self_Attention, self).__init__()
      self.channels = channels
      self.k = k
      self.nonlinear = nonlinear
      
      self.linear1 = nn.Linear(channels, channels//k)
      self.linear2 = nn.Linear(channels//k, channels)
      self.global_pooling = nn.AdaptiveAvgPool2d((1,1))
      
      if nonlinear == 'relu':
          self.activation = nn.ReLU(inplace = True)
      elif nonlinear == 'leakyrelu':
          self.activation = nn.LeakyReLU(0.2)
      elif nonlinear == 'PReLU':
          self.activation = nn.PReLU()
      else:
          raise ValueError
      
    def attention(self, x):
      N, C, H, W = x.size()
      out = torch.flatten(self.global_pooling(x), 1)
      out = self.activation(self.linear1(out))
      out = torch.sigmoid(self.linear2(out)).view(N, C, 1, 1)
      
      return out.mul(x)
      
    def forward(self, x):
      return self.attention(x)
      
class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers = 4, interpolation_type = 'bilinear'):
      super(SPP, self).__init__()
      self.conv = nn.ModuleList()
      self.num_layers = num_layers
      self.interpolation_type = interpolation_type
      
      for _ in range(self.num_layers):
        self.conv.append(ConvLayer(in_channels, in_channels, kernel_size=1, stride=1, dilation = 1, nonlinear = 'PReLU', norm = None))
      
      self.fusion = ConvLayer((in_channels*(self.num_layers+1)), out_channels, kernel_size = 3, stride = 1, norm = 'False', nonlinear = 'PReLU')
    
    def forward(self, x):
      
      N, C, H, W = x.size()
      out = []
      
      for level in range(self.num_layers):
        out.append(F.interpolate(self.conv[level](F.avg_pool2d(x, kernel_size = 4**(level+1), stride = 4**(level+1), padding = 4**(level+1)%2)), size = (H, W), mode = self.interpolation_type))      
      
      out.append(x)
      
      return self.fusion(torch.cat(out, dim = 1))

class Aggreation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
      super(Aggreation, self).__init__()
      self.attention = Self_Attention(in_channels, k = 8, nonlinear = 'relu')
      self.conv = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, dilation = 1, nonlinear = 'PReLU', norm = 'in')
      
    def forward(self, x):
      
      return self.conv(self.attention(x))
      

class Backbone(nn.Module):
    def __init__(self, backbones = 'vgg16'):
      super(Backbone, self).__init__()
      
      if backbones == 'vgg16':
        modules = (models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:-1])
    
        self.block1 = modules[0:4]
        self.block2 = modules[4:9]
        self.block3 = modules[9:16]
        self.block4 = modules[16:23]
        self.block5 = modules[23:]
      
        for param in self.parameters():
            param.requires_grad = False
      
      else:
        raise ValueError
        
    def forward(self, x):
        N, C, H, W = x.size()
        
        out = [x]
        
        out.append(self.block1(out[-1]))
        out.append(self.block2(out[-1]))
        out.append(self.block3(out[-1]))
        out.append(self.block4(out[-1]))
        out.append(self.block5(out[-1]))
        
        return torch.cat([(F.interpolate(item, size = (H, W), mode = 'bicubic') if sum(item.size()[2:]) != sum(x.size()[2:]) else item) for item in out], dim = 1).detach()
        

class Model(nn.Module):
  def __init__(self, channels = 64):
    super(Model, self).__init__()
    
    self.backbone = Backbone()
    
    self.fusion = ConvLayer(in_channels = 1475, out_channels = channels, kernel_size = 1, stride = 1, norm = 'in', nonlinear = 'PReLU')
    
    ##Stage0
    self.block0_1 = ConvLayer(in_channels = channels, out_channels = channels, kernel_size = 1, stride = 1, norm = 'in', nonlinear = 'PReLU')
    self.block0_2 = ConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, norm = 'in', nonlinear = 'PReLU')
    #self.block0_2 = SplAtConv2d(in_channels = channels, channels = channels, kernel_size = (3,3), stride = (1,1), padding = (1,1))
                  
    self.aggreation0_rgb = Aggreation(in_channels = channels*2, out_channels = channels)
    self.aggreation0_mas = Aggreation(in_channels = channels*2, out_channels = channels)
    
    ##Stage1
    self.block1_1 = ConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, dilation = 2, norm = 'in', nonlinear = 'PReLU')  
    self.block1_2 = ConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, dilation = 4, norm = 'in', nonlinear = 'PReLU')
    #self.block1_2 = SplAtConv2d(in_channels = channels, channels = channels, kernel_size = (3,3), stride = (1,1), padding = (4,4), dilation = (4,4))
    
    self.aggreation1_rgb = Aggreation(in_channels = channels*3, out_channels = channels)
    self.aggreation1_mas = Aggreation(in_channels = channels*3, out_channels = channels)
    
    ##Stage2
    self.block2_1 = ConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, dilation = 8, norm = 'in', nonlinear = 'PReLU')
    self.block2_2 = ConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, dilation = 16, norm = 'in', nonlinear = 'PReLU')
    #self.block2_2 = SplAtConv2d(in_channels = channels, channels = channels, kernel_size = (3,3), stride = (1,1), padding = (16,16), dilation = (16,16))
    
    self.aggreation2_rgb = Aggreation(in_channels = channels*3, out_channels = channels)
    self.aggreation2_mas = Aggreation(in_channels = channels*3, out_channels = channels)
    
    ##Stage3
    self.block3_1 = ConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, dilation = 32, norm = 'in', nonlinear = 'PReLU')
    self.block3_2 = ConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, dilation = 64, norm = 'in', nonlinear = 'PReLU')
    #self.block3_2 = SplAtConv2d(in_channels = channels, channels = channels, kernel_size = (3,3), stride = (1,1), padding = (64,64), dilation = (64,64))
    
    self.aggreation3_rgb = Aggreation(in_channels = channels*4, out_channels = channels)
    self.aggreation3_mas = Aggreation(in_channels = channels*4, out_channels = channels)  
    
    ##Stage4
    self.spp_img = SPP(in_channels = channels, out_channels = channels, num_layers = 4, interpolation_type = 'bicubic')
    self.spp_mas = SPP(in_channels = channels, out_channels = channels, num_layers = 4, interpolation_type = 'bicubic')
    
    self.block4_1 = nn.Conv2d(in_channels = channels, out_channels = 4, kernel_size = 1, stride = 1)
    self.block4_2 = nn.Conv2d(in_channels = channels, out_channels = 1, kernel_size = 1, stride = 1)
    
    self.dropout = nn.Dropout(0.2)
    
  def forward(self, x):
    
    out = self.fusion(self.backbone(x))
    
    ##Stage0
    out0_1 = self.block0_1(out)
    out0_2 = self.block0_2(out0_1)
    
    agg0_rgb = self.aggreation0_rgb(torch.cat((out0_1, out0_2), dim = 1))
    agg0_mas = self.aggreation0_mas(torch.cat((out0_1, out0_2), dim = 1))
    
    out0_2 = agg0_rgb.mul(torch.sigmoid(agg0_mas))
    
    #out0_2 = self.dropout(out0_2)
    ##Stage1
    out1_1 = self.block1_1(out0_2)
    out1_2 = self.block1_2(out1_1)
    
    agg1_rgb = self.aggreation1_rgb(torch.cat((agg0_rgb, out1_1, out1_2), dim = 1))
    agg1_mas = self.aggreation1_mas(torch.cat((agg0_mas, out1_1, out1_2), dim = 1))
    
    out1_2 = agg1_rgb.mul(torch.sigmoid(agg1_mas))
    
    #out1_2 = self.dropout(out1_2)
    
    ##Stage2
    out2_1 = self.block2_1(out1_2)
    out2_2 = self.block2_2(out2_1)
    
    agg2_rgb = self.aggreation2_rgb(torch.cat((agg1_rgb, out2_1, out2_2), dim = 1))
    agg2_mas = self.aggreation2_mas(torch.cat((agg1_mas, out2_1, out2_2), dim = 1))
    
    out2_2 = agg2_rgb.mul(torch.sigmoid(agg2_mas))
    
    #out2_2 = self.dropout(out2_2)
    ##Stage3
    out3_1 = self.block3_1(out2_2)
    out3_2 = self.block3_2(out3_1)
    
    agg3_rgb = self.aggreation3_rgb(torch.cat((agg1_rgb, agg2_rgb, out3_1, out3_2), dim = 1))
    agg3_mas = self.aggreation3_mas(torch.cat((agg1_rgb, agg2_rgb, out3_1, out3_2), dim = 1))
    
    ##Stage4
    spp_rgb = self.spp_img(agg3_rgb)
    spp_mas = self.spp_mas(agg3_mas)
    
    spp_rgb = spp_rgb.mul(torch.sigmoid(spp_mas))
    
    out_rgb = (self.block4_1(spp_rgb))
    out_mas = torch.sigmoid(self.block4_2(spp_mas))
    
    alpha = torch.sigmoid(out_rgb[:,-1,:,:].unsqueeze(1))
    
    out_rgb = x.mul(alpha).add(out_rgb[:,:-1, :, :].mul(1 - alpha)).clamp(0,1)
    #out_rgb = out_rgb[:,:-1,:,:].clamp(0,1)
    return out_rgb, out_mas


if __name__ == '__main__':
   t = torch.randn(1, 3, 256, 256).cuda()
   model = Model().cuda()
   res, _ = model(t)
   print(res.shape)