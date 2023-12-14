import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision


class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.vgg16_pretrained = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES)
        self.vgg16_pretrained.requires_grad_ = False
        '''
        Change Maxpool Stride to 1
        '''
        self.vgg16_pretrained.features[30] = nn.MaxPool2d(kernel_size=(1,1), stride=(1,1), padding=0, dilation=1, ceil_mode=False)
        self.vgg16_pretrained.features[23] = nn.MaxPool2d(kernel_size=(1,1), stride=(1,1), padding=0, dilation=1, ceil_mode=False)
        self.vgg16_pretrained.features.add_module('31', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1,1), stride=(1,1), padding=(0,0)))
        self.vgg16_pretrained.features.add_module('32', nn.ReLU(inplace=True))
        # self.vgg16_pretrained.features = nn.Sequential(*list(self.vgg16_pretrained.features.children())[0:24])
        '''
        Remove avgpool and classifier
        '''
        self.vgg16_pretrained = nn.Sequential(*list(self.vgg16_pretrained.children())[:-2])
        

    def forward(self, x):
        x = self.vgg16_pretrained(x)
        return x.clip(min=0, max=1)


class A_Net(nn.Module):
    def __init__(self):
        super(A_Net, self).__init__()
        self.prelu = nn.PReLU(num_parameters=64)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.conv1 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(4,4), stride=(2,2), padding=(1,1))

    
    def forward(self, x):
        x = self.dropout(self.prelu(self.conv1(x)))
        x = self.dropout(self.prelu(self.conv2(x)))
        x = self.dropout(self.prelu(self.conv3(x)))
        x = self.dropout(self.prelu(self.conv4(x)))
        x = self.deconv1(x)
        return x.clip(min=0, max=1)


class DeShadowNet(nn.Module):
    def __init__(self):
        super(DeShadowNet, self).__init__()
        self.gnet = G_Net()
        self.snet = A_Net()
        self.anet = A_Net()
        self.dropout = nn.Dropout(p=0.5)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(8,8), stride=(4,4), padding=(2,2))
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(8,8), stride=(4,4), padding=(2,2))
        self.conv21 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(9,9), stride=(1,1), padding=(4,4))
        self.conv31 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(9,9), stride=(1,1), padding=(4,4))
        self.maxpool21 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2),padding=1, dilation=1, ceil_mode=False)
        self.maxpool22 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2),padding=1, dilation=1, ceil_mode=False)
        self.conv22 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.final_conv = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(1,1), stride=(1,1), padding=(0,0))


    def forward(self, x):
        c11 = self.maxpool21(self.dropout(F.relu(self.conv21(x), inplace=True)))
        c21 = self.maxpool22(self.dropout(F.relu(self.conv31(x), inplace=True)))
        c = self.gnet(x)
        d11 = self.deconv1(c)
        d21 = self.deconv2(c)
        c12 = self.dropout(F.relu(self.conv22(d11), inplace=True))
        c22 = self.dropout(F.relu(self.conv32(d21), inplace=True))
        anet_input = torch.concat([c11, c12], axis=1)
        snet_input = torch.concat([c21, c22], axis=1)
        anet_out = self.anet(anet_input)
        snet_out = self.snet(snet_input)
        merge = torch.concat([anet_out, snet_out], axis=1)
        x = self.final_conv(merge)
        return anet_out.clip(min=0, max=1)
    

if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = DeShadowNet().cuda()
    res = model(t)
    print(res.shape)

    

'''
convert the following tensorflow code to pytorch:

def conv_layer(x, filtershape, stride, name):
    with tf.variable_scope(name):
        filters = tf.get_variable(
            name = 'weight',
            shape = filtershape,
            dtype = tf.float32,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.001),
            trainable = True)
        conv = tf.nn.conv2d(x, filters, [1, stride, stride, 1], padding= 'SAME')
        conv_biases = tf.Variable(tf.constant(0.0, shape = [filtershape[3]], dtype = tf.float32),
                                trainable=True, name ='bias')
        bias = tf.nn.bias_add(conv, conv_biases)
        output = prelu(bias)
        #output = tf.nn.dropout(prelu, keep_prob=keep_prob)
        img_filt = tf.reshape(filters[:,:,:,1], [-1,filtershape[0],filtershape[1],1])
        tf.summary.image('conv_filter',img_filt)
        return output

def deconv_layer(x, filtershape,output_shape, stride, name):
    with tf.variable_scope(name):
        filters = tf.get_variable(
            name = 'weight',
            shape = filtershape,
            dtype = tf.float32,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.001),
            trainable = True)
        deconv = tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1], padding ='SAME')
        #deconv_biases = tf.Variable(tf.constant(0.0, shape = [filtershape[3]], dtype = tf.float32),
        #                        trainable=True, name ='bias')
        #bias = tf.nn.bias_add(deconv, deconv_biases)
        #output = prelu(bias)
        #output = tf.nn.dropout(prelu, keep_prob=keep_prob)
        img_filt = tf.reshape(filters[:,:,:,1], [-1,filtershape[0],filtershape[1],1])
        tf.summary.image('deconv_filter',img_filt)
        return prelu(deconv)

def max_pool_layer(x,filtershape,stride,name):
    return tf.nn.max_pool(x, filtershape, [1, stride, stride, 1], padding ='SAME',name = name)

def A_Net(self,x,G_input,keep_prob): # after conv3 in G_Net  256

        print('making a-network')
        sess=tf.Session()
        with tf.variable_scope('A_Net'):

            # conv2-1
            conv2_1 = conv_layer(x,[9,9,3,96],1,'conv2-1')

            # pool5
            conv2_1_output = max_pool_layer(conv2_1,[1,3,3,1],2,'pool2-1')
            print('conv2-1')
            print(sess.run(tf.shape(conv2_1_output)))

            # conv2-2
            conv2_2_output = conv_layer(G_input,[1,1,256,64],1,'conv2-2') 
            print('conv2-2')
            print(sess.run(tf.shape(conv2_2_output)))

            # concat conv2-1 and conv2-2
            conv_concat = tf.concat(axis=3, values = [conv2_1_output,conv2_2_output], name = 'concat_a_net')

            # conv2-3
            conv2_3 = conv_layer(conv_concat,[5,5,160,64],1,'conv2-3')
            print('conv2-3')
            print(sess.run(tf.shape(conv2_3)))

            # conv2-4
            conv2_4 = conv_layer(conv2_3,[5,5,64,64],1,'conv2-4')
            print('conv2-4')
            print(sess.run(tf.shape(conv2_4)))

            # conv2-5
            conv2_5 = conv_layer(conv2_4,[5,5,64,64],1,'conv2-5')
            print('conv2-5')
            print(sess.run(tf.shape(conv2_5)))

            # conv2-6
            conv2_6 = conv_layer(conv2_5,[5,5,64,64],1,'conv2-6')
            print('conv2-6')
            print(sess.run(tf.shape(conv2_6)))

            # deconv2_1
            deconv2_2 = deconv_layer(conv2_6,[4,4,3,64],[self.batch_size,224,224,3],2,'deconv2-2')
            print('deconv2-2')
            print(sess.run(tf.shape(deconv2_2)))
 
            print('finishing a-network')
            
            tf.summary.image('conv2_1',conv2_1_output[:,:,:,0:3])
            tf.summary.image('conv2_2',conv2_2_output[:,:,:,0:3])
            tf.summary.image('conv2_3',conv2_3[:,:,:,0:3])
            tf.summary.image('conv2_4',conv2_4[:,:,:,0:3])
            tf.summary.image('conv2_5',conv2_5[:,:,:,0:3])
            tf.summary.image('conv2_6',conv2_6[:,:,:,0:3])
            red = tf.reshape(deconv2_2[:,:,:,0], [-1,224,224,1])
            green = tf.reshape(deconv2_2[:,:,:,1], [-1,224,224,1])
            blue = tf.reshape(deconv2_2[:,:,:,2], [-1,224,224,1])
            tf.summary.image('deconv3_1',red)
            tf.summary.image('deconv3_1',green)
            tf.summary.image('deconv3_1',blue)
            #tf.summary.image('deconv2_2-1',deconv2_2[:,:,:,0:1])
            #tf.summary.image('deconv2_2-2',deconv2_2[:,:,:,1:2])
            #tf.summary.image('deconv2_2',deconv2_2[:,:,:,:])
            sess.close()
            return deconv2_2

'''