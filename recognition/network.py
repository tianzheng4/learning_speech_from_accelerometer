import torch
import torch.nn as nn
from torch.nn import functional as F

# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride) #, padding)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# Residual Block
#   adapted from pytorch tutorial
#   https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-
#   intermediate/deep_residual_network/main.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

# Image Transform Network
class ImageTransformNet(nn.Module):
    def __init__(self, img_dim=128):
        super(ImageTransformNet, self).__init__()
        self.img_dim = img_dim
        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # # encoding layers
        # self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=1)
        # self.in1_e = nn.InstanceNorm2d(32, affine=True)

        # self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        # self.in2_e = nn.InstanceNorm2d(64, affine=True)

        # self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        # self.in3_e = nn.InstanceNorm2d(128, affine=True)

        # encoding layers
        self.conv1 = ConvLayer(3*3, 3*32, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(3*32, affine=True)

        self.conv2 = ConvLayer(3*32, 3*64, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(3*64, affine=True)

        self.conv3 = ConvLayer(3*64, 3*128, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(3*128, affine=True)

        # residual layers
        self.num_reslayers = 9
        self.res = nn.ModuleList()
        for i in range(self.num_reslayers):
            self.res.append(ResidualBlock(3*128))

        # decoding layers
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv1 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        for i in range(self.num_reslayers):
            y = self.res[i](y)


        y = y.view(-1, 128, 3*(self.img_dim//4), (self.img_dim//4))

        # decode
        y = self.in3_d(self.deconv3(y))
        y = self.in2_d(self.deconv2(y))
        y = self.tanh(self.in1_d(self.deconv1(y)))

        return y


class ImageTransformNet_Spectrogram(nn.Module):
    def __init__(self, img_dim=128):
        super(ImageTransformNet_Spectrogram, self).__init__()
        self.img_dim = img_dim
        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # # encoding layers
        # self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=1)
        # self.in1_e = nn.InstanceNorm2d(32, affine=True)

        # self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        # self.in2_e = nn.InstanceNorm2d(64, affine=True)

        # self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        # self.in3_e = nn.InstanceNorm2d(128, affine=True)

        # encoding layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(32, affine=True)

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(64, affine=True)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(128, affine=True)

        # residual layers
        self.num_reslayers = 5
        self.res = nn.ModuleList()
        for i in range(self.num_reslayers):
            self.res.append(ResidualBlock(128))

        # decoding layers
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv1 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        for i in range(self.num_reslayers):
            y = self.res[i](y)


        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        y = self.relu(self.tanh(self.in1_d(self.deconv1(y)))*2.0)
        return y.view(-1, 1, 3*self.img_dim, self.img_dim)

class ImageTransformNet_Spectrogram_Sentence(nn.Module):
    def __init__(self, img_col=128, img_row=1280):
        super(ImageTransformNet_Spectrogram_Sentence, self).__init__()
        self.img_col, self.img_row = img_col, img_row
        # nonlineraity
        self.relu = nn.ReLU()
        self.leaky_relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer(3, 16, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(16, affine=True)

        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(32, affine=True)

        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(64, affine=True)

        # residual layers
        self.num_reslayers = 5
        self.res = nn.ModuleList()
        for i in range(self.num_reslayers):
            self.res.append(ResidualBlock(64))

        # decoding layers
        self.deconv3 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv2 = UpsampleConvLayer(32, 16, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.InstanceNorm2d(16, affine=True)

        self.deconv1 = UpsampleConvLayer(16, 3, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x):
        # encode
        y = self.leaky_relu(self.in1_e(self.conv1(x)))
        y = self.leaky_relu(self.in2_e(self.conv2(y)))
        y = self.leaky_relu(self.in3_e(self.conv3(y)))
        # residual layers
        for i in range(self.num_reslayers):
            y = self.res[i](y)

        # decode
        y = self.leaky_relu(self.in3_d(self.deconv3(y)))
        y = self.leaky_relu(self.in2_d(self.deconv2(y)))
        y = self.sigmoid(self.in1_d(self.deconv1(y)))
        return y.view(-1, 1, 3*self.img_col, self.img_row)



class ImageTransformNet_Phase_Spectrogram(nn.Module):
    def __init__(self, img_dim=128):
        super(ImageTransformNet_Phase_Spectrogram, self).__init__()
        self.img_dim = img_dim
        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # # encoding layers
        # self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=1)
        # self.in1_e = nn.InstanceNorm2d(32, affine=True)

        # self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        # self.in2_e = nn.InstanceNorm2d(64, affine=True)

        # self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        # self.in3_e = nn.InstanceNorm2d(128, affine=True)

        # encoding layers
        self.conv1 = ConvLayer(3*3, 3*32, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(3*32, affine=True)

        self.conv2 = ConvLayer(3*32, 3*64, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(3*64, affine=True)

        self.conv3 = ConvLayer(3*64, 3*128, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(3*128, affine=True)

        # residual layers
        self.num_reslayers = 9
        self.res = nn.ModuleList()
        for i in range(self.num_reslayers):
            self.res.append(ResidualBlock(3*128))

        # decoding layers
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv1 = UpsampleConvLayer(32, 2, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(2, affine=True)

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        for i in range(self.num_reslayers):
            y = self.res[i](y)


        y = y.view(-1, 128, 3*(self.img_dim//4), (self.img_dim//4))

        # decode
        y = self.in3_d(self.deconv3(y))
        y = self.in2_d(self.deconv2(y))
        y = self.tanh(self.in1_d(self.deconv1(y)))

        return y


class ClassNet(nn.Module):

    def __init__(self, img_col=128, img_row=32, num_classes=36):
        super(ClassNet, self).__init__()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.drop1d = nn.Dropout(p=0.25)
        self.drop2d = nn.Dropout2d(p=0.25)

        # encoding layers
        self.conv1 = ConvLayer(3, 32, kernel_size=3, stride=1)
        self.in1_e = nn.InstanceNorm2d(32, affine=True)

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(64, affine=True)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(128, affine=True)

        # residual layers
        self.num_reslayers = 3
        self.res = nn.ModuleList()
        for i in range(self.num_reslayers):
            self.res.append(ResidualBlock(128))

        self.feature_dim = (img_col//4)*(img_row//4)*128

        self.feature_dim2 = 1024
        self.fc1 = nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim2*2)

        self.fc2 = nn.Linear(in_features=self.feature_dim2, out_features=num_classes)

    def reparameterize(self, mu, logvar):

        sigma = 1e-6 + torch.log(1.0 + torch.exp(logvar))

        if self.training:

            noise = sigma * torch.normal(torch.zeros(mu.size(), dtype=torch.float)).cuda()
            z = mu + noise

            return z, sigma

        else:
            return mu, sigma

    def forward(self, x):

        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        for i in range(self.num_reslayers):
            y = self.res[i](y)

        y = y.view(-1, self.feature_dim)

        y = self.fc1(y)
        mu = y[:, 0:self.feature_dim2]
        logvar = y[:, self.feature_dim2:2*self.feature_dim2]

        z, sigma = self.reparameterize(mu, logvar)
        # z, sigma = mu, torch.zeros(mu.size()).float().cuda()

        logits = self.fc2(z)

        return z, mu, sigma, logits

    def loss(self, mu, sigma, logits, label):

        KLD = 0.5 * torch.sum(sigma**2 + mu**2 - torch.log(1e-8 + sigma**2) - 1)

        y_xent = F.cross_entropy(logits, label)

        return KLD, y_xent
