
from turtle import down
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]


# Ups--Resblock(BottleneckResblock)
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BottleneckResblock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckResblock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # 利用1x1卷积下降通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 利用3x3卷积进行特征提取
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 利用1x1卷积上升通道数
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Downs--ConvBlock_3,2
class ConvBlock32(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock32,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,2,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)
# Downs--ConvBlock_3,1
class ConvBlock31(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock31,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)
# Middles--SEBLock
class SEBlock(nn.Module):
    def __init__(self,channel,ratio=16):
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//ratio,False),
            nn.ReLU(),
            nn.Linear(channel//ratio,channel,False),
            nn.Sigmoid(),
        )
    def forward(self,x):
        b,c,h,w=x.size()
        # b,c,h,w->b,c,1,1
        avg=self.avg_pool(x).view([b,c])
        # b,c->b,c//ratio->b,c->b,c,1,1
        fc=self.fc(avg).view([b,c,1,1])
        ###### 注意这里论文好像还加了原始的featuremap
        return x*fc

# Middles--PAM+CAM
__all__ = ['PAM_Module', 'CAM_Module']
class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

# Middles--ASPP
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[1,3,6,9], out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

# Middles--MSCBlock
# Middles--ConvBlock11
class ConvBlock11(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock11,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)
class MSCBlock(nn.Module):
    def __init__(self,in_channels,out_channels,atrous_rates):
        super(MSCBlock,self).__init__()
        self.aspp=ASPP(in_channels,atrous_rates,out_channels)
        self.conv31=ConvBlock31(in_channels,out_channels)
        self.pam=PAM_Module(in_channels)
        self.cam=CAM_Module(in_channels)
        self.conv11=ConvBlock11(in_channels,out_channels)
    def forward(self,x):
        aspp=self.aspp(x)
        conv31out=self.conv31(aspp)
        pam=self.pam(conv31out)
        cam=self.cam(conv31out)
        concat=torch.cat((pam,cam),dim=1)
        # print(concat.shape)
        # torch.Size([8, 512, 126, 126])
        conv11out=self.conv11(concat)
        return conv11out

# if __name__ == "__main__":
#     model = MSCBlock(64,64,[1,3,6,9])
#     feature_maps = torch.randn((8, 64, 128, 128))
#     a=model(feature_maps)
#     print(feature_maps.shape)
#     print(a.shape)



class BLSNet(nn.Module):
    def __init__(self,in_channels,out_channels,features=[64,128,256,512],atrous_rates=[1,3,6,9]):
        super(BLSNet,self).__init__()
        self.downs=nn.ModuleList()
        self.ups=nn.ModuleList()
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
              
        # Down part of BLSNet
        self.downs.append(ConvBlock32(in_channels,features[0]))
        in_channels=features[0]
        self.downs.append(ConvBlock31(in_channels,features[1]))
        in_channels=features[1]
        self.downs.append(ConvBlock31(in_channels,features[2]))
        in_channels=features[2]
        self.downs.append(ConvBlock31(in_channels,features[3]))

        # print(features)
        # print(features[-1])

        # for feature in features:
        #     print(feature)
        #     self.downs.append(ConvBlock32(in_channels,feature[0]))
        #     self.downs.append(ConvBlock31(in_channels,feature))

        #     in_channels=feature


        # Up part of BLSNet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=3,stride=2))
            self.ups.append(BottleneckResblock(feature*2,feature))
            
            # self.bottleneck=BottleneckResblock(features[-1],features[-1]*2)
            # 11卷积，用做最后的输出分类
            self.final_conv=nn.Conv2d(features[0],out_channels,kernel_size=1)
            
        # Middle part of BLSNet(SEBlock+MSCBlock)
        self.se=SEBlock(in_channels)
        self.msc=MSCBlock(in_channels,out_channels,atrous_rates)

    def forward(self,x):
        # 四个下采样层（一个卷积+3个（卷积+池化），注意这里需要修改，第一个卷积后不应该跟池化）
        # 同时保留每个阶段下采样后的特征图，放到skip_connections中
        skip_connections=[]
        # print(self.downs[0],self.downs[1])
        down0,down1,down2,down3=self.downs[0],self.downs[1],self.downs[2],self.downs[3]
        x=down0(x)
        # print(x)
        skip_connections.append(x)
        print(x.shape)         
        x=self.se(x)

        x=down1(x)
        skip_connections.append(x)
        x=self.pool(x)
        x=self.se(x)

        x=down2(x)
        skip_connections.append(x)
        x=self.pool(x)
        x=self.se(x)

        x=down3(x)
        skip_connections.append(x)
        x=self.pool(x)
        x=self.msc(x)
        print(x)
        
        print(skip_connections)
        print("///////")
        skip_connections=skip_connections[::-1]
        print(skip_connections)
        
        for idx in range(0,len(self.ups),2):
            x=self.ups[idx](x)
            skip_connection=skip_connections[idx//2]

            if x.shape!=skip_connection.shape:
                x=TF.resize(x,size=skip_connection.shape[2:])

            concat_skip=torch.cat((skip_connection,x),dim=1)
            x=self.ups[idx+1](concat_skip)

        return self.final_conv(x)

if __name__ == '__main__':
    x=torch.randn(2,3,256,256)
    # br=BottleneckResblock()
    # print(br(x).shape)s
    net=BLSNet(3,1)
    net(x)
    # print(net(x).shape)

