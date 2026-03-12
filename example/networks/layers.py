import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .masker import Masker
from .carafe import CARAFE


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class Projector(nn.Module):
    def __init__(self, word_dim=512, in_dim=2, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(
            conv_layer(in_dim, 64, 3, padding=1), # 增加特征表达能力
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_layer(64, in_dim, 3, padding=1), # 降回原通道数
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )
        # textual projector
        out_dim = in_dim * in_dim * kernel_size * kernel_size + 2
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        B, C, H, W = x.size()

        word = self.txt(word)
        weight, bias = word[:, :-2], word[:, -2:]
        bias=bias.reshape(2)
        weight = weight.reshape(C, C, self.kernel_size, self.kernel_size)
        out = F.conv2d(x,weight,padding=self.kernel_size // 2,bias=bias)
        return out




class FPN_AD(nn.Module):
    def __init__(self,
                 in_channels=[32, 256, 256],
                 out_channels=[32, 32, 32]):
        super(FPN_AD, self).__init__()

        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))

        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)

        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)

        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)

        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))
        self.masker3 = Masker(32, 32)
        self.masker4 = Masker(32, 32)
        self.masker5 = Masker(32, 32)
        self.carafe = CARAFE(in_channels[2],in_channels[2],up_factor=2)
        self.upsample_cas = CARAFE(out_channels[1], out_channels[1], up_factor=2)
        self.upsample_confound = CARAFE(out_channels[1], out_channels[1], up_factor=2)
    def forward(self, imgs):
        #v3 32*512*512 v4 256*256*256 v5 256*128*128
        v3, v4, v5 = imgs
        fq3 = F.avg_pool2d(v3, 2, 2)  # 32*256*256

        mask3_sup = self.masker3(fq3) 
        mask3_inf = torch.ones_like(mask3_sup) - mask3_sup 
        fq3_sup = fq3 * mask3_sup
        fq3_inf = fq3 * mask3_inf
        
        f4 = self.f2_v_proj(v4) # 32*256*256
        fq4 = self.f4_proj4(f4)
        mask4_sup = self.masker4(fq4)
        mask4_inf = torch.ones_like(mask4_sup) - mask4_sup
        fq4_sup = fq4 * mask4_sup
        fq4_inf = fq4 * mask4_inf

        fq5 = self.carafe(v5)
        fq5 = self.f1_v_proj(fq5) 
        mask5_sup = self.masker5(fq5)
        mask5_inf = torch.ones_like(mask5_sup) - mask5_sup
        fq5_sup = fq5 * mask5_sup
        fq5_inf = fq5 * mask5_inf

        fq_sup = torch.cat([fq3_sup, fq4_sup, fq5_sup], dim=1)
        fq_sup = self.aggr(fq_sup)

        fq_inf = torch.cat([fq3_inf, fq4_inf, fq5_inf], dim=1)
        fq_inf = self.aggr(fq_inf)
        
        fq_sup = self.upsample_cas(fq_sup)
        fq_inf = self.upsample_confound(fq_inf)

        return fq_sup , fq_inf
