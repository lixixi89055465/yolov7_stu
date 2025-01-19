# -*- coding: utf-8 -*-
# @Time : 2025/1/19 10:18
# @Author : nanji
# @Site : 
# @File : backbone.py
# @Software: PyCharm 
# @Comment :
import torch.nn as nn
import torch


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1,
                 s=1, p=None, g=1, act=SiLU()):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if \
            act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Multi_Concat_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)

        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i == 0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        x_all = [x_1, x_2]
        # [-1,-3,-5,-6]=>[5,3,1,0]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class Transition_Block(nn.Module):
    def __init__(self, c1, c2):
        super(Transition_Block, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)
        self.p = MP()

    def forward(self, x):
        # 160, 160, 256 => 80, 80, 256 => 80, 80, 128
        x_1 = self.mp(x)


class Backbone(nn.Module):
    def __init__(self,
                 transition_channels,
                 block_channels,
                 n,
                 phi,
                 pretrained=False):
        super().__init__()
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        # -----------------------------------------------#
        ids = {
            'l': [-1, -3, -5, -6],
            'x': [-1, -3, -5, -7, -8],
        }[phi]
        # 640, 640, 3 => 640, 640, 32 => 320, 320, 64
        self.stem = nn.Sequential(
            Conv(3, transition_channels, 3, 1),
            Conv(transition_channels, transition_channels * 2, 3, 2),
            Conv(transition_channels * 2, transition_channels * 2, 3, 1)
        )
        # 320,320,64 => 160,160,128=>160,160,256
        self.dark2 = nn.Sequential(
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),
            Multi_Concat_Block(transition_channels * 4,
                               block_channels * 2,
                               transition_channels * 8,
                               n=n, ids=ids)
        )
        # 160, 160, 256 => 80, 80, 256 => 80, 80, 512
        self.dark3 = nn.Sequential(

        )
