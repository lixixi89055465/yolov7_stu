# -*- coding: utf-8 -*-
# @Time : 2025/1/18 22:58
# @Author : nanji
# @Site : 
# @File : yolo.py
# @Software: PyCharm 
# @Comment :

import torch.nn as nn
from nets.backbone import SiLU, autopad, Conv, Backbone, Multi_Concat_Block
import torch


class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self,
                 c1,
                 c2,
                 k=3,
                 s=1,
                 p=None,
                 g=1,
                 act=SiLU(),
                 deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        assert k == 3
        assert autopad(k, p) == 1


class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        # 输出通道数为c2
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], dim=1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat(y1, y2), dim=1)


class YoloBody(nn.Module):
    def __init__(self,
                 anchors_mask,
                 num_classes,
                 phi,
                 pretrained=False,
                 ):
        super(YoloBody, self).__init__()
        # -----------------------------------------------#
        #   定义了不同yolov7版本的参数
        # -----------------------------------------------#
        transition_channels = {'l': 32, 'x': 40}[phi]
        block_channels = 32
        panet_channels = {'l': 32, 'x': 64}[phi]
        e = {'l': 2, 'x': 1}[phi]
        n = {'l': 4, 'x': 6}[phi]
        ids = {'l': [-1, -2, -3, -4, -5, -6], 'x': [-1, -3, -5, -7, -8]}[phi]
        conv = {'l': RepConv, 'x': Conv}[phi]
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        # ---------------------------------------------------#
        self.backbone = Backbone(transition_channels,
                                 block_channels,
                                 n,
                                 phi,
                                 pretrained=pretrained)
        # ------------------------加强特征提取网络------------------------#
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 20, 20, 1024 => 20, 20, 512
        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16)
        # 20, 20, 512 => 20, 20, 256 => 40, 40, 256
        self.conv_for_P5 = Conv(transition_channels * 16, transition_channels * 8)
        # 40, 40, 1024 => 40, 40, 256
        self.conv_for_feat2 = Conv(transition_channels * 32, transition_channels * 8)
        # 40, 40, 512 => 40, 40, 256
        self.conv3_for_upsample1 = Multi_Concat_Block(
            transition_channels * 16,
            panet_channels * 4,
            transition_channels * 8,
            e=e,
            n=n,
            ids=ids
        )
        self.down_sample1 = Transition_Block(transition_channels * 4, transition_channels * 4)
        # 40, 40, 512 => 40, 40, 256
        self.conv3_for_downsample1 = Multi_Concat_Block(
            transition_channels * 16,
            panet_channels * 4,
            transition_channels * 8,
            e=e,
            n=n,
            ids=ids
        )

        # 40, 40, 256 => 40, 40, 128 => 80, 80, 128
        self.conv_for_P4 = Conv(transition_channels * 8, transition_channels * 4)
        # 80, 80, 512 => 80, 80, 128
        self.conv_for_feat1 = Conv(transition_channels * 16, transition_channels * 4)
        # 80, 80, 256 => 80, 80, 128
        self.conv3_for_upsample2 = (
            Multi_Concat_Block(transition_channels * 8,
                               panet_channels * 2,
                               transition_channels * 4,
                               e=e,
                               n=n,
                               ids=ids))

    def forward(self, x):
        # backbone
        feat1, feat2, feat3 = self.backbone.forward(x)
        # ------------------------加强特征提取网络------------------------#
        # 20, 20, 1024 => 20, 20, 512
        P5 = self.sppcspc(feat3)
        # 20, 20, 512 => 20, 20, 256
        P5_conv = self.conv_for_p5(P5)
        # 20, 20, 256 => 40, 40, 256
        P5_upsample = self.upsample(P5_conv)
        # 40, 40, 256 cat 40, 40, 256 => 40, 40, 512
        P4 = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        # 40, 40, 512 => 40, 40, 256
        P4 = self.conv3_for_upsample1(P4)
        # 40, 40, 256 => 40, 40, 128
        P4_conv = self.conv_for_P4(P4)
        # 40, 40, 128 => 80, 80, 128
        P4_upsample = self.upsample(P4_conv)
        # 80, 80, 128 cat 80, 80, 128 => 80, 80, 256
        P3 = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        # 80, 80, 256 => 80, 80, 128
        P3 = self.conv3_for_upsample2(P3)

        # 80, 80, 128 => 40, 40, 256
        P3_downsample = self.down_sample1(P3)
