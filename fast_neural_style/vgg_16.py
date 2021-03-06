#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/13 下午2:32
# @Author  : Leon
# @Site    : 
# @File    : Vgg16.py
# @Software: PyCharm
# @Description: 计算高层特征 VGG-16
import torch
import torch.nn as nn
from collections import namedtuple
from torchvision.models import vgg16


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)
