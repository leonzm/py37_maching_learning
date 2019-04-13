#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/13 上午10:42
# @Author  : Leon
# @Site    : 
# @File    : spatial_pyramidpool_2d.py
# @Software: PyCharm
# @Description: SPP-Net，参考：https://oidiotlin.com/sppnet-tutorial/
import math
import torch
import torch.nn as nn


class SpatialPyramidPool2D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side=(1, 2, 4)):
        super(SpatialPyramidPool2D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        out = None
        for n in self.out_side:
            w_r, h_r = map(lambda s: math.ceil(s / n), x.size()[2:])  # Receptive Field Size
            s_w, s_h = map(lambda s: math.floor(s / n), x.size()[2:])  # Stride
            max_pool = nn.MaxPool2d(kernel_size=(w_r, h_r), stride=(s_w, s_h))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out


if __name__ == '__main__':
    from utils import image_util

    img = image_util.base64_to_img(image_util.img_to_base64('data/dog_1.jpg'))
    tensor_img = torch.Tensor(img)
    # 一个例子，batch=1
    tensor_img = tensor_img.view(1, tensor_img.size()[0], tensor_img.size()[1], tensor_img.size()[2])
    # 图像的 (宽度, 高度, 深度) 更新为 (深度, 宽度, 高度)
    tensor_img = torch.transpose(torch.transpose(tensor_img, 2, 3), 1, 2)  # (1, w, h, d) -> (1, w, d, h) -> (1, d, w, h)
    # print(tensor_img.size())  # (1, 3, 558, 576)

    pool = SpatialPyramidPool2D(out_side=(1, 2, 4))
    print(pool(tensor_img).size())  # (1, 63)
    pass
