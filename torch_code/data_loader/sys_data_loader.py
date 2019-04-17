#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/17 下午3:51
# @Author  : Leon
# @Site    : 
# @File    : about_image_folder.py
# @Software: PyCharm
# @Description: torch 自带的 ImageFolder 与 DataLoader 的使用，文件必须遵循其规范
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# data 文件夹下文件组织：
# -- root_dir
#     -- label1
#         -- img1.png
#         -- img2.png
#         -- img3.png
#         -- img3.png
#         -- ...
#     -- label2
#         -- img1.png
#         -- img2.png
#         -- img3.png
#         -- img3.png
#         -- ...
#     -- label3
#         -- img1.png
#         -- img2.png
#         -- img3.png
#         -- img3.png
#         -- ...
#     -- ...

dataset = ImageFolder(root='data',
                      transform=transforms.Compose([
                          transforms.Resize((100, 100)),  # 图像缩放，transforms.Scale(100)
                          transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
                          transforms.ToTensor(),  # PIL.Image (H,W,C) -> FloadTensor [C,H,W]
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B 每层的归一化用到的均值和方差
                      ]))
dataloader = DataLoader(dataset,
                        batch_size=2,  # 每批次多少数据
                        shuffle=True,  # 是否打乱
                        num_workers=4  # 加载数据线程数
                        )
for step, batch_sample in enumerate(dataloader):
    images, labels = batch_sample
    print('step: {} | image size: {} | label size: {} | label: {}'.format(step, len(images), len(labels), labels))
    pass
# step: 0 | image size: 2 | label size: 2 | label: tensor([1, 0])
# step: 1 | image size: 2 | label size: 2 | label: tensor([0, 2])
# step: 2 | image size: 2 | label size: 2 | label: tensor([2, 1])
# step: 3 | image size: 2 | label size: 2 | label: tensor([1, 2])
# step: 4 | image size: 2 | label size: 2 | label: tensor([0, 2])
# step: 5 | image size: 2 | label size: 2 | label: tensor([0, 1])
