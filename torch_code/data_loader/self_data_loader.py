#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/17 下午3:53
# @Author  : Leon
# @Site    : 
# @File    : self_data_loader.py
# @Software: PyCharm
# @Description: 自实现数据集包装，效果与 ImageFolder 相似
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class DefaultDataset1(Dataset):
    """
    数据集加载器1
    支持的加载文件组织
    -- root_dir
        -- label1
            -- img1.png
            -- img2.png
            -- img3.png
            -- img3.png
            -- ...
        -- label2
            -- img1.png
            -- img2.png
            -- img3.png
            -- img3.png
            -- ...
        -- label3
            -- img1.png
            -- img2.png
            -- img3.png
            -- img3.png
            -- ...
        -- ...
    """

    def __init__(self, root_dir, transform=None):
        """
        初始化

        :param root_dir: str。根目录文件夹路径
        :param transform: torchvision.transforms。图像的操作
        """
        self.root_dir = root_dir.replace('//', '/').replace('\\', '/')
        self.root_dir = self.root_dir[:-1] if self.root_dir.endswith('/') else self.root_dir
        self.transform = transform

        # 加载数据集所有文件路径
        file_list = []  # [label_name, file_path]
        label_names = os.listdir(self.root_dir)
        for label_name in sorted(label_names):
            for file_name in os.listdir('{}/{}'.format(self.root_dir, label_name)):
                file_path = '{}/{}/{}'.format(self.root_dir, label_name, file_name)
                file_list.append([label_name, file_path])
                pass
            pass
        # print('加载数据集所有文件路径数：{}'.format(len(file_list)))
        self.file_list = file_list

        # 标签编码。0, 1, 2
        label_index = {}  # {label_name: index}
        for label_name in label_names:
            label_index[label_name] = len(label_index)
        self.label_index = label_index

    def __getitem__(self, index):
        label_name, file_path = self.file_list[index]
        image = Image.open(file_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label_index[label_name]

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    dateset1 = DefaultDataset1(root_dir='data',
                               transform=transforms.Compose([
                                   transforms.Resize((100, 100)),  # 图像缩放，transforms.Scale(100)
                                   transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
                                   transforms.ToTensor(),  # PIL.Image (H,W,C) -> FloadTensor [C,H,W]
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B 每层的归一化用到的均值和方差
                               ]))

    # 逐一加载样本
    # for i, (img, label) in enumerate(dateset1):
    #     print(i, img.shape, label)  # img.shape = (3, 100, 100)
    #     pass

    # 通过 DataLoader 的方式批量加载样本
    dataloader = DataLoader(dateset1,
                            batch_size=2,  # 每批次多少数据
                            shuffle=True,  # 是否打乱
                            num_workers=4  # 加载数据线程数
                            )
    for step, batch_sample in enumerate(dataloader):
        images, labels = batch_sample
        print('step: {} | image size: {} | label size: {} | label: {}'.format(step, len(images), len(labels), labels))
    pass
# step: 0 | image size: 2 | label size: 2 | label: tensor([0, 1])
# step: 1 | image size: 2 | label size: 2 | label: tensor([1, 0])
# step: 2 | image size: 2 | label size: 2 | label: tensor([2, 2])
# step: 3 | image size: 2 | label size: 2 | label: tensor([2, 0])
# step: 4 | image size: 2 | label size: 2 | label: tensor([2, 1])
# step: 5 | image size: 2 | label size: 2 | label: tensor([1, 0])
