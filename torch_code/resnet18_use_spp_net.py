#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/18 下午5:05
# @Author  : Leon
# @Site    : 
# @File    : resnet18_use_spp_net.py
# @Software: PyCharm
# @Description: 使用 CIFAR-10 图像集训练 ResNet-18，用 SPP-Net
# 这里为了测试，将图像手动调为 (896,896)，经过 SPP-Net 后为 512 * 21 = 10752
import os
import math
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable


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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.spp_net = SpatialPyramidPool2D()
        self.fc = nn.Linear(10752, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.spp_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

EPOCH = 10  # 训练次数
BATCH_SIZE = 100
LR = 0.001
CIFAR_LABEL_NAMES = ['airplane', 'automobile', 'brid', 'cat', 'deer', 'dog', 'frog', 'horse', 'skip', 'truck']
DOWNLOAD_FILE_PATH = './cifar/'
DOWNLOAD_CIFAR = False

# 定义是否使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('使用 {} 训练'.format('gpu' if torch.cuda.is_available() else 'cpu'))

# 加载训练集测试集
if not (os.path.exists(DOWNLOAD_FILE_PATH)) or not os.listdir(DOWNLOAD_FILE_PATH):
    DOWNLOAD_CIFAR = True
train_data = torchvision.datasets.CIFAR10(
    root=DOWNLOAD_FILE_PATH,
    train=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((896, 896)),
        torchvision.transforms.ToTensor()
    ]),
    download=DOWNLOAD_CIFAR
)
test_data = torchvision.datasets.CIFAR10(root=DOWNLOAD_FILE_PATH, train=False)
print('训练集，样本维度：{}，标签数：{}，标签类别数：{}'.format(train_data.train_data.shape, len(train_data.train_labels),
                                           len(set(train_data.train_labels))))
# 训练集，样本维度：(50000, 32, 32, 3)，标签数：50000，标签类别数：10


resnet18 = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
# print(resnet18)
optimizer = torch.optim.Adam(resnet18.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 训练集 DataLoader
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = b_x.to(device), b_y.to(device)
        v_b_x, v_b_y = Variable(b_x), Variable(b_y)
        output = resnet18(v_b_x)
        loss = loss_func(output, v_b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == (len(train_loader) - 1):
            # 训练集 batch 的准确率
            train_predict_y = torch.max(output, 1)[1].data.squeeze().cpu().numpy()
            train_accuracy = float((train_predict_y == v_b_y.data.cpu().numpy()).astype(int).sum()) / float(v_b_y.size(0))

            print('Epoch: ', epoch, 'step: ', step,
                  '| train loss: %.4f' % loss.data.cpu().numpy(),
                  '| train accuracy: %.2f' % train_accuracy)
        pass
    pass
