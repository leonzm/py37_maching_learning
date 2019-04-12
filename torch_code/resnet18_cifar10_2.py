#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/11 下午5:33
# @Author  : Leon
# @Site    : 
# @File    : resnet18_cifar10_2.py
# @Software: PyCharm
# @Description: 使用 CIFAR-10 图像集训练 ResNet-18
# 因为原 ResNet-18 的输出是 (3, 224, 224)，而 CIFAR-10 的大小是 (3, 32, 32)，故重写 ResNet18
# 参考：https://blog.csdn.net/sunqiande88/article/details/80100891
# 显存不够可调整测试集使用量
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(  # input (3, 32, 32)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )  # output (63, 32, 32)
        self.layer1 = self.make_layer(block, 64, 2, stride=1)  # output (64, 32, 32)
        self.layer2 = self.make_layer(block, 128, 2, stride=2)  # output (128, 16, 16)
        self.layer3 = self.make_layer(block, 256, 2, stride=2)  # output (256, 8, 8)
        self.layer4 = self.make_layer(block, 512, 2, stride=2)  # output (512, 4, 4)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):  # input (3, 32, 32)
        out = self.conv1(x)  # output (63, 32, 32)
        out = self.layer1(out)  # output (64, 32, 32)
        out = self.layer2(out)  # output (128, 16, 16)
        out = self.layer3(out)  # output (256, 8, 8)
        out = self.layer4(out)  # output (512, 4, 4)
        out = F.avg_pool2d(out, 4)  # output (512, 1, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_resnet18():
    return ResNet(ResidualBlock)


EPOCH = 135  # 训练次数
BATCH_SIZE = 128
LR = 0.1
CIFAR_LABEL_NAMES = ['airplane', 'automobile', 'brid', 'cat', 'deer', 'dog', 'frog', 'horse', 'skip', 'truck']
DOWNLOAD_FILE_PATH = './cifar/'
DOWNLOAD_CIFAR = False

# 定义是否使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('使用 {} 训练'.format('gpu' if torch.cuda.is_available() else 'cpu'))

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B 每层的归一化用到的均值和方差
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = torchvision.datasets.CIFAR10(root=DOWNLOAD_FILE_PATH, train=True, download=DOWNLOAD_CIFAR, transform=transform_train)  # 训练数据集
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

test_data = torchvision.datasets.CIFAR10(root=DOWNLOAD_FILE_PATH, train=False)
test_x = torch.from_numpy(test_data.test_data[:]).type(torch.FloatTensor) / 255
test_x = torch.transpose(torch.transpose(test_x, 2, 3), 1, 2)  # (50000, 32, 32, 3) -> (50000, 3, 32, 32)
test_y = torch.Tensor(test_data.test_labels[:]).type(torch.LongTensor)  # 注意 y 必须是 LongTensor 类型
test_x, test_y = test_x.to(device), test_y.to(device)
test_x, test_y = Variable(test_x), Variable(test_y)

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

resnet18 = get_resnet18().to(device)
# print(resnet18)

# 定义损失函数和优化方式
loss_func = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = torch.optim.SGD(resnet18.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

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

            # 测试集的准确率
            test_output = resnet18(test_x)
            # 测试集损失
            test_loss = loss_func(test_output, test_y)
            # 测试集准确率
            test_predict_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
            test_accuracy = float((test_predict_y == test_y.data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, 'step: ', step,
                  '| train loss: %.4f' % loss.data.cpu().numpy(),
                  '| test loss: %.4f' % test_loss.data.cpu().numpy(),
                  '| train accuracy: %.2f' % train_accuracy,
                  '| test accuracy: %.2f' % test_accuracy)
        pass
    pass

