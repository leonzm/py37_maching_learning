#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 下午6:39
# @Author  : Leon
# @Site    : 
# @File    : test.py
# @Software: PyCharm
# @Description: 使用 CIFAR-10 图像集训练 CNN
import os
import torch
import random
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.autograd import Variable

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
    transform=torchvision.transforms.ToTensor(),
    # Converts a PIL.Image or numpy.ndarray to # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_CIFAR
)
test_data = torchvision.datasets.CIFAR10(root=DOWNLOAD_FILE_PATH, train=False)
print('训练集，样本维度：{}，标签数：{}，标签类别数：{}'.format(train_data.train_data.shape, len(train_data.train_labels),
                                           len(set(train_data.train_labels))))
print('测试集，样本维度：{}，标签数：{}，标签类别数：{}'.format(test_data.test_data.shape, len(test_data.test_labels),
                                           len(set(test_data.test_labels))))
# 训练集，样本维度：(50000, 32, 32, 3)，标签数：50000，标签类别数：10
# 测试集，样本维度：(10000, 32, 32, 3)，标签数：10000，标签类别数：10

# 随机显示一张图片
# random_img_index = random.randint(0, train_data.train_data.shape[0])
# plt.imshow(train_data.train_data[random_img_index])
# plt.title(CIFAR_LABEL_NAMES[train_data.train_labels[random_img_index]])
# plt.show()


# 普通卷积网络 INPUT -> [[CONV]*1 -> POOL]*2 -> [FC]*1。卷积后大小 (W - F + 2P)/S + 1
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Sequential(  # input shape (3, 32, 32)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=1),  # output shape (16, 30, 30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output shape (16, 15, 15)
        )
        self.cnn2 = nn.Sequential(  # input shape (16, 15, 15)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=1),  # output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接，输出 10 个类别

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图层
        out = self.out(x)
        return out


cnn = CNN().to(device)
# print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 训练集 DataLoader
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 测试集 x, y。注意和训练集不同，这里需要手动将像素压到 [0.0, 1.0] 间。注意 训练集的数据必须是 FloatTensor 类型
test_x = torch.from_numpy(test_data.test_data).type(torch.FloatTensor) / 255
test_x = torch.transpose(torch.transpose(test_x, 2, 3), 1, 2)  # (50000, 32, 32, 3) -> (50000, 3, 32, 32)
test_y = torch.Tensor(test_data.test_labels[:]).type(torch.LongTensor)  # 注意 y 必须是 LongTensor 类型
test_x, test_y = test_x.to(device), test_y.to(device)
test_x, test_y = Variable(test_x), Variable(test_y)

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = b_x.to(device), b_y.to(device)
        v_b_x, v_b_y = Variable(b_x), Variable(b_y)
        output = cnn(v_b_x)
        loss = loss_func(output, v_b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            test_output = cnn(test_x)
            # 测试集损失
            test_loss = loss_func(test_output, test_y)
            # 测试集准确率
            test_predict_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
            test_accuracy = float((test_predict_y == test_y.data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, 'step: ', step,
                  '| train loss: %.4f' % loss.data.cpu().numpy(),
                  '| test loss: %.4f' % test_loss.data.cpu().numpy(),
                  '| test accuracy: %.2f' % test_accuracy)
        pass
    pass
# Epoch:  9 step:  400 | train loss: 0.8363 | test loss: 0.9693 | test accuracy: 0.67
