#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/11 下午2:15
# @Author  : Leon
# @Site    : 
# @File    : resnet18_cifar10.py
# @Software: PyCharm
# @Description: # @Description: 使用 CIFAR-10 图像集训练 ResNet-18
# 因为原 ResNet-18 的输出是 (3, 224, 224)，而 CIFAR-10 的大小是 (3, 32, 32)，故这里复写 forward() 方法，去掉 fc 前的池化层的使用
# 如果内存不够用，可减少 batch_size 大小；也可以减少测试集的使用量
import os
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
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
    # Converts a PIL.Image or numpy.ndarray to # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_CIFAR
)
test_data = torchvision.datasets.CIFAR10(root=DOWNLOAD_FILE_PATH, train=False)
print('训练集，样本维度：{}，标签数：{}，标签类别数：{}'.format(train_data.train_data.shape, len(train_data.train_labels),
                                           len(set(train_data.train_labels))))
print('测试集，样本维度：{}，标签数：{}，标签类别数：{}'.format(test_data.test_data.shape, len(test_data.test_labels),
                                           len(set(test_data.test_labels))))
# 训练集，样本维度：(50000, 32, 32, 3)，标签数：50000，标签类别数：10
# 测试集，样本维度：(10000, 32, 32, 3)，标签数：10000，标签类别数：10


def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    # x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


torchvision.models.ResNet.forward = forward
resnet18 = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
# print(resnet18)
optimizer = torch.optim.Adam(resnet18.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 训练集 DataLoader
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 测试集 x, y。注意和训练集不同，这里需要手动将像素压到 [0.0, 1.0] 间。注意 训练集的数据必须是 FloatTensor 类型
test_x = torch.from_numpy(test_data.test_data[:]).type(torch.FloatTensor) / 255
test_x = torch.transpose(torch.transpose(test_x, 2, 3), 1, 2)  # (50000, 32, 32, 3) -> (50000, 3, 32, 32)
test_y = torch.Tensor(test_data.test_labels[:]).type(torch.LongTensor)  # 注意 y 必须是 LongTensor 类型
test_x, test_y = test_x.to(device), test_y.to(device)
test_x, test_y = Variable(test_x), Variable(test_y)

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
            test_output = resnet18(test_x)
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
# Epoch:  9 step:  400 | train loss: 0.2184 | test loss: 0.9151 | test accuracy: 0.76
