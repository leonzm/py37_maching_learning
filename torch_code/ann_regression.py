#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 下午5:25
# @Author  : Leon
# @Site    : 
# @File    : ann_regression.py
# @Software: PyCharm
# @Description:
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, feature):
        # 正向传播输入值，神经网络分析输出值
        hidden_output = F.relu(self.hidden(feature))  # 激励函数（隐藏层的线性值）
        output = self.predict(hidden_output)
        return output


net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)
# Net (
#   (hidden): Linear (1 -> 10)
#   (predict): Linear (10 -> 1)
# )

# 创建一些假数据。模拟 y = a * x^2 + b，给 y 加一些噪声
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 这里的unsqueeze() 方法用于给原张量增加一个维度
y = x.pow(2) + 0.2 * torch.rand(x.size())
x = Variable(x)
y = Variable(y)

plt.ion()  # 开启交互模式
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数、学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式（均方差）

for t in range(200):
    prediction = net(x)  # 计算输出值/预测值
    loss = loss_func(prediction, y)  # 计算两者的误差

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播，计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    if t % 5 == 0:
        plt.cla()  # 清除画布
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
    pass

plt.ioff()  # 停止画图
plt.show()
