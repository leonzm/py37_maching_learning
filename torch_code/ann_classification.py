#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 下午5:27
# @Author  : Leon
# @Site    : 
# @File    : ann_classification.py
# @Software: PyCharm
# @Description:
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 制作两个中心的分类模拟数据
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)
# 注意神经网络的输入必须是 FloatTensor 类型，输出必须是 LongTensor 类型
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # cat() 是合并数据，0 表示沿着从上到下的纬度连接张量
y = torch.cat((y0, y1), 0).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

plt.ion()  # 开启交互模式


# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.output = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, feature):
        # 正向传播输入值，神经网络分析输出值
        hidden_output = F.relu(self.hidden(feature))  # 激励函数（隐藏层的线性值）
        output = self.output(hidden_output)
        return output


# 有两个特征，所以有两个输入；输出有两个类别，所以有两个输出
net = Net(n_feature=2, n_hidden=10, n_output=2)
print(net)
# Net (
#   (hidden_output): Linear (2 -> 10)
#   (output): Linear (10 -> 2)
# )

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数、学习率
loss_func = torch.nn.CrossEntropyLoss()

for t in range(100):
    output = net(x)
    loss = loss_func(output, y)

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播，计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(output, 1)[1]  # max() 返回最大的值，0 为最大值，1 为最大值的下标
        pred_y = prediction.data.numpy().squeeze()  # np.squeeze() 函数可以删除数组形状中的单维度条目
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
    pass

plt.ioff()  # 停止画图
plt.show()
