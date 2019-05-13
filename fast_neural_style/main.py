#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/13 下午2:38
# @Author  : Leon
# @Site    : 
# @File    : main.py
# @Software: PyCharm
# @Description:
import tqdm
import torch as t
import torchvision as tv
import torchnet as tnt
from torch.utils import data
from torch.autograd import Variable
from torch.nn import functional as F
from fast_neural_style.vgg_16 import Vgg16
from fast_neural_style.config import Config
from fast_neural_style.transformer_net import TransformerNet
from fast_neural_style.util import Visualizer, get_style_data, gram_matrix, normalize_batch


def train(**kwargs):
    """
    训练模型
    :param kwargs:
    :return:
    """
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    vis = Visualizer(opt.env)

    # 数据加载
    transfroms = tv.transforms.Compose([
        tv.transforms.Scale(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 255)
    ])
    dataset = tv.datasets.ImageFolder(opt.data_root, transfroms)
    dataloader = data.DataLoader(dataset, opt.batch_size)

    # 转换网络
    transformer = TransformerNet()
    if opt.model_path:
        transformer.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))

    # 损失网络 Vgg16
    vgg = Vgg16().eval()

    # 优化器
    optimizer = t.optim.Adam(transformer.parameters(), opt.lr)

    # 获取风格图片的数据
    style = get_style_data(opt.style_path)
    vis.img('style', (style[0] * 0.225 + 0.45).clamp(min=0, max=1))

    if opt.use_gpu:
        transformer.cuda()
        style = style.cuda()
        vgg.cuda()

    # 风格图片的gram矩阵
    style_v = Variable(style, volatile=True)
    features_style = vgg(style_v)
    gram_style = [Variable(gram_matrix(y.data)) for y in features_style]

    # 损失统计
    style_meter = tnt.meter.AverageValueMeter()
    content_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.epoches):
        content_meter.reset()
        style_meter.reset()

        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):

            # 训练
            optimizer.zero_grad()
            if opt.use_gpu:
                x = x.cuda()
            x = Variable(x)
            y = transformer(x)
            y = normalize_batch(y)
            x = normalize_batch(x)
            features_y = vgg(y)
            features_x = vgg(x)

            # content loss
            content_loss = opt.content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)

            # style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gram_y = gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            # 损失平滑
            content_meter.add(content_loss.data[0])
            style_meter.add(style_loss.data[0])

            if (ii + 1) % opt.plot_every == 0:
                # 可视化
                vis.plot('content_loss', content_meter.value()[0])
                vis.plot('style_loss', style_meter.value()[0])
                # 因为x和y经过标准化处理(normalize_batch)，所以需要将它们还原
                vis.img('output', (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                vis.img('input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

        # 保存visdom和模型
        vis.save([opt.env])
        t.save(transformer.state_dict(), 'checkpoints/%s_style.pth' % epoch)


def style(content_img_path='content_image.jpg', result_path='output.jpg'):
    """
    使用训练好的模型进行风格迁移

    :param content_img_path: str
    :param result_path: str
    :return:
    """
    opt = Config()

    # 图片处理
    content_image = tv.datasets.folder.default_loader(content_img_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    content_image = Variable(content_image, volatile=True)

    # 模型
    style_model = TransformerNet().eval()
    style_model.load_state_dict(
        t.load('checkpoints/1_style.pth', map_location=lambda _s, _: _s))

    if opt.use_gpu:
        content_image = content_image.cuda()
        style_model.cuda()
    else:
        content_image = content_image.cpu()
        style_model.cpu()

    # 风格迁移与保存
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image((output_data / 255).clamp(min=0, max=1), result_path)
    print('风格迁移完毕，输出地址：{}'.format(result_path))


if __name__ == '__main__':
    # 训练
    # train()

    # 风格迁移
    style()
    pass
