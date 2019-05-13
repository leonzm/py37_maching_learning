#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/13 下午2:35
# @Author  : Leon
# @Site    : 
# @File    : config.py
# @Software: PyCharm
# @Description:
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Config(object):
    image_size = 256  # 图片大小
    batch_size = 8
    data_root = 'data/style_transfer'  # 数据集存放路径：style_transfer/coco/a.jpg
    num_workers = 4  # 多线程加载数据
    use_gpu = False  # 使用GPU

    style_path = 'style_image.jpg'  # 风格图片存放路径
    lr = 1e-3  # 学习率

    env = 'neural-style'  # visdom env
    plot_every = 10  # 每10个batch可视化一次

    epoches = 2  # 训练epoch

    content_weight = 1e5  # content_loss 的权重
    style_weight = 1e10  # style_loss的权重

    model_path = None  # 预训练模型的路径
    debug_file = '/tmp/debugnn'  # touch $debug_fie 进入调试模式

    content_path = 'content_image.jpg'  # 需要进行风格迁移的图片
    result_path = 'output.png'  # 风格迁移结果的保存路径
