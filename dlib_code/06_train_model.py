#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/22 下午2:16
# @Author  : Leon
# @Site    : 
# @File    : 06_train_model.py
# @Software: PyCharm
# @Description: 训练模型
# 参考：https://blog.csdn.net/hongbin_xu/article/details/78443289
import dlib

# options用于设置训练的参数和模式
options = dlib.simple_object_detector_training_options()
# Since faces are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.
options.add_left_right_image_flips = True
# 支持向量机的C参数，通常默认取为5
options.C = 5
# 线程数
options.num_threads = 4
options.be_verbose = True

# 获取路径
train_folder = 'img/cats_train/'
test_folder = 'img/cats_test/'
train_xml_path = 'img/cats_train/cat.xml'
test_xml_path = 'img/cats_test/cats.xml'

print('training file path:' + train_xml_path)
# print(train_xml_path)
print('testing file path:' + test_xml_path)
# print(test_xml_path)

# 开始训练
print('start training:')
dlib.train_simple_object_detector(train_xml_path, 'model/06_detector.svm', options)

# 训练集/测试集 的 召回率/查全率、精度/准确率
print('Training accuracy: {}'.format(dlib.test_simple_object_detector(train_xml_path, 'model/06_detector.svm')))
# Training accuracy: precision: 1, recall: 0.722222, average precision: 0.722222
print('Testing accuracy: {}'.format(dlib.test_simple_object_detector(test_xml_path, 'model/06_detector.svm')))
# Testing accuracy: precision: 1, recall: 0.727273, average precision: 0.727273
