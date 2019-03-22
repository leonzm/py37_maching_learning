#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/22 下午3:59
# @Author  : Leon
# @Site    : 
# @File    : 08_train_model_by_face_every_points.py
# @Software: PyCharm
# @Description: 训练人脸特征点检测器
# 使用该训练好的模型，参考 02_face_feature_point_calibration.py
# 参考：https://blog.csdn.net/hongbin_xu/article/details/78511923
# 论文：http://www.nada.kth.se/~sullivan/Papers/Kazemi_cvpr14.pdf
import dlib

# 参数设置
options = dlib.shape_predictor_training_options()
options.oversampling_amount = 300
options.nu = 0.05
options.tree_depth = 2
options.be_verbose = True

# 导入已标记的标签 xml 文件
training_xml_path = 'img/faces/training_with_face_landmarks.xml'
# 进行训练，训练好的模型将保存为 predictor.dat
dlib.train_shape_predictor(training_xml_path, 'model/08_predictor.dat', options)

# 训练集的准确率
print('\nTraining accuracy:{0}'.format(dlib.test_shape_predictor(training_xml_path, 'model/08_predictor.dat')))
# 导入测试集的xml文件
testing_xml_path = 'img/faces/testing_with_face_landmarks.xml'
# 测试集的准确率
print('\Testing accuracy:{0}'.format(dlib.test_shape_predictor(testing_xml_path, 'model/08_predictor.dat')))
