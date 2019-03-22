#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/22 上午10:34
# @Author  : Leon
# @Site    : 
# @File    : 02_face_feature_point_calibration.py
# @Software: PyCharm
# @Description: 人脸特征点标定
# 参考：https://blog.csdn.net/hongbin_xu/article/details/78348086
import dlib
import cv2

# shape_predictor_68_face_landmarks.dat 是进行人脸标定的模型，它是基于HOG特征的，这里是他所在的路径
face_landmark_file_path = 'dat_file/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()  # 获取人脸分类器
predictor = dlib.shape_predictor(face_landmark_file_path)  # 获取人脸检测器

img_paths = ['img/children1.jpg', 'img/children2.jpeg']
for img_path in img_paths:
    # opencv 读取图片，并显示
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # 摘自官方文档：
    # image is a numpy ndarray containing either an 8bit grayscale or RGB image.
    # opencv读入的图片默认是bgr格式，我们需要将其转换为rgb格式；都是numpy的ndarray类。
    b, g, r = cv2.split(img)  # 分离三个颜色通道
    img2 = cv2.merge([r, g, b])  # 融合三个颜色通道生成新图片

    dets = detector(img, 1)  # 使用detector进行人脸检测 dets为返回的结果
    print('Number of faces detected: {}'.format(len(dets)))  # 打印识别到的人脸个数
    # enumerate是一个Python的内置方法，用于遍历索引
    # index是序号；face是dets中取出的dlib.rectangle类的对象，包含了人脸的区域等信息
    # left()、top()、right()、bottom()都是dlib.rectangle类的方法，对应矩形四条边的位置
    for index, face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
                                                                     face.bottom()))

        # 这里不需要画出人脸的框了
        # left = face.left()
        # top = face.top()
        # right = face.right()
        # bottom = face.bottom()
        # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        # cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
        # cv2.imshow(f, img)

        shape = predictor(img, face)  # 寻找人脸的68个标定点
        # print(shape)
        # print(shape.num_parts)
        # 遍历所有点，打印出其坐标，并用蓝色的圈表示出来
        for i, pt in enumerate(shape.parts()):
            print('Part {}: {}'.format(i, pt))
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)

        # 在新窗口中显示
        cv2.namedWindow(img_path, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(img_path, img)

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()
