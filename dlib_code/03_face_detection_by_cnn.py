#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/22 上午10:38
# @Author  : Leon
# @Site    : 
# @File    : 03_face_detection_by_cnn.py
# @Software: PyCharm
# @Description: 调用cnn人脸检测
# 参考：https://blog.csdn.net/hongbin_xu/article/details/78359520
import dlib
import cv2

# 导入cnn模型
cnn_face_detector = dlib.cnn_face_detection_model_v1('dat_file/mmod_human_face_detector.dat')

img_paths = ['img/children1.jpg', 'img/children2.jpeg']
for img_path in img_paths:
    # opencv 读取图片，并显示
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # opencv的bgr格式图片转换成rgb格式
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])

    # 进行检测
    dets = cnn_face_detector(img, 1)

    # 打印检测到的人脸数
    print('Number of faces detected: {}'.format(len(dets)))
    # 遍历返回的结果
    # 返回的结果是一个mmod_rectangles对象。这个对象包含有2个成员变量：dlib.rectangle类，表示对象的位置；dlib.confidence，表示置信度。
    for i, d in enumerate(dets):
        face = d.rect
        print('Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}'.format(i, face.left(), face.top(),
                                                                                          face.right(), d.rect.bottom(),
                                                                                          d.confidence))

        # 在图片中标出人脸
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.namedWindow(img_path, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(img_path, img)

k = cv2.waitKey(0)
cv2.destroyAllWindows()
