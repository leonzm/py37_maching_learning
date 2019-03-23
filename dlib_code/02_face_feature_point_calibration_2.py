#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/23 上午9:31
# @Author  : Leon
# @Site    : 
# @File    : 02_face_feature_point_calibration_2.py.py
# @Software: PyCharm
# @Description: 通过摄像头进行人脸特征点标定
import dlib
import cv2

# shape_predictor_68_face_landmarks.dat 是进行人脸标定的模型，它是基于HOG特征的，这里是他所在的路径
face_landmark_file_path = 'dat_file/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()  # 获取人脸分类器
predictor = dlib.shape_predictor(face_landmark_file_path)  # 获取人脸检测器

cap = cv2.VideoCapture(0)  # 0为默认计算机默认摄像头，1可以更换来源
while True:
    ret, img = cap.read()
    # 宽和高改成 640x480
    # cap.set(3, 640)
    # cap.set(4, 480)

    cv2.imshow('frame', img)

    # 面部特征点
    b, g, r = cv2.split(img)  # 分离三个颜色通道
    img2 = cv2.merge([r, g, b])  # 融合三个颜色通道生成新图片
    dets = detector(img, 1)  # 使用detector进行人脸检测 dets为返回的结果
    for index, face in enumerate(dets):
        shape = predictor(img, face)  # 寻找人脸的68个标定点
        for i, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
        cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break
        # when everything done , release the capture
cap.release()
cv2.destroyAllWindows()
