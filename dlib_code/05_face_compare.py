#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/22 上午10:49
# @Author  : Leon
# @Site    : 
# @File    : 05_face_compare.py
# @Software: PyCharm
# @Description: 人脸对比
# 参考：https://blog.csdn.net/hongbin_xu/article/details/78390982
import dlib
import cv2
import numpy as np
import pandas as pd

# 读入模型
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('dat_file/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dat_file/dlib_face_recognition_resnet_model_v1.dat')

# 需要比对的人脸图片集合
img_file_paths = ['img/Jeong-eun1.png', 'img/Jeong-eun2.png', 'img/Jeong-eun3.png',
                  'img/Putin1.png', 'img/Putin2.png', 'img/Putin3.png',
                  'img/Trump1.png', 'img/Trump2.png', 'img/Trump3.png']

face_descriptors = {}  # {img_name: face_descriptor}
for img_file_path in img_file_paths:
    # opencv 读取图片，并显示
    img = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
    # opencv 的bgr格式图片转换成rgb格式
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])

    dets = detector(img2, 1)  # 人脸标定，参数：image[adarray], upsample_num_times[int] 上采样值，1 即原图，2 即放大为原来的两倍
    # print("Number of faces detected: {}".format(len(dets)))

    # 画出面部特征点
    face_descriptor = None
    for i, face in enumerate(dets):
        # print('face {}; left {}; top {}; right {}; bottom {}'.format(i, face.left(), face.top(), face.right(), face.bottom()))
        features = shape_predictor(img2, face)  # 提取68个特征点
        # print(type(features.parts()), len(features.parts()))  # <class 'dlib.points'> 68
        for j, pt in enumerate(features.parts()):
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
        cv2.namedWindow(img_file_path, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(img_file_path, img)

        # 计算人脸的128维的向量
        face_descriptor = face_rec_model.compute_face_descriptor(img2, features)
        # print(type(face_descriptor), len(face_descriptor))  # <class 'dlib.vector'> 128
        face_descriptors[img_file_path.split('/')[1].split('.')[0]] = list(face_descriptor)

# 计算各人脸两两的距离
data = []
for img_name1 in sorted(face_descriptors.keys()):
    vec1 = face_descriptors[img_name1]
    row = []
    for img_name2 in sorted(face_descriptors.keys()):
        vec2 = face_descriptors[img_name2]
        distance = np.linalg.norm(np.array(vec1) - np.array(vec2))  # 欧式距离
        row.append(distance)
    data.append(row)
df = pd.DataFrame(data, columns=sorted(face_descriptors.keys()), index=sorted(face_descriptors.keys()))
print(df)
#             Jeong-eun1  Jeong-eun2  Jeong-eun3    Putin1    Putin2    Putin3    Trump1    Trump2    Trump3
# Jeong-eun1    0.000000    0.386690    0.104262  0.846605  0.819661  0.805846    0.813001  0.827334  0.799998
# Jeong-eun2    0.386690    0.000000    0.377021  0.871688  0.846151  0.814723    0.831483  0.885501  0.851240
# Jeong-eun3    0.104262    0.377021    0.000000  0.861081  0.830948  0.819738    0.820362  0.836118  0.802373
# Putin1        0.846605    0.871688    0.861081  0.000000  0.272036  0.402693    0.712809  0.757410  0.800893
# Putin2        0.819661    0.846151    0.830948  0.272036  0.000000  0.371581    0.698071  0.725311  0.723036
# Putin3        0.805846    0.814723    0.819738  0.402693  0.371581  0.000000    0.742176  0.793913  0.798830
# Trump1        0.813001    0.831483    0.820362  0.712809  0.698071  0.742176    0.000000  0.472453  0.485686
# Trump2        0.827334    0.885501    0.836118  0.757410  0.725311  0.793913    0.472453  0.000000  0.447227
# Trump3        0.799998    0.851240    0.802373  0.800893  0.723036  0.798830    0.485686  0.447227  0.000000

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()
