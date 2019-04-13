#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/13 上午11:00
# @Author  : Leon
# @Site    : 
# @File    : image_util.py
# @Software: PyCharm
# @Description:
import io
import cv2
import base64
import imageio


def img_to_base64(file_path):
    """
    图片转 base64

    :param file_path: str，图片文件路径
    :return: str
    """
    with open(file_path, 'rb') as f:
        encoded = base64.b64encode(f.read())
        return str(encoded, encoding='utf-8')


def base64_to_img_file(img_base64_str, file_path):
    """
    base64 转图片保存

    :param img_base64_str: str，图片的 base64 编码
    :param file_path: str，转成图片后的存储路径
    :return:
    """
    img_base64_bytes = base64.decodebytes(bytes(img_base64_str, encoding='utf-8'))
    with open(file_path, 'wb+') as f:
        f.write(img_base64_bytes)


def base64_to_img(img_base64_str):
    """
    base64 转（OpenCV）图片

    :param img_base64_str: str，图片的 base64 编码
    :return: ndArray
    """
    img = imageio.imread(io.BytesIO(base64.b64decode(img_base64_str)))
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 显示图片（调试）
    # cv2.imshow("reconstructed.jpg", cv2_img)
    # cv2.waitKey(0)
    return cv2_img
