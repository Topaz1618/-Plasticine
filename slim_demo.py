import cv2
import numpy as np
import math

import os
import cv2
import mathlib
import asyncio
from time import time, sleep

from keras.layers import *

from nnlib import nnlib
from facelib import S3FDExtractor, LandmarksExtractor


class Handler(object):
    def __init__(self):
        device_config = nnlib.DeviceConfig(cpu_only=True,
                                           force_gpu_idx=0,
                                           allow_growth=True)

        self.frame = 0

        # S3FD
        nnlib.import_all(device_config)
        S3FD_model_path = os.path.join('facelib', 'S3FD.h5')
        S3FD_model = nnlib.keras.models.load_model(S3FD_model_path)
        self.s3fd_model = S3FDExtractor(S3FD_model)

        nnlib.import_all(device_config)
        self.landmark_model = LandmarksExtractor(nnlib.keras)
        self.landmark_model.manual_init()

    def handle_image(self, im):
        self.frame += 1
        start_time = time()
        rects = self.s3fd_model.extract(im)
        print("extract time  >>>", time() - start_time)
        if len(rects) == 0:
            print("no lms")
        else:
            # cv2.rectangle(im, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 4)
            print("res", rects)
            lms = self.landmark_model.extract(im, rects[:1])
            return lms


# 双线性插值法
def BilinearInsert(src, x, y):
    try:
        _, _, c = src.shape
        if c == 3:
            x1 = int(x)
            x2 = x1 + 1
            y1 = int(y)
            y2 = y1 + 1
            part1 = src[y1, x1].astype(np.float) * (float(x2) - x) * (float(y2) - y)
            part2 = src[y1, x2].astype(np.float) * (x - float(x1)) * (float(y2) - y)
            part3 = src[y2, x1].astype(np.float) * (float(x2) - x) * (y - float(y1))
            part4 = src[y2, x2].astype(np.float) * (x - float(x1)) * (y - float(y1))
            insertValue = part1 + part2 + part3 + part4
            return insertValue.astype(np.int8)
    except Exception as e:
        print(e)


def localTranslationWarp(s, im, startX, startY, endX, endY, radius):
    """

    :param s: 力度
    :param im: 原图
    :param startX: 起始点 x
    :param startY: 起始点 y
    :param endX:   截止点 x
    :param endY:   截止点 y
    :param radius: 半径
    :return:   插值后图片
    """
    ddradius = float(radius * radius) # 顶部点和底部点直线距离
    copyImg = np.zeros(im.shape, np.uint8)
    copyImg = im.copy()

    ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY) # 顶部点和中心点直线距离
    H, W, C = im.shape
    for i in range(W - 1): # 循环每一帧像素
        for j in range(H - 1):
            if math.fabs(i - startX) > radius or math.fabs(j - startY) > radius:  # 变绝对值浮点数,先过滤掉肯定不对的
                # 循环到每帧项目的 x, y 和顶部点x, y 相减，大于计算出的半径的肯定就不用管了， 这个是在一定范围内进行插值
                continue

            distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)   # 计算出当前点和顶部点的直线距离
            if (distance < ddradius):
                # 再一次过滤不需要插值的部分（插值范围为半径为x的圆内）
                ratio = (ddradius - distance) / (ddradius - distance + ddmc) #
                ratio = s * ratio * ratio

                UX = i - ratio * (endX - startX)
                UY = j - ratio * (endY - startY)
                value = BilinearInsert(im, UX, UY)
                copyImg[j, i] = value
    return copyImg


def slim_face(im, s):
    """ 瘦脸
    :param im:
    :param s:
    :return:
    """
    landmarks = slim.handle_image(im)

    if len(landmarks) == 0:
        return

    lms = landmarks[0].tolist()
    left_landmark = lms[3]
    left_landmark_button = lms[5]
    right_landmark = lms[13]
    right_landmark_button = lms[15]
    endPt = lms[30]

    slim_left = math.sqrt(
        (left_landmark[0] - left_landmark_button[0]) * (left_landmark[0] - left_landmark_button[0]) +
        (left_landmark[1] - left_landmark_button[1]) * (left_landmark[1] - left_landmark_button[1]))

    slim_right = math.sqrt(
        (right_landmark[0] - right_landmark_button[0]) * (right_landmark[0] - right_landmark_button[0]) +
        (right_landmark[1] - right_landmark_button[1]) * (right_landmark[1] - right_landmark_button[1]))

    image = localTranslationWarp(s, im, left_landmark[0], left_landmark[1], endPt[0], endPt[1], slim_left)

    image = localTranslationWarp(s, image, right_landmark[0], right_landmark[1], endPt[0], endPt[1], slim_right)

    cv2.putText(image, "200 %", (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
    cv2.imwrite('./data_output/face1_200.jpg', image)


if __name__ == '__main__':
    slim = Handler()
    im = cv2.imread('./data_input/test0.jpg')
    slim_face(im, s=2)