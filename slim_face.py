import os
import cv2
import glob
import math
from time import sleep, time
import matplotlib
matplotlib.use('agg')

import numpy as np
from time import time

from nnlib import nnlib
import matplotlib.pyplot as plt
from facelib import S3FDExtractor, LandmarksExtractor


class SlimFace(object):
    def __init__(self):
        self.frame = 0
        self.rects = None
        self.lms_update = False
        self.landmarks = None
        self.slim_strength = {}
        self.slim_strength_copy = {}
        self.landmark_coordinate_list = {}
        self.slim_part_pixel_list = {}
        self.im_pixel_list = []
        self.cp = 0
        device_config = nnlib.DeviceConfig(cpu_only=False,
                                           force_gpu_idx=0,
                                           allow_growth=True)

        # S3FD
        nnlib.import_all(device_config)
        S3FD_model_path = os.path.join('facelib', 'S3FD.h5')
        S3FD_model = nnlib.keras.models.load_model(S3FD_model_path)
        self.s3fd_model = S3FDExtractor(S3FD_model)

        nnlib.import_all(device_config)
        self.landmark_model = LandmarksExtractor(nnlib.keras)
        self.landmark_model.manual_init()

    def generate_landmark_coordinate(self, lms, l1, l2, r1, r2, end):
        """

        :param lms: 所有关键点位置
        :param l1: 左脸顶部点坐标
        :param l2: 左脸底部点坐标
        :param r1: 右脸顶部点坐标
        :param r2: 右脸底部点坐标
        :param end: 忘了
        :return: 左脸横纵坐标和半径， 右脸横纵坐标和半径
        """
        left_landmark = lms[l1]
        left_landmark_button = lms[l2]
        right_landmark = lms[r1]
        right_landmark_button = lms[r2]
        endPt = lms[end]

        left_r = math.sqrt(
            (left_landmark[0] - left_landmark_button[0]) * (left_landmark[0] - left_landmark_button[0]) +
            (left_landmark[1] - left_landmark_button[1]) * (left_landmark[1] - left_landmark_button[1]))

        right_r = math.sqrt(
            (right_landmark[0] - right_landmark_button[0]) * (right_landmark[0] - right_landmark_button[0]) +
            (right_landmark[1] - right_landmark_button[1]) * (right_landmark[1] - right_landmark_button[1]))

        return [endPt, [left_r, left_landmark[0], left_landmark[1]], [right_r, right_landmark[0], right_landmark[1]]]

    def generate_face_pixel(self, r, pixel_coordinate, landmark_x, landmark_y, face_part):
        """

        :param r:
        :param pixel_coordinate:
        :param landmark_x:
        :param landmark_y:
        :param face_part:
        :return:
        """
        i, j = pixel_coordinate
        radius = float(r * r)
        distance = (i - landmark_x) * (i - landmark_x) + (j - landmark_y) * (j - landmark_y)
        if math.fabs(i - landmark_x) <= r and math.fabs(j - landmark_y) <= r:
            if (distance < radius):
                self.slim_part_pixel_list[face_part].append(pixel_coordinate)

    def generate_slim_part_params(self, lms, face_part):
        if face_part == "cheek":
            landmark_coordinate = self.generate_landmark_coordinate(lms, 3, 5, 13, 11, 29) # Original key Point: 3, 5, 13, 15, 30
        elif face_part == "humerus":
            landmark_coordinate = self.generate_landmark_coordinate(lms, 1, 17, 15, 26, 27)
        elif face_part == "chin":
            landmark_coordinate = self.generate_landmark_coordinate(lms, 5, 7, 11, 9, 33) # Original key Point: 5, 7, 11, 13, 33

        self.landmark_coordinate_list[face_part] = landmark_coordinate

        endPt, left_point, right_point = self.landmark_coordinate_list[face_part]
        left_r, left_startX, left_startY = left_point
        right_r, right_startX, right_startY = right_point

        [self.generate_face_pixel(left_r, pixel, left_startX, left_startY, face_part) for pixel in self.im_pixel_list]
        [self.generate_face_pixel(right_r, pixel, right_startX, right_startY, face_part) for pixel in self.im_pixel_list]

    def set_slim_strength(self, cheek_strength, humerus_strength, chin_strength):
        self.slim_strength_copy['cheek'] = [cheek_strength, cheek_strength]
        self.slim_strength_copy['humerus'] = [humerus_strength, humerus_strength]
        self.slim_strength_copy['chin'] = [chin_strength, chin_strength]

        self.slim_strength['cheek'] = [cheek_strength, cheek_strength]
        self.slim_strength['humerus'] = [humerus_strength, humerus_strength]
        self.slim_strength['chin'] = [chin_strength, chin_strength]

    def update_slim_part_params(self, lms):
        self.slim_part_pixel_list = {'humerus': [], 'cheek': [], 'chin': []}
        self.landmark_coordinate_list = {}
        self.generate_slim_part_params(lms, 'cheek')
        self.generate_slim_part_params(lms, 'humerus')
        self.generate_slim_part_params(lms, 'chin')
        return self.slim_part_pixel_list, self.landmark_coordinate_list

    def compare_rects_change(self, rects):
        # r, r1 = self.rects[0], rects[0]
        # ux, uy = math.fabs(r[0] - r1[0]), math.fabs(r[1] - r1[1])
        # update_status = (False if ux + uy < 30 or math.sqrt(ux * ux + uy * uy) < 60 else True)
        # return update_status
        if len(self.rects) > 0 and len(rects) > 0:
            r, r1 = self.rects[0], rects[0]
            face_proportion = math.fabs(r[0] - r[2]) / 960 * 300
            ux, uy = math.fabs(r[0] - r1[0]), math.fabs(r[1] - r1[1])
            print("r r1 ux uy face_proportion", r, r1, ux, uy, face_proportion)
            # update_status = (False if ux + uy < 30 or math.sqrt(ux * ux + uy * uy) < 60 else True)
            update_status = (
                False if ux + uy < face_proportion or math.sqrt(ux * ux + uy * uy) < face_proportion else True)
            # update_status = (False if ux + uy < 30 else True)
            print("update_status")
            return update_status

    def get_landmark(self, im):
        self.frame += 1
        if self.frame == 1:
            rects = self.s3fd_model.extract(im)
            print("!!!!!", len(im), im.shape, rects)
            lms = self.landmark_model.extract(im, rects[:1])
            print(lms)
            c_lms = lms
            self.rects = rects
            self.landmarks = lms
            self.lms_update = True

        else:
            print("!!!!")
            rects = self.s3fd_model.extract(im)
            update_status = self.compare_rects_change(rects)
            if not update_status:
                c_lms = self.landmark_model.extract(im, rects[:1])
                lms = self.landmarks
                self.lms_update = False
            else:
                print("get new lanmark")
                lms = self.landmark_model.extract(im, rects[:1])
                c_lms = lms
                self.rects = rects
                self.landmarks = lms
                self.lms_update = True

        return rects, c_lms, lms, self.lms_update

    def BilinearInsert(self, im, x, y):
        try:
            x1, y1 = int(x), int(y)
            x2, y2 = x1 + 1, y1 + 1
            part1 = im[y1, x1].astype(np.float) * (float(x2) - x) * (float(y2) - y)
            part2 = im[y1, x2].astype(np.float) * (x - float(x1)) * (float(y2) - y)
            part3 = im[y2, x1].astype(np.float) * (float(x2) - x) * (y - float(y1))
            part4 = im[y2, x2].astype(np.float) * (x - float(x1)) * (y - float(y1))
            insertValue = part1 + part2 + part3 + part4
            return insertValue.astype(np.int8)
        except Exception as e:
            print(e)

    def slim_face(self, i, j, s, im, copy_im, landmark_center_point, landmark_coordinate):
        endX, endY = landmark_center_point
        r, landmark_x, landmark_y = landmark_coordinate
        dradius = float(r * r)
        ddmc = (endX - landmark_x) * (endX - landmark_x) + (endY - landmark_y) * (endY - landmark_y)
        distance = (i - landmark_x) * (i - landmark_x) + (j - landmark_y) * (j - landmark_y)
        if (distance < dradius):
            ratio = (dradius - distance) / (dradius - distance + ddmc)
            ratio = s * ratio * ratio
            UX, UY = i - ratio * (endX - landmark_x), j - ratio * (endY - landmark_y)
            copy_im[j, i] = self.BilinearInsert(im, UX, UY)
        return copy_im

    def localTranslationWarp(self, im, slim_part):
        try:
            copyImg = im.copy()
            endPt, leftPt, rightPt = self.landmark_coordinate_list[slim_part]
            if slim_part == 'cheek':
                print(slim_part, self.slim_strength[slim_part])
            for pixel in self.slim_part_pixel_list[slim_part]:
                i, j = pixel
                c = self.slim_face(i, j, self.slim_strength[slim_part][0], im, copyImg, endPt, leftPt)
                cheek_im = self.slim_face(i, j, self.slim_strength[slim_part][1], im, c, endPt, rightPt)
            return copyImg
        except TypeError as e:
            return cheek_im

    def change_slim_power(self, lms, rects):
        x1, x2 = rects[0][0], rects[0][2]
        l = lms[3]
        r = lms[13]
        c = lms[30]
        r_ux, r_uy = math.fabs(r[0] - c[0]), math.fabs(r[1] - c[1])
        l_ux, l_uy = math.fabs(l[0] - c[0]), math.fabs(l[1] - c[1])
        r_face = math.sqrt(r_ux * r_ux + r_uy * r_uy)
        l_face = math.sqrt(l_ux * l_ux + l_uy * l_uy)
        compare_face = r_face - l_face
        face_width = math.fabs(x1 - x2)
        f = math.fabs(compare_face) / face_width
        if f > 0.15:
            cp = 0
            defult_strength = self.slim_strength_copy['cheek'][0]
            if defult_strength > 0:
                cp = - (f - 0.15) * 100 * 0.3
                if defult_strength + cp < 0.3:
                    cp = - math.fabs(defult_strength + math.fabs(defult_strength / 4))
                if math.fabs(cp - self.cp) > 0.3:
                    cp = self.cp - 0.3

            elif defult_strength < 0:
                cp = (f - 0.15) * 100 * 0.3
                if defult_strength + cp > - 0.3:
                    cp = math.fabs(defult_strength + math.fabs(defult_strength / 4))
                if cp - self.cp > 0.3:
                    cp = self.cp + 0.3
            self.cp = cp
            print("left >>>", cp)
            if compare_face > 0:
                current_power = self.slim_strength_copy['cheek'][0] + cp
                self.slim_strength['cheek'][0] = current_power
            elif compare_face < 0:
                print("wtf")
                current_power = self.slim_strength_copy['cheek'][1]
                self.slim_strength['cheek'][1] = current_power + cp

    def slim_handler(self, im):
        h, w, _ = im.shape
        im = cv2.resize(im, (int(w / 2), int(h / 2)))
        [[self.im_pixel_list.append((i, j)) for j in range(int(h / 2) - 2)] for i in range(int(w / 2) - 2)]
        rects, c_lms, landmarks, lms_update = self.get_landmark(im)
        lms = landmarks[0].tolist()
        (self.update_slim_part_params(lms) if lms_update else False)
        lms2 = c_lms[0].tolist()
        self.change_slim_power(lms2, rects)
        cheek_im = self.localTranslationWarp(im, 'cheek')
        humerus_im = self.localTranslationWarp(cheek_im, 'humerus')
        chin_im = self.localTranslationWarp(humerus_im, 'chin')
        im = cv2.resize(chin_im, (w, h))
        cv2.imwrite("A.jpg", im)
        return im


def put_frame():
    count = 0
    start_time = time()
    cap = cv2.VideoCapture('./media/jcdemo.mp4')
    while cap.isOpened():
        print("put count", count)
        count += 1
        ret, im = cap.read()
        if count == 1:
            slim.set_slim_strength(cheek_strength=-2.0, humerus_strength=-0.2, chin_strength=1.5)
            # slim.slim_strength_copy = slim.slim_strength
        if not ret:
            print(time() - start_time)
            break
        res = slim.slim_handler(im)
        cv2.imwrite("data_output/slim/{}.jpg".format(count), res)
        # out.write(res)


def put_img():
    idx  = 0
    for i in range(5):
        im = cv2.imread("data_input/test_face{}.png".format(i+1))
        slim.set_slim_strength(
            cheek_strength=1.6,
            humerus_strength=0.2,
            chin_strength=2.2)
        res_im = slim.slim_handler(im)
        image = np.concatenate((im, res_im), axis=1)
        cv2.imwrite("./data_output/slim_face{}.jpg".format(i), image)

        cv2.imshow("slim face", image)
        sleep(1)
        cv2.waitKey(1)

        idx += 1



if __name__ == "__main__":
    # file_list = os.listdir('.')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter("ssslim.avi", fourcc, 24.0, (1920, 1080))
    slim = SlimFace()
    # put_frame()
    put_img()
