import os
import cv2
import math
import numpy as np
from time import time

from nnlib import nnlib
from facelib import S3FDExtractor, LandmarksExtractor


class SlimFace(object):
    def __init__(self):
        self.frame = 0
        self.cp = 0
        self.rects = None
        self.lms_update = False
        self.landmarks = None
        self.slim_strength = {}
        self.slim_strength_copy = {}
        self.landmark_coordinate_list = {}
        self.slim_part_pixel_list = {}
        self.im_pixel_list = []
        device_config = nnlib.DeviceConfig(cpu_only=True, force_gpu_idx=0, allow_growth=True)

        # S3FD
        nnlib.import_all(device_config)
        self.s3fd_model = S3FDExtractor()
        # self.landmark_model = LandmarksExtractor(nnlib.keras, manual_init=True)

        self.landmark_model = LandmarksExtractor(nnlib.keras)


    def generate_face_pixel(self, r, pixel_coordinate, landmark_x, landmark_y, face_part):
        i, j = pixel_coordinate
        radius = float(r * r)
        distance = (i - landmark_x) * (i - landmark_x) + (j - landmark_y) * (j - landmark_y)
        if math.fabs(i - landmark_x) <= r and math.fabs(j - landmark_y) <= r:
            if (distance < radius):
                self.slim_part_pixel_list[face_part].append(pixel_coordinate)

    def get_coordinate(self, coordinate_list):
        endPt, left_landmark, left_landmark_button, right_landmark, right_landmark_button = coordinate_list
        left_r = math.sqrt(
            (left_landmark[0] - left_landmark_button[0]) * (left_landmark[0] - left_landmark_button[0]) +
            (left_landmark[1] - left_landmark_button[1]) * (left_landmark[1] - left_landmark_button[1]))

        if right_landmark is None and right_landmark_button is None:
            return [endPt, [left_r, left_landmark[0], left_landmark[1]], []]

        right_r = math.sqrt(
            (right_landmark[0] - right_landmark_button[0]) * (right_landmark[0] - right_landmark_button[0]) +
            (right_landmark[1] - right_landmark_button[1]) * (right_landmark[1] - right_landmark_button[1]))

        return [endPt, [left_r, left_landmark[0], left_landmark[1]], [right_r, right_landmark[0], right_landmark[1]]]

    def generate_landmark_coordinate(self, lms, l1, l2, r1, r2, end):
        left_landmark = lms[l1]
        left_landmark_button = lms[l2]
        endPt = lms[end]

        if not r1 and r2:
            return [endPt, left_landmark, left_landmark_button, None, None]

        right_landmark = lms[r1]
        right_landmark_button = lms[r2]

        return [endPt, left_landmark, left_landmark_button, right_landmark, right_landmark_button]

    def generate_slim_part_params(self, lms, face_part, rects=None):
        """

        :param lms: 关键点
        :param face_part:  变形区域
        :param rects:  忘了
        :return:
        """
        if face_part == "cheek":    # 脸颊
            coordinate_list = self.generate_landmark_coordinate(lms, 3, 5, 13, 11, 30)

        elif face_part == "humerus":    # 颧骨
            coordinate_list = self.generate_landmark_coordinate(lms, 1, 17, 15, 26, 27)

        elif face_part == "chin":   # 瘦下巴
            coordinate_list = self.generate_landmark_coordinate(lms, 5, 7, 11, 9, 33)

        elif face_part == "pull_chin":  # 拉伸下巴
            coordinate_list = self.generate_landmark_coordinate(lms, 8, 10, 0, 0, 57)

        elif face_part == "forehead":  # 瘦脑门
            left_landmark = [rects[0][0], rects[0][1] + math.fabs(rects[0][1] - lms[0][1]) / 2]
            right_landmark = [rects[0][2], rects[0][1] + math.fabs(rects[0][1] - lms[16][1]) / 2]
            left_landmark_button = lms[0]
            right_landmark_button = lms[16]
            endPt = [lms[21][0] + math.fabs(lms[21][0] - lms[22][0]) / 2, lms[21][1]]
            coordinate_list = [endPt, left_landmark, left_landmark_button, right_landmark, right_landmark_button]

        elif face_part == "pull_forehead":  # 拉伸脑门
            r = rects[0]
            left_landmark = [r[0] + math.fabs(r[2] - r[0]) / 2, r[1]]
            left_landmark_button = [r[0], r[1]]
            endPt = lms[30]
            coordinate_list = [endPt, left_landmark, left_landmark_button, None, None]


        landmark_coordinate = self.get_coordinate(coordinate_list)
        self.landmark_coordinate_list[face_part] = landmark_coordinate

        endPt, left_point, right_point = self.landmark_coordinate_list[face_part]
        left_r, left_startX, left_startY = left_point
        [self.generate_face_pixel(left_r, pixel, left_startX, left_startY, face_part) for pixel in self.im_pixel_list]

        if len(right_point) != 0:
            right_r, right_startX, right_startY = right_point
            [self.generate_face_pixel(right_r, pixel, right_startX, right_startY, face_part) for pixel in self.im_pixel_list]

    def set_slim_strength(self, cheek_strength, humerus_strength, chin_strength, forehead_strength, pull_chin_strength, pull_forehead_strength):
        self.slim_strength['cheek'] = [cheek_strength, cheek_strength]
        self.slim_strength['humerus'] = [humerus_strength, humerus_strength]
        self.slim_strength['chin'] = [chin_strength, chin_strength]
        self.slim_strength['forehead'] = [forehead_strength, forehead_strength]
        self.slim_strength['pull_chin'] = [pull_chin_strength, pull_chin_strength]
        self.slim_strength['pull_forehead'] = [pull_forehead_strength, pull_forehead_strength]

        self.slim_strength_copy['cheek'] = [cheek_strength, cheek_strength]
        self.slim_strength_copy['humerus'] = [humerus_strength, humerus_strength]
        self.slim_strength_copy['chin'] = [chin_strength, chin_strength]
        self.slim_strength_copy['forehead'] = [forehead_strength, forehead_strength]
        self.slim_strength_copy['pull_chin'] = [pull_chin_strength, pull_chin_strength]
        self.slim_strength_copy['pull_forehead'] = [pull_forehead_strength, pull_forehead_strength]


    def update_slim_part_params(self, rects, lms):
        self.slim_part_pixel_list = {'humerus': [], 'cheek': [], 'chin': [], 'forehead': [], 'pull_chin': [], 'pull_forehead': []}
        self.landmark_coordinate_list = {}
        self.generate_slim_part_params(lms, 'cheek')
        self.generate_slim_part_params(lms, 'humerus')
        self.generate_slim_part_params(lms, 'chin')
        self.generate_slim_part_params(lms, 'forehead', rects)
        self.generate_slim_part_params(lms, 'pull_chin')
        self.generate_slim_part_params(lms, 'pull_forehead', rects)
        return self.slim_part_pixel_list, self.landmark_coordinate_list

    def compare_rects_change(self, rects):
        try:
            # print("compare_rects_change", self.rects, rects)
            if len(self.rects) > 0 and len(rects) > 0:
                r, r1 = self.rects[0], rects[0]
                face_proportion = math.fabs(r[0] - r[2]) / self.w * 400
                ux, uy = math.fabs(r[0] - r1[0]), math.fabs(r[1] - r1[1])
                # print("r r1 ux uy face_proportion", r, r1, ux, uy, face_proportion)
                # update_status = (False if ux + uy < 30 or math.sqrt(ux * ux + uy * uy) < 60 else True)
                update_status = (False if ux + uy < face_proportion or math.sqrt(ux * ux + uy * uy) < face_proportion else True)
                # update_status = (False if ux + uy < 30 else True)
                return update_status
            else:
                return False
        except:
            print("wtf")
            pass

    def get_landmark(self, im):
        self.frame += 1
        if self.frame == 1:
            rects = self.s3fd_model.extract(im)
            if len(rects) > 0:
                lms = self.landmark_model.extract(im, rects[:1])
                current_landmarks = lms
                self.rects = rects
                self.landmarks = lms
                self.lms_update = True
        else:
            rects = self.s3fd_model.extract(im)
            if len(rects) > 0:
                update_status = self.compare_rects_change(rects)
                if not update_status:
                    lms = self.landmarks
                    current_landmarks = self.landmark_model.extract(im, rects[:1])
                    self.lms_update = False
                else:
                    print(" >>> get new lanmark")
                    lms = self.landmark_model.extract(im, rects[:1])
                    current_landmarks = lms
                    self.rects = rects
                    self.landmarks = lms
                    self.lms_update = True
        return rects, current_landmarks, lms, self.lms_update

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
            if len(rightPt) == 0:
                for pixel in self.slim_part_pixel_list[slim_part]:
                    i, j = pixel
                    c = self.slim_face(i, j, self.slim_strength[slim_part][0], im, copyImg, endPt, leftPt)
            else:
                for pixel in self.slim_part_pixel_list[slim_part]:
                    i, j = pixel
                    c = self.slim_face(i, j, self.slim_strength[slim_part][0], im, copyImg, endPt, leftPt)
                    cheek_im = self.slim_face(i, j, self.slim_strength[slim_part][1], im, c, endPt, rightPt)
            return copyImg
        except TypeError as e:
            return cheek_im

    def update_pixel_list(self, im):
        h, w, _ = im.shape
        [[self.im_pixel_list.append((i, j)) for j in range(int(h / 2) - 2)]
         for i in range(int(w / 2) - 2)]

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
                    cp = math.fabs(defult_strength + math.fabs(defult_strength / 4))
                # if cp - self.cp > 0.3:
                #     cp = self.cp - 0.3
                # elif cp - self.cp < -0.3:
                #     cp = self.cp + 0.3
                cp = -cp

            elif defult_strength < 0:
                cp = (f - 0.15) * 100 * 0.3
                print("get cp", cp, self.cp)
                if defult_strength + cp > - 0.3:
                    print("set min value")
                    cp = math.fabs(defult_strength + math.fabs(defult_strength / 4))
                if cp - self.cp > 0.3:
                    cp = self.cp + 0.3
                elif cp - self.cp < -0.3:
                    cp = self.cp - 0.3
            self.cp = cp
            print("self.cp", self.cp)
            if compare_face > 0:
                current_power = self.slim_strength_copy['cheek'][0] + cp
                self.slim_strength['cheek'][0] = current_power

            elif compare_face < 0:
                current_power = self.slim_strength_copy['cheek'][1]
                self.slim_strength['cheek'][1] = current_power + cp
        print("cheek >>> ", self.slim_strength['cheek'])

    def slim_handler(self, im):
        h, w, _ = im.shape
        self.w = int(w / 2)
        im = cv2.resize(im, (int(w / 2), int(h / 2)))
        rects, current_landmarks, landmarks, lms_update = self.get_landmark(im)
        lms = landmarks[0].tolist()
        lms2 = current_landmarks[0].tolist()
        (self.update_slim_part_params(rects, lms) if lms_update else False)
        self.change_slim_power(lms2, rects)
        cheek_im = self.localTranslationWarp(im, 'cheek')
        humerus_im = self.localTranslationWarp(cheek_im, 'humerus')
        chin_im = self.localTranslationWarp(humerus_im, 'chin')
        forehead_im = self.localTranslationWarp(chin_im, 'forehead')
        pull_chin_im = self.localTranslationWarp(forehead_im, 'pull_chin')
        pull_forehead_im = self.localTranslationWarp(pull_chin_im, 'pull_forehead')
        im = cv2.resize(pull_forehead_im, (w, h))
        return im


def put_frame():
    count = 0
    start_time = time()
    cap = cv2.VideoCapture('03-2.mp4')
    while cap.isOpened():
        print("put count", count)
        count += 1
        ret, im = cap.read()
        if count == 1:
            slim.set_slim_strength(cheek_strength=-2.0, humerus_strength=-0.2, chin_strength=1.5, forehead_strength=1, pull_chin_strength=1, pull_forehead_strength=0)
        if not ret or count > 1000:
            print(time() - start_time)
            break
        res = slim.slim_handler(im)
        out.write(res)


def put_img():
    im = cv2.imread('./data/.jpg')
    slim.set_slim_strength(cheek_strength=2, humerus_strength=2, chin_strength=3, forehead_strength=1, pull_chin_strength=1)
    res = slim.slim_handler(im)
    cv2.imwrite("aa.jpg", res)
    # print(res)

    idx  = 0
    for i in range(3):
        im = cv2.imread("./data_input/test{}.jpg".format(i))
        slim.set_slim_strength(cheek_strength=2, humerus_strength=2, chin_strength=3, forehead_strength=1,pull_chin_strength=1)
        res_im = slim.slim_handler(im)
        image = np.concatenate((im, res_im), axis=1)
        cv2.imwrite("./data_output/slim{}.jpg".format(i), image)
        cv2.waitKey(1)

        idx += 1


if __name__ == "__main__":
    file_list = os.listdir('.')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("n_ori_slim.avi", fourcc, 24.0, (1920, 1080))
    slim = SlimFace()
    # put_frame()
    put_img()