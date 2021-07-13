import cv2
import os

from nnlib import nnlib
from facelib import LandmarksExtractor, S3FDExtractor
import numpy as np
import math


class Handler(object):
    def __init__(self):
        device_config = nnlib.DeviceConfig(cpu_only=True,
                                           force_gpu_idx=0,
                                           allow_growth=True)
        self.frame = 0
        self.rects = None
        self.landmarks = None

        nnlib.import_all(device_config)
        S3FD_model_path = os.path.join('facelib', 'S3FD.h5')
        S3FD_model = nnlib.keras.models.load_model(S3FD_model_path)
        self.s3fd_model = S3FDExtractor(S3FD_model)

        nnlib.import_all(device_config)
        self.landmark_model = LandmarksExtractor(nnlib.keras)
        self.landmark_model.manual_init()

    def handle_image(self, im):
        self.frame += 1
        lms_update = False
        # if self.frame == 1:
        if self.frame < 5: # for test
            rects = self.s3fd_model.extract(im)
            print("rects >>> ", rects)
            # 画框
            r = rects[0]
            cv2.rectangle(im, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 4)

            lms = self.landmark_model.extract(im, rects[:1])

            # 画点
            f1 = lms[0]
            l = [3, 5, 13, 15, 30, 1, 17, 15, 26, 27, 5, 7, 11, 13, 33]
            for i in range(68):
                cv2.circle(im, (int(f1[i][0]), int(f1[i][1])), 2, (0, 0, 255), lineType=cv2.LINE_AA)
                cv2.putText(im, str(i), (int(f1[i][0]), int(f1[i][1])), 1, 1, (255, 255, 255), 1)

            cv2.imwrite(f"{self.frame}.jpg", im)
            self.rects = rects
            self.landmarks = lms
            lms_update = True
        else:
            rects = self.s3fd_model.extract(im)
            r = rects[0]
            r1 = self.rects[0]
            print(">>>>>", math.fabs(r[0] - r1[0]), math.fabs(r[1] - r1[1]))
            cv2.rectangle(im, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 4)
            cv2.rectangle(im, (r1[0], r1[1]), (r1[2], r1[3]), (255, 255, 255), 4)
            cv2.putText(im, "{} {}".format(math.fabs(r[0] - r1[0]), math.fabs(r[1] - r1[1])), (20, 25), 1, 1, (255, 255, 255), 1)


            f1 = self.landmarks[0]
            l = [3, 5, 13, 15, 30, 1, 17, 15, 26, 27, 5, 7, 11, 13, 33]
            for i in range(68):
                cv2.circle(im, (int(f1[i][0]), int(f1[i][1])), 2, (0, 0, 255), lineType=cv2.LINE_AA)
                cv2.putText(im, str(i), (int(f1[i][0]), int(f1[i][1])), 1, 1, (255, 255, 255), 1)
            # 稳定
            # print(rects[0], self.rects[0], type(rects))
            r1 = np.array(rects[0])
            r2 = np.array(self.rects[0])
            c = abs(np.sum(np.array(r1) - np.array(r2)))
            cv2.putText(im, "rects1:{} rects2:{} c:{} >>>> ".format(r1, r2, c), (5, 10), 1, 1, (255, 255, 255), 1)
            # print("current rects:{} current landmark:{} c:{} >>>> ".format(r1, r2, c))
            if c < 30:
                lms = self.landmarks
            else:
                print("\n get new lanmark", c)
                self.rects = rects
                lms = self.landmark_model.extract(im, rects[:1])
                self.landmarks = lms
                lms_update = True
                # self.landmarks = lms  # 测试时落下了
        return lms, lms_update, im


def put_frame():
    """　For test. """
    count = 0
    cap = cv2.VideoCapture('./media/jiangchao.mp4')
    while cap.isOpened():
        count += 1
        ret, im = cap.read()
        if not ret or ret > 2000:
            break
        h, w, _ = im.shape
        lms, lms_update, im = a.handle_image(im)
        out.write(im)
        cv2.imshow("iii", im)
        cv2.waitKey(1)

def put_img():
    im = cv2.imread("data_input/test_face1.png")
    idx  = 0
    while True:
        if idx > 1:
            break

        lms, lms_update, im = a.handle_image(im)
        cv2.imshow("iii", im)
        cv2.waitKey(2)
        idx += 1


def put_img_new(num):
    idx  = 0
    while True:
        if idx > num:
            break

        img_name = f"data_input/test_face{idx+1}.png"
        im = cv2.imread(img_name)
        lms, lms_update, im = face_detection.handle_image(im)
        cv2.imshow("Show topaz", im)
        cv2.waitKey(1)
        idx += 1

if __name__ == "__main__":
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("jvideo.mp4", fourcc, 24.0, (1920, 1080))
    face_detection = Handler()
    # put_frame()
    # put_img()

    put_img_new(4)


